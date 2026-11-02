#!/usr/bin/env python3
"""Tests for ranger's prompt building, fallback tool-call parser and turn loop.

Run with:  python3 -m pytest tests -q

The chat endpoint is faked throughout: a canned response per call, with every
request payload recorded. Nothing here opens a socket, touches a real service
or depends on where pytest was started — every repository is a throwaway
directory under tempfile.
"""

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import httpx   # noqa: E402
import ranger  # noqa: E402


# --- canned model responses ---

def assistant_text(text):
    """A chat completion whose message is plain content, no tool calls."""
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def assistant_tool_call(name, arguments, tc_id="call_0"):
    """A chat completion carrying a native OpenAI-style tool call."""
    return {"choices": [{"message": {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": tc_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }}]}


class FakeModel:
    """Stands in for ranger.call_model: hands out canned responses and keeps
    every request payload it was given."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.payloads = []

    def __call__(self, messages, tools=None):
        self.payloads.append({"messages": list(messages), "tools": tools})
        return next(self.responses)


class ParseToolCallsTests(unittest.TestCase):
    """The <|tool_call_start|>[Func(args)]<|tool_call_end|> fallback format the
    docstring promises (bug:d4b7df1)."""

    def parse(self, content):
        return ranger.parse_tool_calls_from_content(content)

    def test_a_single_call_is_parsed_with_its_name_and_arguments(self):
        calls = self.parse(
            "<|tool_call_start|>[Grep(pattern='foo', path='/x')]<|tool_call_end|>")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "Grep")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"pattern": "foo", "path": "/x"})

    def test_several_calls_in_one_block_come_out_in_order_with_unique_ids(self):
        calls = self.parse(
            '<|tool_call_start|>[Read(path=\'/a\', offset=2), Grep(pattern="x")]'
            '<|tool_call_end|>')
        self.assertEqual([c["function"]["name"] for c in calls], ["Read", "Grep"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"path": "/a", "offset": 2})
        self.assertEqual(json.loads(calls[1]["function"]["arguments"]),
                         {"pattern": "x"})
        self.assertEqual(len({c["id"] for c in calls}), 2)

    def test_arguments_come_out_with_the_type_they_were_written_with(self):
        calls = self.parse(
            "<|tool_call_start|>[Grep(pattern='foo', head_limit=3, context=0.5, "
            "flag=True, off=False, missing=None)]<|tool_call_end|>")
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"pattern": "foo", "head_limit": 3, "context": 0.5,
                          "flag": True, "off": False, "missing": None})

    def test_content_without_a_block_yields_no_calls(self):
        for content in ("", "I searched and found nothing.",
                        "<|tool_call_start|>[", "[Grep(pattern='x')]",
                        "<|tool_call_start|>[]<|tool_call_end|>"):
            self.assertEqual(self.parse(content), [])

    def test_text_around_the_block_is_not_part_of_the_call(self):
        calls = self.parse(
            "Let me look for it. "
            "<|tool_call_start|>[Read(path='/a')]<|tool_call_end|> That was it.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "Read")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"path": "/a"})


class BuildSystemPromptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name) / "work"
        self.work.mkdir()

    def test_the_prompt_names_the_workspace_and_its_sorted_contents(self):
        (self.work / "zeta.py").write_text("")
        (self.work / "alpha.py").write_text("")
        prompt = ranger.build_system_prompt(str(self.work))
        self.assertIn(f"Workspace: {self.work}", prompt)
        listing = prompt.split("Top-level contents:\n", 1)[1]
        self.assertLess(listing.index("alpha.py"), listing.index("zeta.py"))

    def test_the_listing_is_capped_at_fifty_entries(self):
        for n in range(55):
            (self.work / f"e{n:02d}").write_text("")
        prompt = ranger.build_system_prompt(str(self.work))
        listing = prompt.split("Top-level contents:\n", 1)[1]
        self.assertIn("e49", listing)
        self.assertNotIn("e50", listing)

    def test_the_prompt_requires_absolute_paths(self):
        self.assertIn("Always use absolute paths",
                      ranger.build_system_prompt(str(self.work)))

    def test_the_definition_graph_rule_only_appears_with_graph_tools(self):
        with_graph = ranger.build_system_prompt(str(self.work), graph_tools=True)
        without_graph = ranger.build_system_prompt(str(self.work), graph_tools=False)
        self.assertIn("definition graph", with_graph)
        self.assertIn("Where NAME", with_graph)
        self.assertIn("Uses NAME", with_graph)
        self.assertNotIn("definition graph", without_graph)
        self.assertNotIn("Where NAME", without_graph)


class RunToolTests(unittest.TestCase):
    """Dispatch only: the tools' own behaviour lives in tests/test_tools.py."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name) / "work"
        self.work.mkdir()

    def test_an_unknown_tool_is_an_error_string_not_an_exception(self):
        self.assertEqual(ranger.run_tool("WebSearch", {}, str(self.work)),
                         "Error: unknown tool 'WebSearch'")

    def test_lookup_is_case_insensitive_and_still_passes_work_dir(self):
        # The model writes the name the way it likes and the path relative to
        # the repository; both must still land on the right tool in work_dir.
        (self.work / "note.txt").write_text("one content line\n")
        out = ranger.run_tool("read", {"path": "note.txt"}, str(self.work))
        self.assertIn("1|one content line", out)


class MainLoopTestCase(unittest.TestCase):
    """A temp repository with real files, and every service faked."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        (self.repo / "pkg").mkdir(parents=True)
        (self.repo / "beta.txt").write_text("beta needle line\n")
        (self.repo / "pkg" / "alpha.txt").write_text("alpha needle line\n")

    def run_main(self, model, *extra_args, candidates="", structure="",
                 directory=None):
        """Run ranger.main with the model and the retrieval layer faked.
        Returns (exit_code, stdout, stderr); 0 means main returned normally."""
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(ranger, "call_model", model), \
             mock.patch.object(ranger, "trailblazer_ready", lambda work_dir: None), \
             mock.patch.object(ranger, "index_candidates",
                               lambda *a, **k: (candidates, structure)), \
             mock.patch.object(ranger, "graph_tool_functions", lambda *a, **k: {}), \
             mock.patch.object(sys, "argv",
                               ["ranger.py", "-C", str(directory or self.repo),
                                "-q", "where is the needle", *extra_args]), \
             contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                ranger.main()
            except SystemExit as e:
                return e.code, out.getvalue(), err.getvalue()
        return 0, out.getvalue(), err.getvalue()

    # --- one exploration turn, end to end ---

    def test_a_tool_call_runs_and_its_result_goes_back_to_the_model(self):
        target = self.repo / "beta.txt"
        model = FakeModel([
            assistant_tool_call("Read", {"path": str(target)}),
            assistant_text("beta.txt:1 holds the needle"),
        ])
        code, out, _err = self.run_main(model)
        self.assertEqual(code, 0)
        self.assertEqual(len(model.payloads), 2)
        history = model.payloads[1]["messages"]
        self.assertEqual([m["role"] for m in history],
                         ["system", "user", "assistant", "tool"])
        tool_msg = history[-1]
        self.assertIn("1|beta needle line", tool_msg["content"])
        self.assertEqual(tool_msg["tool_call_id"], "call_0")
        self.assertIn("beta.txt:1 holds the needle", out)

    def test_a_tool_call_written_in_the_content_fallback_is_executed(self):
        model = FakeModel([
            assistant_text(
                "I should list the files. "
                "<|tool_call_start|>[Glob(pattern='*.txt', directory='.')]"
                "<|tool_call_end|>"),
            assistant_text("pkg/alpha.txt and beta.txt hold the needle"),
        ])
        code, out, _err = self.run_main(model)
        self.assertEqual(code, 0)
        self.assertEqual(len(model.payloads), 2)
        tool_msgs = [m for m in model.payloads[1]["messages"]
                     if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("alpha.txt", tool_msgs[0]["content"])
        self.assertIn("beta.txt", tool_msgs[0]["content"])
        self.assertEqual(tool_msgs[0]["tool_call_id"], "call_Glob_0")
        self.assertIn("pkg/alpha.txt and beta.txt hold the needle", out)

    def test_the_final_answer_is_printed_to_stdout(self):
        model = FakeModel([assistant_text("pkg/alpha.txt:1 — the needle lives here")])
        code, out, _err = self.run_main(model)
        self.assertEqual(code, 0)
        self.assertEqual(len(model.payloads), 1)
        self.assertEqual(out.strip(), "pkg/alpha.txt:1 — the needle lives here")

    def test_the_structure_appendix_follows_the_answer_when_present(self):
        model = FakeModel([assistant_text("pkg/alpha.txt:1 the answer")])
        code, out, _err = self.run_main(
            model, candidates="pkg/alpha.txt:1-1 the needle line",
            structure="pkg/alpha.txt defines needle()")
        self.assertEqual(code, 0)
        self.assertIn(
            "pkg/alpha.txt:1 the answer\n\n"
            "--- structure (from the index, not the model) ---\n"
            "pkg/alpha.txt defines needle()",
            out)

    def test_no_appendix_heading_when_the_index_says_nothing(self):
        model = FakeModel([assistant_text("just the answer")])
        code, out, _err = self.run_main(model, structure="")
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "just the answer")

    def test_max_turns_ends_with_one_forced_final_request(self):
        tool_turn = assistant_tool_call("Grep", {"pattern": "needle"})
        model = FakeModel([
            tool_turn, tool_turn,
            assistant_text("final findings at pkg/alpha.txt:1"),
        ])
        code, out, _err = self.run_main(
            model, "--max-turns", "2",
            structure="pkg/alpha.txt defines needle()")
        self.assertEqual(code, 0)
        # Two exploration turns, then the one forced answer request.
        self.assertEqual(len(model.payloads), 3)
        last = model.payloads[2]["messages"]
        self.assertEqual(last[-1],
                         {"role": "user",
                          "content": "Provide your findings based on what you've searched so far."})
        # Both tool answers are in the history the final request carries.
        self.assertEqual([m["role"] for m in last].count("tool"), 2)
        self.assertIn("final findings at pkg/alpha.txt:1", out)
        self.assertIn("--- structure (from the index, not the model) ---", out)
        self.assertIn("pkg/alpha.txt defines needle()", out)

    def test_a_directory_that_does_not_exist_exits_1_before_asking_the_model(self):
        model = FakeModel([assistant_text("never reached")])
        code, _out, err = self.run_main(model, directory=str(self.root / "nope"))
        self.assertEqual(code, 1)
        self.assertIn("is not a directory", err)
        self.assertEqual(model.payloads, [])


class CallModelTests(unittest.TestCase):
    """The chat call over a faked httpx.Client: no socket is ever opened."""

    @staticmethod
    def client_returning(response):
        """A patch for ranger.httpx.Client that answers with `response`, plus
        the dict in which the fake records the request it was given."""
        recorded = {}

        class FakeClient:
            def __init__(self, *a, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def post(self, url, json=None):
                recorded["url"] = url
                recorded["payload"] = json
                return response

        return mock.patch.object(ranger.httpx, "Client", FakeClient), recorded

    @staticmethod
    def fake_response(status, **kw):
        request = httpx.Request("POST", "http://fake/v1/chat/completions")
        return httpx.Response(status, request=request, **kw)

    def test_success_returns_the_decoded_json_and_sends_the_payload(self):
        served = {"choices": [{"message": {"content": "hi"}}]}
        patcher, recorded = self.client_returning(self.fake_response(200, json=served))
        messages = [{"role": "user", "content": "q"}]
        with patcher:
            out = ranger.call_model(messages)
        self.assertEqual(out, served)
        self.assertEqual(recorded["url"],
                         f"{ranger.SERVER_URL}/v1/chat/completions")
        payload = recorded["payload"]
        self.assertEqual(payload["model"], ranger.MODEL)
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(payload["tools"], ranger.TOOL_DEFS)

    def test_an_error_status_reports_to_stderr_and_raises(self):
        patcher, _recorded = self.client_returning(
            self.fake_response(500, text="router exploded"))
        err = io.StringIO()
        with patcher, contextlib.redirect_stderr(err):
            with self.assertRaises(httpx.HTTPStatusError):
                ranger.call_model([{"role": "user", "content": "q"}])
        self.assertIn("ERROR 500", err.getvalue())
        self.assertIn("router exploded", err.getvalue())


if __name__ == "__main__":
    unittest.main()
