#!/usr/bin/env python3
"""Tests for trailblazer.py and for ranger's choice of candidate source.

Run with:  python3 -m unittest discover -s tests

Trailblazer is replaced by a fake HTTP service on a random loopback port, so
nothing here starts, stops or depends on the real one.
"""

import contextlib
import io
import json
import subprocess
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import index           # noqa: E402
import mini_ranger     # noqa: E402
import ranger          # noqa: E402
import trailblazer     # noqa: E402

HEAD_BOB = {
    "id": 7,
    "kind": "function",
    "qualified": "headBob",
    "path": "assets/headBob.ts",
    "start_line": 10,
    "end_line": 24,
    "signature": "headBob(t: number)",
    "stale": False,
}
STEP_BOB = {
    "id": 8,
    "kind": "function",
    "qualified": "stepBob",
    "path": "assets/stepBob.ts",
    "start_line": 5,
    "end_line": 30,
    "signature": "stepBob(dt: number)",
    "stale": False,
}
BOB_OFFSET = {
    "id": 9,
    "kind": "const",
    "qualified": "bobOffset",
    "path": "assets/headBob.ts",
    "start_line": 3,
    "end_line": 3,
    "signature": "bobOffset = 0.12",
    "stale": False,
}

SEARCH_RESULT = [
    {"node": HEAD_BOB, "score": 0.83, "via": "vector"},
    {"node": STEP_BOB, "score": 0.79, "via": "vector"},
    {"node": BOB_OFFSET, "score": 1.0, "via": "exact"},
]

USES_RESULT = [
    {"node": STEP_BOB, "target": HEAD_BOB, "kind": "calls", "line": 42, "uncertain": True},
    {"node": BOB_OFFSET, "target": HEAD_BOB, "kind": "references", "line": 18, "uncertain": False},
]

STATUS_RESULT = {
    "files": 160,
    "nodes": 912,
    "edges": 2401,
    "pending_vectors": 0,
    "stale_files": 0,
    "following": "events",
    "embedding": "running",
}


class Handler(BaseHTTPRequestHandler):
    """Answers from the tables on the server object, and records what it was asked."""

    protocol_version = "HTTP/1.1"

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = {}
        op = self.path.strip("/")
        fake = cast("FakeServer", self.server)
        fake.calls.append((op, body))
        status, payload = fake.answer(op, body)
        out = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, format, *args):
        pass


class FakeServer(ThreadingHTTPServer):
    """A Trailblazer that answers from a table and remembers what it was asked."""

    def __init__(self, address, answer):
        super().__init__(address, Handler)
        self.calls = []
        self.answer = answer


def start_fake(answer):
    """Run a fake Trailblazer on a free loopback port; returns (url, server)."""
    server = FakeServer(("127.0.0.1", 0), answer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_address[1]}", server


def watched_answer(op, body):
    if op == "status":
        return 200, STATUS_RESULT
    if op == "search":
        return 200, SEARCH_RESULT
    if op == "uses":
        # One entry per edge into the definition(s) the name resolves to.
        if "id" in body:
            return 200, [e for e in USES_RESULT if e["target"]["id"] == body["id"]]
        return 200, [e for e in USES_RESULT if matches(e["target"]["qualified"], body.get("name"))]
    if op == "where":
        # Trailblazer answers where with bare node summaries, matched on the
        # qualified name or on its last segment.
        return 200, [n for n in (HEAD_BOB, STEP_BOB, BOB_OFFSET) if matches(n["qualified"], body.get("name"))]
    return 404, {"error": "unknown operation"}


def matches(qualified, asked):
    """Trailblazer falls back from a qualified name to its last segment."""
    if not asked:
        return False
    asked = asked.split("/")[0]
    return qualified == asked or qualified.split(".")[-1] == asked


def exploding_sync(*args, **kwargs):
    raise AssertionError("index.sync must not run when Trailblazer watches the repo")


class FakeWaiter:
    """Stands in for the inotifywait process: stderr lines, then an exit code.

    The waiter is what start_service() watches for "Watches established", so a
    fake yields that line (or fails before it) and wait() returns at once: no
    real inotify, no real backstop sleep, whatever the host's limits allow."""

    def __init__(self, lines, wait_code=0):
        self.stderr = iter(lines)
        self.wait_code = wait_code
        self.killed = False

    def wait(self):
        return self.wait_code

    def kill(self):
        self.killed = True


class TrailblazerClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name)
        self.repo = Path(self.tmp.name) / "repo"
        (self.repo / "assets").mkdir(parents=True)
        self.lines = [f"// line {n}" for n in range(1, 41)]
        (self.repo / "assets" / "headBob.ts").write_text("\n".join(self.lines) + "\n")
        (self.repo / "assets" / "stepBob.ts").write_text("\n".join(self.lines) + "\n")
        self.env = mock.patch.dict(
            "os.environ", {"TRAILBLAZER_STATE_DIR": str(self.state)}
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def write_endpoint(self, url, token="t0ken"):
        (self.state / "endpoint.json").write_text(json.dumps({"url": url, "token": token}))

    def serve(self, answer=watched_answer):
        url, server = start_fake(answer)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return url, server

    # --- call() and watched(): every failure is None, not an exception ---

    def test_no_endpoint_file_means_none(self):
        self.assertIsNone(trailblazer.call("status", {"repo": str(self.repo)}))
        self.assertIsNone(trailblazer.watched(self.repo))

    def test_unreadable_endpoint_file_means_none(self):
        (self.state / "endpoint.json").write_text("{not json")
        self.assertIsNone(trailblazer.watched(self.repo))

    def test_connection_refused_means_none(self):
        self.write_endpoint("http://127.0.0.1:9")  # nothing listens here
        self.assertIsNone(trailblazer.call("status", {"repo": str(self.repo)}))
        self.assertIsNone(trailblazer.watched(self.repo))

    def test_not_watched_is_its_own_answer(self):
        url, _server = self.serve(lambda op, body: (409, {"error": "not watched"}))
        self.write_endpoint(url)
        self.assertIs(trailblazer.call("status", {"repo": str(self.repo)}), trailblazer.NOT_WATCHED)
        self.assertIsNone(trailblazer.watched(self.repo))

    def test_error_status_means_none(self):
        url, _server = self.serve(lambda op, body: (401, {"error": "unauthorized"}))
        self.write_endpoint(url, token="wrong")
        self.assertIsNone(trailblazer.watched(self.repo))

    def test_watched_returns_the_status_dict(self):
        url, server = self.serve()
        self.write_endpoint(url)
        status = trailblazer.watched(self.repo) or {}
        self.assertEqual(status.get("nodes"), STATUS_RESULT["nodes"])
        op, body = server.calls[0]
        self.assertEqual(op, "status")
        self.assertEqual(body["repo"], str(self.repo))

    # --- search_hits(): index.py's hit shape, text read from disk ---

    def test_search_hits_convert_nodes_to_hits(self):
        url, _server = self.serve()
        self.write_endpoint(url)
        hits = trailblazer.search_hits(self.repo, "head bob")
        self.assertEqual([h["path"] for h in hits], ["assets/headBob.ts", "assets/stepBob.ts", "assets/headBob.ts"])
        first = hits[0]
        self.assertEqual((first["start"], first["end"]), (10, 24))
        self.assertAlmostEqual(first["score"], 0.83)
        self.assertEqual(first["text"], "\n".join(self.lines[9:24]))
        self.assertEqual(hits[2]["start"], 3)
        self.assertEqual(hits[2]["text"], self.lines[2])

    def test_search_hits_of_a_missing_file_keep_their_range(self):
        url, _server = self.serve()
        self.write_endpoint(url)
        (self.repo / "assets" / "stepBob.ts").unlink()
        hits = trailblazer.search_hits(self.repo, "head bob")
        self.assertEqual(hits[1]["text"], "")

    def test_search_hits_are_empty_when_the_service_is_down(self):
        self.write_endpoint("http://127.0.0.1:9")
        self.assertEqual(trailblazer.search_hits(self.repo, "head bob"), [])

    # --- structure_text(): users of the best hits, then named symbols ---

    def test_structure_text_names_the_best_hits_and_their_users(self):
        url, _server = self.serve()
        self.write_endpoint(url)
        hits = trailblazer.search_hits(self.repo, "head bob")
        text = trailblazer.structure_text(self.repo, hits, "head bob")
        self.assertIn("headBob", text)
        self.assertIn("stepBob", text)
        self.assertIn("assets/stepBob.ts:42", text)
        self.assertIn("uncertain", text)

    def test_structure_text_reports_named_symbols(self):
        url, _server = self.serve()
        self.write_endpoint(url)
        hits = trailblazer.search_hits(self.repo, "head bob")
        text = trailblazer.structure_text(self.repo, hits, "where is `stepBob` used?")
        self.assertIn("stepBob", text)
        self.assertIn("declared", text)

    def test_structure_text_is_empty_when_the_service_is_down(self):
        self.write_endpoint("http://127.0.0.1:9")
        self.assertEqual(trailblazer.structure_text(self.repo, [], "head bob"), "")

    # --- a graph that cannot answer is not a graph that says "nothing" ---

    def test_where_and_uses_are_none_when_the_graph_fails(self):
        url, _server = self.serve(lambda op, body: (503, {"error": "operation failed"})
                                  if op in ("where", "uses") else watched_answer(op, body))
        self.write_endpoint(url)
        self.assertIsNone(trailblazer.where_lines(self.repo, "headBob"))
        self.assertIsNone(trailblazer.uses_edges(self.repo, 7))

    def test_where_and_uses_are_empty_lists_for_a_real_empty_answer(self):
        url, _server = self.serve(lambda op, body: (200, []) if op in ("where", "uses")
                                  else watched_answer(op, body))
        self.write_endpoint(url)
        self.assertEqual(trailblazer.where_lines(self.repo, "headBob"), [])
        self.assertEqual(trailblazer.uses_edges(self.repo, 7), [])

    def test_structure_text_says_unknown_when_the_graph_fails(self):
        url, _server = self.serve(lambda op, body: (503, {"error": "operation failed"})
                                  if op in ("where", "uses") else watched_answer(op, body))
        self.write_endpoint(url)
        hits = trailblazer.search_hits(self.repo, "head bob")
        text = trailblazer.structure_text(self.repo, hits, "where is `stepBob` used?")
        self.assertIn("users unknown: graph unavailable", text)
        self.assertNotIn("no other definition uses it", text)
        self.assertNotIn("used by: nothing", text)

    def test_structure_text_reports_nothing_only_when_it_knows(self):
        url, _server = self.serve(lambda op, body: (200, []) if op in ("where", "uses")
                                  else watched_answer(op, body))
        self.write_endpoint(url)
        hits = trailblazer.search_hits(self.repo, "head bob")
        text = trailblazer.structure_text(self.repo, hits, "where is `stepBob` used?")
        self.assertIn("no other definition uses it", text)
        self.assertNotIn("unavailable", text)

    def test_a_graph_that_goes_away_mid_run_is_not_an_empty_graph(self):
        url, server = self.serve()
        self.write_endpoint(url)
        self.assertIsNotNone(trailblazer.watched(self.repo))  # status succeeds
        server.shutdown()  # ... and then the service stops
        server.server_close()
        self.assertIsNone(trailblazer.where_lines(self.repo, "headBob"))
        self.assertIsNone(trailblazer.uses_edges(self.repo, 7))

    # --- ranger.index_candidates(): which source it takes ---

    def test_index_candidates_uses_trailblazer_and_never_syncs(self):
        url, _server = self.serve()
        self.write_endpoint(url)
        with mock.patch.object(index, "sync", exploding_sync):
            text, appendix = ranger.index_candidates(str(self.repo), "head bob", False)
        self.assertIn("assets/headBob.ts:10-24", text)
        self.assertIn("headBob", appendix)
        # The block AFTER the label: a graph note would be printed there, so
        # assert on the text that follows the range label, not the text before it.
        block = text.split("assets/headBob.ts:10-24", 1)[1].split("\n\n", 1)[0]
        for label in ("defines:", "imports:", "imported by:", "uses:"):
            self.assertNotIn(label, block)  # no graph of its own

    def test_index_candidates_falls_back_to_its_own_index(self):
        calls = []

        def fake_sync(work_dir, verbose=False):
            calls.append(work_dir)
            return "ranger_test"

        hits = [{"path": "assets/headBob.ts", "start": 10, "end": 24, "score": 0.7,
                 "text": "\n".join(self.lines[9:24])}]
        with mock.patch.object(index, "sync", fake_sync), \
             mock.patch.object(index, "search", lambda *a, **k: hits):
            text, appendix = ranger.index_candidates(str(self.repo), "head bob", False)
        self.assertEqual(calls, [str(self.repo)])
        self.assertIn("assets/headBob.ts:10-24", text)
        self.assertIsInstance(appendix, str)

    def test_index_candidates_survives_a_broken_index(self):
        with mock.patch.object(index, "sync", side_effect=RuntimeError("qdrant down")):
            self.assertEqual(ranger.index_candidates(str(self.repo), "head bob", False), ("", ""))

    def test_index_candidates_uses_its_own_index_when_trailblazer_has_no_hits(self):
        # Trailblazer watches the repo but has nothing to say: vectors still being
        # built, or its search failed. That is not a reason to hand the model nothing.
        def no_hits(op, body):
            if op == "search":
                return 200, []
            return watched_answer(op, body)

        url, _server = self.serve(no_hits)
        self.write_endpoint(url)
        calls = []
        hits = [{"path": "assets/stepBob.ts", "start": 5, "end": 30, "score": 0.4, "text": ""}]
        with mock.patch.object(index, "sync", lambda *a, **k: calls.append(a) or "coll"), \
             mock.patch.object(index, "search", lambda *a, **k: hits):
            text, appendix = ranger.index_candidates(str(self.repo), "head bob", False)
        self.assertEqual(len(calls), 1)
        self.assertIn("assets/stepBob.ts:5-30", text)

    def test_empty_trailblazer_hits_say_which_source_was_used(self):
        def no_hits(op, body):
            return (200, []) if op == "search" else watched_answer(op, body)

        url, _server = self.serve(no_hits)
        self.write_endpoint(url)
        with mock.patch.object(index, "sync", return_value="coll"), \
             mock.patch.object(index, "search", return_value=[]), \
             contextlib.redirect_stderr(io.StringIO()) as err:
            ranger.index_candidates(str(self.repo), "head bob", False)
        self.assertIn("trailblazer watches the repo but has no hits", err.getvalue())
        self.assertIn("using ranger's own index", err.getvalue())

    def test_no_notice_when_trailblazer_is_not_watched(self):
        self.write_endpoint("http://127.0.0.1:9")  # nothing is listening
        with mock.patch.object(index, "sync", return_value="coll"), \
             mock.patch.object(index, "search", return_value=[]), \
             contextlib.redirect_stderr(io.StringIO()) as err:
            ranger.index_candidates(str(self.repo), "head bob", False)
        self.assertEqual(err.getvalue(), "")

    def test_watched_is_asked_once_per_run(self):
        url, server = self.serve()
        self.write_endpoint(url)
        status = trailblazer.watched(str(self.repo))  # what main() does once
        ranger.index_candidates(str(self.repo), "head bob", False, status)
        self.assertEqual([op for op, _ in server.calls].count("status"), 1)
        server.calls.clear()
        ranger.graph_tool_functions(str(self.repo), status)
        self.assertEqual(server.calls, [])  # the answer it was handed is used as it is

    # --- the two graph tools, only where Trailblazer watches ---

    def test_graph_tools_exist_only_where_trailblazer_watches(self):
        self.assertEqual(ranger.graph_tool_functions(str(self.repo)), {})
        url, _server = self.serve()
        self.write_endpoint(url)
        functions = ranger.graph_tool_functions(str(self.repo))
        self.assertEqual(sorted(functions), ["Uses", "Where"])
        self.assertIn("assets/headBob.ts:10-24 headBob", functions["Where"](name="headBob"))
        self.assertIn("assets/stepBob.ts:42 stepBob (calls, uncertain)", functions["Uses"](name="headBob"))
        self.assertIn("not declared", functions["Where"](name="nothingHere"))
        self.assertIn("no definition uses", functions["Uses"](name="nothingHere"))

    def test_graph_tools_report_an_unavailable_graph_not_an_empty_one(self):
        # status answers, then the graph stops: a 503 must not become "not declared"
        # and "nothing uses this".
        url, _server = self.serve(lambda op, body: (503, {"error": "operation failed"})
                                  if op in ("where", "uses") else watched_answer(op, body))
        self.write_endpoint(url)
        functions = ranger.graph_tool_functions(str(self.repo))
        self.assertEqual(sorted(functions), ["Uses", "Where"])
        self.assertEqual(functions["Where"](name="headBob"), ranger.GRAPH_UNAVAILABLE)
        self.assertEqual(functions["Uses"](name="headBob"), ranger.GRAPH_UNAVAILABLE)
        self.assertNotIn("not declared", functions["Where"](name="headBob"))
        self.assertNotIn("no definition uses", functions["Uses"](name="headBob"))

    def test_graph_tools_say_unavailable_when_the_service_goes_away(self):
        url, server = self.serve()
        self.write_endpoint(url)
        functions = ranger.graph_tool_functions(str(self.repo))
        self.assertIn("assets/headBob.ts:10-24 headBob", functions["Where"](name="headBob"))
        server.shutdown()  # connection refused from here on
        server.server_close()
        self.assertEqual(functions["Where"](name="headBob"), ranger.GRAPH_UNAVAILABLE)
        self.assertEqual(functions["Uses"](name="headBob"), ranger.GRAPH_UNAVAILABLE)

    def test_graph_tools_report_an_empty_graph_as_empty(self):
        url, _server = self.serve(lambda op, body: (200, []) if op in ("where", "uses")
                                  else watched_answer(op, body))
        self.write_endpoint(url)
        functions = ranger.graph_tool_functions(str(self.repo))
        self.assertIn("not declared", functions["Where"](name="headBob"))
        self.assertIn("no definition uses", functions["Uses"](name="headBob"))

    def test_graph_tools_are_absent_from_the_prompt_without_trailblazer(self):
        self.assertNotIn("definition graph", ranger.build_system_prompt(str(self.repo), False))
        self.assertIn("definition graph", ranger.build_system_prompt(str(self.repo), True))

    # --- ensure_watched(): waking the service and asking for a watch ---

    def no_systemctl(self):
        """Fails the test if anything reaches the real systemctl."""
        return mock.patch.object(trailblazer.subprocess, "run",
                                 side_effect=AssertionError("systemctl must not be called"))

    def test_ensure_watched_asks_an_unwatched_repo_to_be_watched(self):
        asked = []

        def unwatched_until_asked(op, body):
            if op == "watch":
                asked.append(body["repo"])
                return 200, {"watched": True}
            if op == "status" and not asked:
                return 409, {"error": "not watched"}
            return watched_answer(op, body)

        url, server = self.serve(unwatched_until_asked)
        self.write_endpoint(url)
        with self.no_systemctl(), contextlib.redirect_stderr(io.StringIO()):
            status = trailblazer.ensure_watched(str(self.repo))
        self.assertEqual(status, STATUS_RESULT)
        self.assertEqual([op for op, _ in server.calls], ["status", "watch", "status"])
        self.assertEqual(asked, [str(self.repo)])

    def test_ensure_watched_needs_nothing_for_a_watched_repo(self):
        url, server = self.serve()
        self.write_endpoint(url)
        with self.no_systemctl():
            self.assertEqual(trailblazer.ensure_watched(str(self.repo)), STATUS_RESULT)
        self.assertEqual([op for op, _ in server.calls], ["status"])

    def test_ensure_watched_leaves_a_running_but_silent_service_alone(self):
        # Busy or wedged: a restart would throw away its indexing.
        self.write_endpoint("http://127.0.0.1:9")  # nothing is listening
        runs = []

        def systemctl(argv, **kw):
            runs.append(argv)
            return subprocess.CompletedProcess(argv, 0)  # is-active: yes

        with mock.patch.object(trailblazer.subprocess, "run", systemctl), \
             contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertIsNone(trailblazer.ensure_watched(str(self.repo)))
        self.assertEqual([argv[2] for argv in runs], ["is-active"])
        self.assertIn("leaving it alone", err.getvalue())

    def test_ensure_watched_starts_a_stopped_service_on_its_endpoint_event(self):
        # The service publishes endpoint.json by rename once its API listens; the
        # wait is on that event, so the status asked right after it is answered.
        # The waiter itself is faked: no real inotifywait, so the test passes
        # whatever the host's inotify instance limit allows.
        url, server = self.serve()
        runs = []

        def systemctl(argv, **kw):
            runs.append(argv)
            if argv[2] == "is-active":
                return subprocess.CompletedProcess(argv, 3)  # inactive
            temp = self.state / "endpoint.json.tmp"
            temp.write_text(json.dumps({"url": url, "token": "t0ken"}))
            temp.rename(self.state / "endpoint.json")
            return subprocess.CompletedProcess(argv, 0)

        waiter = FakeWaiter(["Setting up watches.\n", "Watches established.\n"], wait_code=0)
        with mock.patch.object(trailblazer.subprocess, "run", systemctl), \
             mock.patch.object(trailblazer.subprocess, "Popen", lambda *a, **k: waiter), \
             contextlib.redirect_stderr(io.StringIO()):
            status = trailblazer.ensure_watched(str(self.repo))
        self.assertEqual([argv[2] for argv in runs], ["is-active", "start"])
        self.assertEqual(status, STATUS_RESULT)
        self.assertEqual([op for op, _ in server.calls], ["status"])

    def test_a_start_that_publishes_nothing_fires_the_backstop(self):
        # The backstop is faked too: the waiter's wait() returns nonzero at once
        # instead of blocking for the real START_BACKSTOP_S.
        def systemctl(argv, **kw):
            return subprocess.CompletedProcess(argv, 3 if argv[2] == "is-active" else 0)

        waiter = FakeWaiter(["Setting up watches.\n", "Watches established.\n"], wait_code=1)
        with mock.patch.object(trailblazer.subprocess, "run", systemctl), \
             mock.patch.object(trailblazer.subprocess, "Popen", lambda *a, **k: waiter), \
             mock.patch.object(trailblazer, "START_BACKSTOP_S", 1), \
             contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertIsNone(trailblazer.ensure_watched(str(self.repo)))
        self.assertIn("anomaly", err.getvalue())

    def test_a_waiter_that_cannot_set_up_says_why_and_does_not_start(self):
        # inotifywait gives up when its instances are exhausted: its own stderr
        # must reach the operator, with the limit to raise, and the service
        # stays stopped — the operator decides, not the script.
        runs = []

        def systemctl(argv, **kw):
            runs.append(argv)
            return subprocess.CompletedProcess(argv, 3 if argv[2] == "is-active" else 0)

        waiter = FakeWaiter(["Setting up watches.\n",
                             "Couldn't initialize inotify: Too many open files\n"],
                            wait_code=1)
        with mock.patch.object(trailblazer.subprocess, "run", systemctl), \
             mock.patch.object(trailblazer.subprocess, "Popen", lambda *a, **k: waiter), \
             contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertFalse(trailblazer.start_service())
            self.assertIsNone(trailblazer.ensure_watched(str(self.repo)))
        message = err.getvalue()
        self.assertIn("Couldn't initialize inotify: Too many open files", message)
        self.assertIn("fs.inotify.max_user_instances", message)
        self.assertEqual([argv[2] for argv in runs], ["is-active", "is-active"])  # never started

    def test_a_search_slowed_by_indexing_still_answers(self):
        # Trailblazer's own embedding backlog delays a search past the plain
        # per-operation timeout; that must not read as "no hits".
        def slow_search(op, body):
            if op == "search":
                time.sleep(0.3)
            return watched_answer(op, body)

        url, _server = self.serve(slow_search)
        self.write_endpoint(url)
        with mock.patch.object(trailblazer.call, "__defaults__", (0.1,)):
            self.assertIsNone(trailblazer.call("search", {"repo": str(self.repo), "text": "x"}))
            hits = trailblazer.search_hits(str(self.repo), "head bob")
        self.assertIn("assets/headBob.ts", [h["path"] for h in hits])

    # --- mini_ranger.py's command line: retrieval only, no model ---

    def test_mini_ranger_prints_candidates_and_structure_without_a_model(self):
        url, _server = self.serve()
        self.write_endpoint(url)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.no_systemctl(), \
             mock.patch.object(index, "sync", exploding_sync), \
             mock.patch.object(ranger, "call_model", side_effect=AssertionError("no model")), \
             mock.patch.object(sys, "argv", ["mini_ranger.py", "-C", str(self.repo), "-q", "head bob"]):
            mini_ranger.main()
        self.assertIn("--- candidates, best match first ---", out.getvalue())
        self.assertIn("assets/headBob.ts:10-24", out.getvalue())
        self.assertIn("--- structure ---", out.getvalue())

    def test_mini_ranger_exits_1_when_no_index_source_can_answer(self):
        # No endpoint file, so Trailblazer cannot watch; ranger's own index
        # raises, so it cannot answer either: nothing could answer, and the
        # message must say that rather than claiming two false facts.
        err = io.StringIO()
        inactive = subprocess.CompletedProcess(["systemctl", "--user"], 3)
        with contextlib.redirect_stderr(err), \
             mock.patch.object(trailblazer.subprocess, "run", return_value=inactive), \
             mock.patch.object(trailblazer.subprocess, "Popen",
                               side_effect=OSError("inotifywait missing")), \
             mock.patch.object(index, "sync", side_effect=RuntimeError("qdrant down")), \
             mock.patch.object(sys, "argv", ["mini_ranger.py", "-C", str(self.repo), "-q", "head bob"]):
            with self.assertRaises(SystemExit) as caught:
                mini_ranger.main()
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("no index source could answer", err.getvalue())
        self.assertIn("Trailblazer", err.getvalue())
        self.assertIn("own index", err.getvalue())

    def test_mini_ranger_exits_2_when_the_index_answered_with_no_hits(self):
        # The index answered (sync succeeded, search returned nothing) and the
        # question names no code-like symbol, so there are no candidates: a
        # different situation from having no source at all, so a different
        # message and exit status.
        def no_hits(op, body):
            return (200, []) if op == "search" else watched_answer(op, body)

        url, _server = self.serve(no_hits)
        self.write_endpoint(url)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), \
             mock.patch.object(index, "sync", return_value="coll"), \
             mock.patch.object(index, "search", return_value=[]), \
             mock.patch.object(sys, "argv",
                               ["mini_ranger.py", "-C", str(self.repo), "-q", "how does this work"]):
            with self.assertRaises(SystemExit) as caught:
                mini_ranger.main()
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("the index answered but found no candidate locations", err.getvalue())

    def test_candidates_with_source_names_which_source_answered(self):
        url, _server = self.serve()
        self.write_endpoint(url)
        text, _appendix, source = mini_ranger.candidates_with_source(str(self.repo), "head bob", False)
        self.assertEqual(source, "trailblazer")
        self.assertIn("assets/headBob.ts:10-24", text)

        with mock.patch.object(index, "sync", return_value="coll"), \
             mock.patch.object(index, "search", return_value=[]):
            text, _appendix, source = mini_ranger.candidates_with_source(
                str(self.repo), "how does this work", False, watched=None)
        self.assertEqual(source, "index")
        self.assertEqual(text, "")

        with mock.patch.object(index, "sync", side_effect=RuntimeError("qdrant down")):
            _text, _appendix, source = mini_ranger.candidates_with_source(
                str(self.repo), "head bob", False, watched=None)
        self.assertIsNone(source)

    # --- index.py's command line ---

    def run_index_cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.object(sys, "argv", ["index.py", *argv]):
            index.main()
        return out.getvalue()

    def test_cli_query_uses_trailblazer_and_does_not_sync(self):
        url, _server = self.serve()
        self.write_endpoint(url)
        out = self.run_index_cli("-C", str(self.repo), "-q", "head bob")
        self.assertIn("0.830  assets/headBob.ts:10-24", out)
        self.assertIn("assets/stepBob.ts:5-30", out)
        self.assertIn("assets/headBob.ts:3-3", out)

    def test_cli_own_index_flag_syncs_and_searches(self):
        self.write_endpoint("http://127.0.0.1:9")  # Trailblazer is not there
        calls = []
        hits = [{"path": "assets/headBob.ts", "start": 10, "end": 24, "score": 0.5, "text": ""}]
        with mock.patch.object(index, "sync", lambda *a, **k: calls.append(a)), \
             mock.patch.object(index, "search", lambda *a, **k: hits):
            out = self.run_index_cli("-C", str(self.repo), "-q", "head bob")
        self.assertEqual(len(calls), 1)
        self.assertIn("0.500  assets/headBob.ts:10-24", out)

    def test_cli_query_syncs_when_trailblazer_has_no_hits(self):
        def no_hits(op, body):
            return (200, []) if op == "search" else watched_answer(op, body)

        url, _server = self.serve(no_hits)
        self.write_endpoint(url)
        calls = []
        hits = [{"path": "assets/stepBob.ts", "start": 5, "end": 30, "score": 0.4, "text": ""}]
        with mock.patch.object(index, "sync", lambda *a, **k: calls.append(a)), \
             mock.patch.object(index, "search", lambda *a, **k: hits):
            out = self.run_index_cli("-C", str(self.repo), "-q", "head bob")
        self.assertEqual(len(calls), 1)
        self.assertIn("0.400  assets/stepBob.ts:5-30", out)


if __name__ == "__main__":
    unittest.main()