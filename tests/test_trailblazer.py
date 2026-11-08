#!/usr/bin/env python3
"""Tests for ranger's graph tools and the scout CLI path that ranger shares.

Run with:  python3 -m unittest discover -s tests

Trailblazer is replaced by a fake HTTP service on a random loopback port, so
nothing here starts, stops or depends on the real one.
"""

import contextlib
import io
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO)]; sys.path.append(str(Path.home() / "projects" / "scout"))  # scout: canonical source

import scout     # noqa: E402
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

    def no_systemctl(self):
        """Fails the test if anything reaches the real systemctl."""
        return mock.patch.object(trailblazer.subprocess, "run",
                                 side_effect=AssertionError("systemctl must not be called"))

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

    # --- scout.py's command line: retrieval only, no model ---

    def test_scout_prints_candidates_and_structure_without_a_model(self):
        url, _server = self.serve()
        self.write_endpoint(url)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.no_systemctl(), \
             mock.patch.object(ranger, "call_model", side_effect=AssertionError("no model")), \
             mock.patch.object(sys, "argv", ["scout.py", "-C", str(self.repo), "-q", "head bob"]):
            scout.main()
        self.assertIn("--- candidates, best match first ---", out.getvalue())
        self.assertIn("assets/headBob.ts:10-24", out.getvalue())
        self.assertIn("--- structure ---", out.getvalue())


if __name__ == "__main__":
    unittest.main()
