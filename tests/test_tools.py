#!/usr/bin/env python3
"""Tests for ranger's Grep, Glob and Read tools.

Run with:  python3 -m pytest tests -q

Every test builds a throwaway repository under tempfile and changes into an
unrelated directory, so nothing here depends on where pytest was started.
"""

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import ranger  # noqa: E402


class ToolTestCase(unittest.TestCase):
    """A temp repository under test, with the process cwd set elsewhere."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.work = self.root / "work"
        self.process_cwd = self.root / "elsewhere"
        self.work.mkdir()
        self.process_cwd.mkdir()
        self.previous_cwd = os.getcwd()
        os.chdir(self.process_cwd)
        self.addCleanup(os.chdir, self.previous_cwd)


class ToolPathTests(ToolTestCase):
    # bug:a9b76fa — omitted, empty and relative paths resolve under work_dir.

    def test_grep_without_a_path_searches_work_dir_not_the_process_cwd(self):
        (self.work / "target.txt").write_text("alpha sentinel omega\n")
        (self.process_cwd / "decoy.txt").write_text("alpha sentinel omega\n")
        out = ranger.run_tool("Grep", {"pattern": "sentinel"}, str(self.work))
        self.assertIn("target.txt", out)
        self.assertNotIn("decoy.txt", out)

    def test_glob_without_a_directory_lists_work_dir_not_the_process_cwd(self):
        (self.work / "inside.txt").write_text("")
        (self.process_cwd / "outside.txt").write_text("")
        out = ranger.run_tool("Glob", {"pattern": "*.txt"}, str(self.work))
        self.assertIn("inside.txt", out)
        self.assertNotIn("outside.txt", out)

    def test_read_with_a_relative_path_reads_under_work_dir(self):
        (self.work / "notes.md").write_text("first line\nsecond line\n")
        out = ranger.run_tool("Read", {"path": "notes.md"}, str(self.work))
        self.assertIn("1|first line", out)
        self.assertIn("2|second line", out)

    def test_grep_with_a_relative_path_is_scoped_to_work_dir(self):
        (self.work / "pkg").mkdir()
        (self.work / "pkg" / "rel.txt").write_text("relative needle\n")
        out = ranger.run_tool("Grep", {"pattern": "needle", "path": "pkg"}, str(self.work))
        self.assertIn("rel.txt", out)

    def test_read_with_an_absolute_path_is_unaffected(self):
        (self.work / "abs.txt").write_text("absolute content\n")
        out = ranger.run_tool("Read", {"path": str(self.work / "abs.txt")}, str(self.work))
        self.assertIn("1|absolute content", out)

    def test_a_tool_call_without_work_dir_uses_the_process_cwd(self):
        (self.process_cwd / "here.txt").write_text("cwd needle\n")
        self.assertIn("here.txt", ranger.tool_grep("needle"))
        self.assertIn("1|cwd needle", ranger.tool_read("here.txt"))

    def test_run_tool_passes_work_dir_to_tools_that_accept_it(self):
        seen = []

        def echo(name, **kw):
            seen.append(kw.get("work_dir"))
            return name

        out = ranger.run_tool("Echo", {"name": "hi"}, str(self.work), {"Echo": echo})
        self.assertEqual(out, "hi")
        self.assertEqual(seen, [str(self.work)])

    def test_run_tool_still_reports_an_unknown_tool(self):
        self.assertIn("unknown tool", ranger.run_tool("Nope", {}, str(self.work)))


if __name__ == "__main__":
    unittest.main()
