#!/usr/bin/env python3
"""Tests for ranger's Grep, Glob and Read tools.

Run with:  python3 -m pytest tests -q

Every test builds a throwaway repository under tempfile and changes into an
unrelated directory, so nothing here depends on where pytest was started.
"""

import os
import subprocess
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


class GrepAndGlobOutputTests(ToolTestCase):
    # bug:761165d — a dash-led pattern is a pattern, and rg's exit 2 must not
    # throw away the matches it printed before reporting an unreadable file.

    def test_a_pattern_starting_with_a_dash_is_matched_literally(self):
        (self.work / "flags.txt").write_text("plain line\nopts -x enabled\n")
        out = ranger.tool_grep("-x", path=str(self.work), output_mode="content")
        self.assertIn("opts -x enabled", out)

    def test_an_unreadable_file_does_not_hide_the_matches(self):
        if os.geteuid() == 0:
            self.skipTest("root can read a mode-000 file")
        (self.work / "good.txt").write_text("needle in good\n")
        locked = self.work / "locked.txt"
        locked.write_text("no one reads me\n")
        locked.chmod(0o000)
        self.addCleanup(locked.chmod, 0o644)
        out = ranger.tool_grep("needle", path=str(self.work), output_mode="content")
        self.assertIn("needle in good", out)

    def test_exit_two_with_stdout_still_returns_the_matches(self):
        fake = subprocess.CompletedProcess(
            ["rg"], 2, stdout="hit.txt:1:match here\n",
            stderr="rg: bad.txt: Permission denied (os error 13)\n")
        with mock.patch.object(ranger.subprocess, "run", return_value=fake):
            out = ranger.tool_grep("match", path=str(self.work), output_mode="content")
        self.assertIn("hit.txt:1:match here", out)

    def test_exit_one_means_no_matches(self):
        fake = subprocess.CompletedProcess(["rg"], 1, stdout="", stderr="")
        with mock.patch.object(ranger.subprocess, "run", return_value=fake):
            out = ranger.tool_grep("nothing", path=str(self.work))
        self.assertEqual(out, "No matches found")

    def test_glob_keeps_its_files_when_rg_reports_errors(self):
        fake = subprocess.CompletedProcess(
            ["rg"], 2, stdout="a.txt\nb.txt\n",
            stderr="rg: locked.txt: Permission denied (os error 13)\n")
        with mock.patch.object(ranger.subprocess, "run", return_value=fake):
            out = ranger.tool_glob("*.txt", directory=str(self.work))
        self.assertIn("a.txt", out)
        self.assertIn("b.txt", out)

    def test_glob_exit_one_means_no_files(self):
        fake = subprocess.CompletedProcess(["rg"], 1, stdout="", stderr="")
        with mock.patch.object(ranger.subprocess, "run", return_value=fake):
            out = ranger.tool_glob("*.nope", directory=str(self.work))
        self.assertEqual(out, "No files found")


if __name__ == "__main__":
    unittest.main()
