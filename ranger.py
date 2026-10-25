#!/usr/bin/env python3
"""ranger — codebase exploration via LFM2.5-2.6B on a local llama-server.

Delegates code search to a local LFM2.5 model. Asks for candidate locations
first — from Trailblazer's definition graph and vectors when it watches the
repository, otherwise from ranger's own semantic index (index.py) — then lets
the model verify them with Grep/Glob/Read through the OpenAI-compatible endpoint
and its native tool calling. Returns whatever the model says when it stops
calling tools.

Usage:
    python3 ranger.py -C <repo> -q "<question>"
    python3 ranger.py -C <repo> -q "<question>" --verbose
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

# The retrieval layer: candidate locations and the graph, no model.
from mini_ranger import UNSET, index_candidates, skill_modules, trailblazer_ready, trailblazer_status  # noqa: F401

# Mirai's llama.cpp router (user service mirai-llama-router): one copy of each
# model in memory, shared with her agent.
SERVER_URL = "http://127.0.0.1:8094"
MODEL = "lfm2.5-2.6b:q4"
MAX_TURNS = 100
TOOL_TIMEOUT = 10
MAX_TOOL_OUTPUT = 8000
# What a tool says when the graph cannot answer, as opposed to answering nothing:
# "not looked up" must never read as "not there".
GRAPH_UNAVAILABLE = "the definition graph is unavailable right now; use Grep instead"

# --- Tools ---

def tool_grep(pattern, path=".", output_mode="files_with_matches", context=0, head_limit=0, **kw):
    cmd = ["rg", pattern, "--heading", "--color", "never"]
    if output_mode == "files_with_matches":
        cmd.append("--files-with-matches")
    elif output_mode == "content":
        if context:
            cmd.extend(["-C", str(context)])
        cmd.append("-n")
    cmd.extend(["--glob", "!**/node_modules/**", "--glob", "!**/.git/**",
               "--glob", "!**/build/**", "--glob", "!**/dist/**"])
    if path:
        cmd.append(path)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=TOOL_TIMEOUT)
        result = out.stdout if out.returncode == 0 else out.stderr
    except subprocess.TimeoutExpired:
        return "Grep timed out"
    if head_limit:
        result = "\n".join(result.splitlines()[:head_limit])
    return result[:MAX_TOOL_OUTPUT] or "No matches found"


def tool_glob(pattern, directory=".", **kw):
    cmd = ["rg", "--files", "--glob", pattern,
           "--glob", "!**/node_modules/**", "--glob", "!**/.git/**",
           "--glob", "!**/build/**", "--glob", "!**/dist/**"]
    if directory:
        cmd.append(directory)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=TOOL_TIMEOUT)
        result = out.stdout if out.returncode == 0 else out.stderr
    except subprocess.TimeoutExpired:
        return "Glob timed out"
    lines = result.splitlines()
    if len(lines) > 100:
        lines = lines[:100]
        lines.append(f"... ({len(lines)} shown, more truncated)")
    return "\n".join(lines) if lines else "No files found"


def tool_read(path, offset=1, limit=150, **kw):
    p = Path(path)
    if not p.exists():
        return f"Error: {path} does not exist"
    if not p.is_file():
        return f"Error: {path} is not a file"
    try:
        raw = p.read_text(errors="replace").splitlines()
    except Exception as e:
        return f"Error reading file: {e}"
    if not raw:
        return "File is empty"
    limit = min(limit, 200)
    end = min(offset + limit - 1, len(raw))
    lines = []
    for i in range(offset - 1, end):
        line = raw[i]
        if len(line) > 2000:
            line = line[:2000] + "..."
        lines.append(f"{i+1}|{line}")
    if end < len(raw):
        lines.append(f"... ({len(raw) - end} more lines)")
    return "\n".join(lines)


TOOL_FUNCTIONS = {"Grep": tool_grep, "Glob": tool_glob, "Read": tool_read}

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "Grep",
            "description": "Search file contents using a regex pattern. Use 'files_with_matches' to find which files contain a pattern, or 'content' to see matching lines with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory to search in"},
                    "output_mode": {"type": "string", "enum": ["files_with_matches", "content"], "description": "files_with_matches or content"},
                    "context": {"type": "integer", "description": "Lines of context (content mode)"},
                    "head_limit": {"type": "integer", "description": "Limit output to first N lines"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Glob",
            "description": "List files matching a glob pattern. Example: '*.py', '**/*.ts', 'package.json'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern"},
                    "directory": {"type": "string", "description": "Directory to search in"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read file contents with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to file"},
                    "offset": {"type": "integer", "description": "Starting line (1-indexed)"},
                    "limit": {"type": "integer", "description": "Number of lines"},
                },
                "required": ["path"],
            },
        },
    },
]

# Two more tools, only when Trailblazer watches the repository: its graph already
# knows where a name is declared and who uses it, so those questions cost no
# search at all.

GRAPH_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "Where",
            "description": "Ask the repository's definition graph where a name is declared, as path:start-end qualified. The name may be bare (saveHikes) or qualified (Mini.Store.put/2). More exact than Grep when the same name is declared in several files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Symbol name, optionally qualified"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Uses",
            "description": "Ask the repository's definition graph every definition that uses a name, as path:line qualified (kind), where kind is calls, references, imports, tests or aliases, marked 'uncertain' when the graph could not settle which definition the reference means. Use it before calling two similarly named symbols related or independent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Symbol name, optionally qualified"},
                },
                "required": ["name"],
            },
        },
    },
]


def graph_tool_functions(work_dir, watched=UNSET):
    """The Trailblazer-backed tools, or {} when it does not watch the repository.
    Then the model is left with exactly the tools it has always had. `watched` is
    Trailblazer's status, when the caller has already asked for it."""
    try:
        _index, trailblazer = skill_modules()
        if trailblazer_status(work_dir, watched, trailblazer) is None:
            return {}

        def where(name, **kw):
            nodes = trailblazer.where_lines(work_dir, name)
            if nodes is None:
                return GRAPH_UNAVAILABLE
            lines = [trailblazer.node_line(n) for n in nodes]
            return "\n".join(lines)[:MAX_TOOL_OUTPUT] or f"{name} is not declared in this repository"

        def uses(name, **kw):
            edges = trailblazer.uses_edges(work_dir, name)
            if edges is None:
                return GRAPH_UNAVAILABLE
            lines = [trailblazer.edge_line(e) for e in edges]
            return "\n".join(lines)[:MAX_TOOL_OUTPUT] or f"no definition uses {name}"
    except Exception as e:
        print(f"trailblazer tools unavailable ({e})", file=sys.stderr)
        return {}
    return {"Where": where, "Uses": uses}


def build_system_prompt(work_dir, graph_tools=False):
    # The model is a filter and reranker over the semantic index, not an explorer:
    # it should answer from the candidates when they suffice and search only when
    # they are thin. Short answers matter because generation is the slow part.
    ls_output = "\n".join(sorted(os.listdir(work_dir))[:50])
    graph_rule = (
        "- The repository has a definition graph: Where NAME says where a name is declared "
        "(path:start-end) and Uses NAME lists every definition that uses it with its line and "
        "the kind of use. Prefer those over Grep for 'where is X defined' and 'what uses X'; "
        "check an edge marked uncertain before relying on it.\n"
        if graph_tools else ""
    )
    return (
        "You are a code search reranker. You get a question and candidate code locations "
        "from a semantic index, shown with numbered source lines, a similarity score and each "
        "file's place in the import graph (what it defines, imports and is imported by), plus "
        "exact-name matches for any symbols the question names, each with its edges (defined in, "
        "used by, and files where the same name appears with no import link). Decide which locations really "
        "answer the question and report them. Low similarity (under about 0.65) means the index "
        "found little: search instead of trusting it.\n\n"
        "Rules:\n"
        "- If the candidates already answer the question, answer at once without calling tools.\n"
        "- If they are thin, off-target or incomplete, search: Grep for a symbol or phrase "
        "(output_mode 'content' shows matching lines with line numbers), Read for a specific "
        "line range, Glob to find files by name.\n"
        "- A shared or similar name does not prove two symbols are related. Before calling them "
        "related or independent, Grep each one and check how it is defined, assigned and used. "
        "Quote the line that connects them, or state that no line does.\n"
        + graph_rule +
        "- Do not Read lines you have already been shown.\n"
        "- Always use absolute paths when calling tools.\n\n"
        "Answer format, at most 150 words: one line per finding, written as path:line followed "
        "by a short reason, best match first. No introduction, no summary, no code blocks.\n\n"
        f"Workspace: {work_dir}\n"
        f"Top-level contents:\n{ls_output}\n"
    )


def call_model(messages, tools=None):
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": tools or TOOL_DEFS,
        "temperature": 0.2,
        "frequency_penalty": 0.05,
    }
    with httpx.Client(timeout=3600.0) as client:
        resp = client.post(f"{SERVER_URL}/v1/chat/completions", json=payload)
        if resp.status_code != 200:
            print(f"ERROR {resp.status_code}: {resp.text[:500]}", file=sys.stderr)
        resp.raise_for_status()
        return resp.json()


def parse_tool_calls_from_content(content):
    """Fallback: parse <|tool_call_start|>[FuncName(arg='val')]<|tool_call_end|>"""
    pattern = r"<\|tool_call_start\|>\[(.*?)\]<\|tool_call_end\|>"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return []
    raw = match.group(1)
    calls = []
    for part in re.split(r"\)\s*,\s*(?=[A-Z])", raw):
        part = part.strip().rstrip(")")
        if not part:
            continue
        m = re.match(r"(\w+)\((.*)\)", part, re.DOTALL)
        if not m:
            continue
        func_name = m.group(1)
        args_str = m.group(2).strip()
        args = {}
        if args_str:
            for arg in re.split(r",\s*(?=\w+=)", args_str):
                arg = arg.strip()
                if "=" not in arg:
                    continue
                k, v = arg.split("=", 1)
                k, v = k.strip(), v.strip()
                if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                    v = v[1:-1]
                else:
                    try:
                        v = int(v)
                    except ValueError:
                        try:
                            v = float(v)
                        except ValueError:
                            v = {"True": True, "False": False, "None": None}.get(v, v)
                args[k] = v
        calls.append({"id": f"call_{func_name}_{len(calls)}", "function": {"name": func_name, "arguments": json.dumps(args)}})
    return calls


def run_tool(name, args, work_dir, functions=None):
    known = functions or TOOL_FUNCTIONS
    fn = known.get(name)
    if not fn:
        for k, v in known.items():
            if k.lower() == name.lower():
                fn = v
                break
    if not fn:
        return f"Error: unknown tool '{name}'"
    if "path" in args and not args["path"]:
        args["path"] = work_dir
    if "directory" in args and not args["directory"]:
        args["directory"] = work_dir
    try:
        return fn(**args)
    except Exception as e:
        return f"Error executing {name}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Explore a codebase using local LFM2.5")
    parser.add_argument("-C", "--directory", default=".", help="Repository directory")
    parser.add_argument("-q", "--query", required=True, help="Question about the codebase")
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS, help="Max exploration turns")
    parser.add_argument("--verbose", action="store_true", help="Print each turn to stderr")
    args = parser.parse_args()

    work_dir = str(Path(args.directory).resolve())
    if not Path(work_dir).is_dir():
        print(f"Error: {work_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Trailblazer's status is asked once here and handed to both callers: the
    # candidate source and the extra tools agree by construction.
    watched = trailblazer_ready(work_dir)

    question = args.query
    candidates, structure = index_candidates(work_dir, args.query, args.verbose, watched)
    # Appended to whatever the model answers: the edges as computed, not as the model read them.
    appendix = f"\n\n--- structure (from the index, not the model) ---\n{structure}" if structure else ""
    if candidates:
        question += (
            "\n\nCandidate locations from the semantic index, best match first. "
            "They may be wrong or incomplete.\n\n"
            + candidates
        )

    # Where and Uses are offered only when Trailblazer watches the repository;
    # without it the model has exactly the tools it has always had.
    extra = graph_tool_functions(work_dir, watched)
    functions = {**TOOL_FUNCTIONS, **extra}
    tools = TOOL_DEFS + GRAPH_TOOL_DEFS if extra else TOOL_DEFS
    if args.verbose and extra:
        print(f"  tools: {', '.join(functions)}", file=sys.stderr)

    messages = [
        {"role": "system", "content": build_system_prompt(work_dir, bool(extra))},
        {"role": "user", "content": question},
    ]

    for turn in range(1, args.max_turns + 1):
        if args.verbose:
            print(f"--- Turn {turn} ---", file=sys.stderr)

        t0 = time.perf_counter()
        response = call_model(messages, tools)
        elapsed = time.perf_counter() - t0

        choice = response.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []

        # Fallback: parse native format from content
        if not tool_calls and content:
            tool_calls = parse_tool_calls_from_content(content)

        if args.verbose:
            for tc in tool_calls:
                f = tc.get("function", {})
                print(f"  -> {f.get('name', '?')}({f.get('arguments', '')[:80]})", file=sys.stderr)
            if content and not tool_calls:
                print(f"  content: {content[:200]}", file=sys.stderr)
            print(f"  ({elapsed:.1f}s)", file=sys.stderr)

        # Add assistant message to history (with tool_calls if present)
        messages.append(msg)

        if not tool_calls:
            print(content + appendix)
            return

        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            tc_args = func.get("arguments", {})
            if isinstance(tc_args, str):
                try:
                    tc_args = json.loads(tc_args)
                except json.JSONDecodeError:
                    tc_args = {}

            result = run_tool(name, tc_args, work_dir, functions)
            if args.verbose:
                print(f"  <- {result[:120].replace(chr(10), ' ')}", file=sys.stderr)
            messages.append({"role": "tool", "content": result, "tool_call_id": tc.get("id", "")})

    messages.append({"role": "user", "content": "Provide your findings based on what you've searched so far."})
    response = call_model(messages, tools)
    choice = response.get("choices", [{}])[0]
    print((choice.get("message", {}).get("content") or "No response after max turns.") + appendix)


if __name__ == "__main__":
    main()