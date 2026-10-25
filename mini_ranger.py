#!/usr/bin/env python3
"""mini-ranger — retrieval only: candidate locations and the code graph for a question.

No model runs here. Candidates come from Trailblazer's definition graph and
vectors when it watches the repository, otherwise from ranger's own semantic
index (index.py), plus exact-name matches for symbols in the question and the
graph around the best hits. ranger.py hands the same output to its model to
rerank and analyse.

Usage:
    python3 mini_ranger.py -C <repo> -q "<question>"
    python3 mini_ranger.py -C <repo> -q "<question>" --verbose
"""

import argparse
import sys
from pathlib import Path

UNSET = object()  # "the caller did not ask", so Trailblazer's status is asked once


def skill_modules():
    """index.py and trailblazer.py, which live beside this file."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import index
    import trailblazer
    return index, trailblazer


def trailblazer_status(work_dir, watched=UNSET, tb=None):
    """Trailblazer's status for this repository: asked once per run by main() and
    handed to every caller that needs it."""
    if watched is not UNSET:
        return watched
    return (tb or skill_modules()[1]).watched(work_dir)


def trailblazer_ready(work_dir):
    """Trailblazer's status for this repository, starting the service and asking it
    to watch the repository first when needed; None when it cannot answer, and
    ranger's own index is used instead. Asked once per run."""
    try:
        _index, trailblazer = skill_modules()
        return trailblazer.ensure_watched(work_dir)
    except Exception as e:
        print(f"ranger: Trailblazer unavailable ({e})", file=sys.stderr)
        return None


def index_candidates(work_dir, query, verbose, watched=UNSET):
    """Candidate locations for the question as (prompt text, structure appendix),
    or ("", "") if no index can answer.

    When Trailblazer watches the repository its definition graph and vectors are
    used as they are: ranger embeds nothing and syncs nothing, so the candidates
    are ready in seconds. When it watches the repository but has no hits to give —
    its vectors are still being built, or the call failed — ranger falls back to
    its own index rather than handing the model nothing. A failure of either
    source must not stop the search loop, which still works without candidates,
    only slower. `watched` is Trailblazer's status, when the caller has it.
    """
    try:
        index, trailblazer = skill_modules()
    except Exception as e:
        print(f"ranger index unavailable ({e}); searching without it", file=sys.stderr)
        return "", ""
    watched = trailblazer_status(work_dir, watched, trailblazer)

    # Trailblazer's hits are the candidates when it has any; an empty answer from
    # a service that is up is a reason to use the index below, not to give up.
    hits = trailblazer.search_hits(work_dir, query) if watched is not None else []
    graph = None
    if hits:
        source = "trailblazer"
        text = index.candidates_text(work_dir, hits)
        appendix = trailblazer.structure_text(work_dir, hits, query)
    else:
        if watched is not None:
            print("ranger: trailblazer watches the repo but has no hits for this question "
                  "(vectors still building, or its search failed); using ranger's own index",
                  file=sys.stderr)
        try:
            index.sync(work_dir, verbose=verbose)
            hits = index.search(work_dir, query)
        except Exception as e:
            print(f"ranger index unavailable ({e}); searching without it", file=sys.stderr)
            return "", ""
        source = "index"
        # The dependency map is rebuilt from the files on every query, so it is never stale.
        graph = index.build_graph(work_dir, index.list_files(work_dir))
        text = index.candidates_text(work_dir, hits, graph=graph)
        appendix = index.structure_text(work_dir, hits, query, graph)

    if verbose:
        print(f"  candidates from {source}", file=sys.stderr)
        for h in hits:
            print(f"  {source}: {h['score']:.3f} {h['path']}:{h['start']}-{h['end']}", file=sys.stderr)
    names = index.symbol_matches(work_dir, query, graph=graph)
    if names:
        text += "\n\nExact-name matches for symbols in the question (ripgrep, path:line:text):\n\n" + names
    return text, appendix


def main():
    parser = argparse.ArgumentParser(description="Candidate locations and the code graph for a question, without a model")
    parser.add_argument("-C", "--directory", default=".", help="Repository directory")
    parser.add_argument("-q", "--query", required=True, help="Question about the codebase")
    parser.add_argument("--verbose", action="store_true", help="Print the source and hits to stderr")
    args = parser.parse_args()

    work_dir = str(Path(args.directory).resolve())
    if not Path(work_dir).is_dir():
        print(f"Error: {work_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    candidates, structure = index_candidates(work_dir, args.query, args.verbose,
                                             trailblazer_ready(work_dir))
    if not candidates:
        print("no index could answer (Trailblazer not watching and ranger's own index unavailable)", file=sys.stderr)
        sys.exit(1)
    print("--- candidates, best match first ---\n" + candidates)
    if structure:
        print("\n--- structure ---\n" + structure)


if __name__ == "__main__":
    main()
