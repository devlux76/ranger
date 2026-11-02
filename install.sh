#!/bin/sh
# Install ranger and scout for Claude and for the agents' skills directory.
#
# scout is the retrieval layer (scout.py with index.py and
# trailblazer.py, no model); ranger adds the model on top. Each install directory
# gets the code it runs and a SKILL.md rendered from its template, with
# @SKILL_DIR@ standing for the install path. Run it from the ranger repository;
# it never touches anything else.

set -eu

SRC=$(cd "$(dirname "$0")" && pwd)
RETRIEVAL="$SRC/index.py $SRC/trailblazer.py $SRC/scout.py"

for ROOT in "$HOME/.claude/skills" "$HOME/.agents/skills"; do
    DIR="$ROOT/ranger"
    mkdir -p "$DIR"
    # shellcheck disable=SC2086
    cp $RETRIEVAL "$SRC/ranger.py" "$DIR/"
    rm -f "$DIR/Modelfile" "$DIR/mini_ranger.py"  # scout was called mini-ranger until 2026-11
    sed "s|@SKILL_DIR@|$DIR|g" "$SRC/SKILL.md.in" > "$DIR/SKILL.md"
    echo "installed $DIR"

    DIR="$ROOT/scout"
    mkdir -p "$DIR"
    # shellcheck disable=SC2086
    cp $RETRIEVAL "$DIR/"
    rm -f "$DIR/Modelfile" "$DIR/ranger.py"  # the old 1.2B agent loop
    sed "s|@SKILL_DIR@|$DIR|g" "$SRC/scout.SKILL.md.in" > "$DIR/SKILL.md"
    echo "installed $DIR"
    # The old name; leaving it would list a stale second copy of the skill.
    if [ -d "$ROOT/mini-ranger" ]; then rm -rf "$ROOT/mini-ranger"; echo "removed $ROOT/mini-ranger"; fi
done
