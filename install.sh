#!/bin/sh
# Install ranger and mini-ranger for Claude and for the agents' skills directory.
#
# mini-ranger is the retrieval layer (mini_ranger.py with index.py and
# trailblazer.py, no model); ranger adds the model on top. Each install directory
# gets the code it runs and a SKILL.md rendered from its template, with
# @SKILL_DIR@ standing for the install path. Run it from the ranger repository;
# it never touches anything else.

set -eu

SRC=$(cd "$(dirname "$0")" && pwd)
RETRIEVAL="$SRC/index.py $SRC/trailblazer.py $SRC/mini_ranger.py"

for ROOT in "$HOME/.claude/skills" "$HOME/.agents/skills"; do
    DIR="$ROOT/ranger"
    mkdir -p "$DIR"
    # shellcheck disable=SC2086
    cp $RETRIEVAL "$SRC/ranger.py" "$DIR/"
    rm -f "$DIR/Modelfile"
    sed "s|@SKILL_DIR@|$DIR|g" "$SRC/SKILL.md.in" > "$DIR/SKILL.md"
    echo "installed $DIR"

    DIR="$ROOT/mini-ranger"
    mkdir -p "$DIR"
    # shellcheck disable=SC2086
    cp $RETRIEVAL "$DIR/"
    rm -f "$DIR/Modelfile" "$DIR/ranger.py"  # the old 1.2B agent loop
    sed "s|@SKILL_DIR@|$DIR|g" "$SRC/mini-ranger.SKILL.md.in" > "$DIR/SKILL.md"
    echo "installed $DIR"
done
