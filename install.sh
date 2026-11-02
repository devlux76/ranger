#!/bin/sh
# Install ranger for Claude and the agents; Scout has its own source repository.
#
# Retrieval helpers come from ~/projects/scout. Scout's installed skill paths
# link to that repository and are maintained independently of this installer.

set -eu

SRC=$(cd "$(dirname "$0")" && pwd)
SCOUT_SRC="$HOME/projects/scout"
RETRIEVAL="$SCOUT_SRC/index.py $SCOUT_SRC/trailblazer.py $SCOUT_SRC/scout.py"

for ROOT in "$HOME/.claude/skills" "$HOME/.agents/skills"; do
    DIR="$ROOT/ranger"
    mkdir -p "$DIR"
    # shellcheck disable=SC2086
    cp $RETRIEVAL "$SRC/ranger.py" "$DIR/"
    rm -f "$DIR/Modelfile" "$DIR/mini_ranger.py"  # scout was called mini-ranger until 2026-11
    sed "s|@SKILL_DIR@|$DIR|g" "$SRC/SKILL.md.in" > "$DIR/SKILL.md"
    echo "installed $DIR"

    # Keep removing the retired skill name during upgrades.
    if [ -d "$ROOT/mini-ranger" ]; then rm -rf "$ROOT/mini-ranger"; echo "removed $ROOT/mini-ranger"; fi
done
