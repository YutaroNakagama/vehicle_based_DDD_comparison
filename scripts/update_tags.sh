#!/bin/bash
# Update ctags for Python code in this project
# Usage: ./scripts/update_tags.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

ctags -R --languages=Python --python-kinds=-iv \
    -f "$PROJECT_ROOT/tags" \
    "$PROJECT_ROOT/src/" \
    "$PROJECT_ROOT/scripts/python/"

echo "Updated tags: $(wc -l < "$PROJECT_ROOT/tags") entries"
