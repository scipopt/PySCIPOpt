#!/bin/bash -e

# Update baseline test files

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

for test_file in "$REPO_ROOT"/tests/@types/*.py; do
    echo "Updating mypy baseline for $test_file"
    output_file="${test_file%.*}.mypy.out"
    python -m mypy "$test_file" --warn-unused-ignores | grep "error:" > "$output_file"
done
