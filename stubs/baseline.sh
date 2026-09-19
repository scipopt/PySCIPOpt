#!/bin/bash -e

# Update baseline test files

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python "$REPO_ROOT"/scripts/generate_expr_type_tests.py

for test_file in "$REPO_ROOT"/tests/@types/*.py; do
    echo "Updating mypy baseline for $test_file"
    output_file="${test_file%.*}.mypy.out"
    # write to a temp file first to avoid truncating the output file if mypy fails
    tmp=$(mktemp)
    python -m mypy "$test_file" --warn-unused-ignores | grep "error:" > "$tmp"
    mv "$tmp" "$output_file"
done
