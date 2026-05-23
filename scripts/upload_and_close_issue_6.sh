#!/bin/bash
# Upload results/benchmark_dflash_concurrency.md to GitHub Issue 6 and close it.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"

REPORT_FILE="results/benchmark_dflash_concurrency.md"

if [ ! -f "$REPORT_FILE" ]; then
    echo "Error: $REPORT_FILE not found."
    exit 1
fi

echo "Uploading report to GitHub Issue 6..."
gh issue comment 6 --body-file "$REPORT_FILE"

echo "Closing GitHub Issue 6..."
gh issue close 6

echo "Done! Issue 6 has been successfully updated and closed."
