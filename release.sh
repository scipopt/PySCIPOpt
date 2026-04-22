#!/usr/bin/env bash
set -euo pipefail

VERSION_FILE="src/pyscipopt/_version.py"
SETUP_FILE="setup.py"
CHANGELOG="CHANGELOG.md"
REPO="scipopt/PySCIPOpt"

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        -h|--help) echo "Usage: $0 [--dry-run]"; exit 0 ;;
        *) echo "Error: unknown argument '$arg' (use --dry-run or --help)"; exit 1 ;;
    esac
done

# --- Pre-flight checks ---

if ! command -v gh &>/dev/null; then
    echo "Error: gh CLI is not installed. Install it from https://cli.github.com"
    exit 1
fi

if ! gh auth status &>/dev/null; then
    echo "Error: gh CLI is not authenticated. Run 'gh auth login' first."
    exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: working directory is not clean. Commit, stash, or remove changes first."
    exit 1
fi

CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" != "master" ]]; then
    if [[ "$DRY_RUN" == true ]]; then
        echo "DRY RUN: on '${CURRENT_BRANCH}' — a real run would require master."
    else
        echo "Error: must be on 'master' branch (currently on '${CURRENT_BRANCH}')."
        exit 1
    fi
fi

PUSH_REMOTE=$(git config --get "branch.${CURRENT_BRANCH}.remote" 2>/dev/null || echo origin)

git pull --ff-only

# --- Helper functions ---

validate_version() {
    if [[ ! "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Error: '$1' is not a valid version (expected X.Y.Z)"
        exit 1
    fi
}

# Promote a successful release candidate: tag, push master, clean up RC branch.
finalize_release() {
    local version="$1"
    local rc_branch="release-candidate-v${version}"

    echo "Pending release candidate: v${version}"
    echo "Checking workflow status on ${rc_branch}..."

    local run_data run_count status conclusion url run_id
    run_data=$(gh run list --workflow=build_wheels.yml --repo "$REPO" \
        --branch "$rc_branch" --limit 1 \
        --json databaseId,status,conclusion,url 2>/dev/null || echo "[]")
    run_count=$(echo "$run_data" | jq 'length')

    if [[ "$run_count" == "0" ]]; then
        echo "Error: no workflow run found for branch '${rc_branch}'."
        echo "Either wait a moment and try again, or clean up manually:"
        echo "  git push ${PUSH_REMOTE} --delete ${rc_branch}"
        echo "  git reset --hard HEAD~1"
        exit 1
    fi

    status=$(echo "$run_data" | jq -r '.[0].status')
    conclusion=$(echo "$run_data" | jq -r '.[0].conclusion')
    url=$(echo "$run_data" | jq -r '.[0].url')
    run_id=$(echo "$run_data" | jq -r '.[0].databaseId')

    if [[ "$status" != "completed" ]]; then
        echo "Workflow still running (status: ${status})."
        echo "URL: ${url}"
        echo "Re-run this script once it finishes."
        exit 0
    fi

    if [[ "$conclusion" != "success" ]]; then
        echo "Workflow failed (conclusion: ${conclusion})."
        echo "URL: ${url}"
        echo ""
        if [[ "$DRY_RUN" == true ]]; then
            echo "DRY RUN: would prompt to roll back (delete ${rc_branch} + reset local commit)."
            exit 0
        fi
        read -rp "Roll back (delete ${rc_branch} and reset local release commit)? [Y/n] " confirm
        if [[ ! "${confirm:-Y}" =~ ^[Nn] ]]; then
            git push "$PUSH_REMOTE" --delete "$rc_branch" || echo "(note: remote RC delete failed; clean up manually)"
            git reset --hard HEAD~1
            echo "Rolled back. Repo is at pre-release state."
        fi
        exit 1
    fi

    # Success: promote.
    if [[ "$DRY_RUN" == true ]]; then
        echo "DRY RUN: workflow succeeded (run ${run_id}). Would have run:"
        echo "  git tag v${version}"
        echo "  git push ${PUSH_REMOTE} ${CURRENT_BRANCH}"
        echo "  git push ${PUSH_REMOTE} v${version}"
        echo "  git push ${PUSH_REMOTE} --delete ${rc_branch}"
        exit 0
    fi

    echo "Workflow succeeded. Promoting release v${version}..."
    git tag "v${version}"
    git push "$PUSH_REMOTE" "$CURRENT_BRANCH"
    git push "$PUSH_REMOTE" "v${version}"
    git push "$PUSH_REMOTE" --delete "$rc_branch"

    echo ""
    echo "Done! v${version} is now tagged on ${PUSH_REMOTE}."
    echo ""
    echo "Remaining manual steps:"
    echo "  1. Verify test-pypi:"
    echo "       pip install -i https://test.pypi.org/simple/ PySCIPOpt==${version}"
    echo "  2. Release to production pypi:"
    echo "       gh workflow run build_wheels.yml --repo ${REPO} --ref v${version} -f upload_to_pypi=true -f test_pypi=false"
    echo "  3. Create a GitHub release from tag v${version}:"
    echo "       gh release create v${version} --repo ${REPO} --title v${version} --generate-notes"
    echo "  4. Update readthedocs: Builds -> Build version (latest and stable)"
}

# --- Detect pending release candidate ---
# If HEAD is a local release commit and its RC branch exists on origin, we are in
# the second phase (finalize). Otherwise we are starting a new release.

HEAD_MSG=$(git log -1 --format=%s)
if [[ "$HEAD_MSG" =~ ^release\ v([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
    PENDING_VERSION="${BASH_REMATCH[1]}"
    RC_BRANCH_CHECK="release-candidate-v${PENDING_VERSION}"
    if git ls-remote --heads --exit-code "$PUSH_REMOTE" "$RC_BRANCH_CHECK" &>/dev/null; then
        finalize_release "$PENDING_VERSION"
        exit 0
    else
        echo "Error: HEAD is 'release v${PENDING_VERSION}' but no RC branch '${RC_BRANCH_CHECK}' on ${PUSH_REMOTE}."
        echo "A previous run may have been interrupted. Recover manually:"
        echo "  a) Push the RC branch and re-run:"
        echo "       git push ${PUSH_REMOTE} HEAD:refs/heads/${RC_BRANCH_CHECK}"
        echo "  b) Or undo the local commit to start over:"
        echo "       git reset --hard HEAD~1"
        exit 1
    fi
fi

# --- Read current version ---

CURRENT_VERSION=$(sed -n "s/^__version__.*'\(.*\)'/\1/p" "$VERSION_FILE")
validate_version "$CURRENT_VERSION"
MAJOR=$(echo "$CURRENT_VERSION" | cut -d. -f1)
MINOR=$(echo "$CURRENT_VERSION" | cut -d. -f2)
PATCH=$(echo "$CURRENT_VERSION" | cut -d. -f3)

echo "Current version: ${CURRENT_VERSION}"

# --- Prompt for bump type ---

echo ""
echo "Release type:"
echo "  1) patch  -> $((MAJOR)).$((MINOR)).$((PATCH + 1))"
echo "  2) minor  -> $((MAJOR)).$((MINOR + 1)).0"
echo "  3) major  -> $((MAJOR + 1)).0.0"
echo ""
read -rp "Select [1/2/3]: " bump_type

case "$bump_type" in
    1|patch) NEW_VERSION="$((MAJOR)).$((MINOR)).$((PATCH + 1))" ;;
    2|minor) NEW_VERSION="$((MAJOR)).$((MINOR + 1)).0" ;;
    3|major) NEW_VERSION="$((MAJOR + 1)).0.0" ;;
    *) echo "Error: invalid selection '${bump_type}'"; exit 1 ;;
esac

# --- Check tag doesn't already exist ---

if git rev-parse "v${NEW_VERSION}" &>/dev/null; then
    echo "Error: tag 'v${NEW_VERSION}' already exists locally."
    exit 1
fi

if git ls-remote --tags --exit-code "$PUSH_REMOTE" "refs/tags/v${NEW_VERSION}" &>/dev/null; then
    echo "Error: tag 'v${NEW_VERSION}' already exists on ${PUSH_REMOTE}."
    exit 1
fi

# --- Show summary and confirm ---

echo ""
echo "Unreleased changelog entries:"
echo "-----------------------------"
sed -n '/^## Unreleased$/,/^## [0-9]/{/^## [0-9]/!p;}' "$CHANGELOG" | head -30
echo "-----------------------------"

UNRELEASED_BULLETS=$(sed -n '/^## Unreleased$/,/^## [0-9]/{/^## [0-9]/!p;}' "$CHANGELOG" | grep -c '^- ' || true)
if [[ "$UNRELEASED_BULLETS" == "0" ]]; then
    echo ""
    echo "Warning: the Unreleased section has no bullet entries."
    read -rp "Release anyway? [y/N] " empty_confirm
    [[ "${empty_confirm:-N}" =~ ^[Yy] ]] || exit 0
fi

TODAY=$(date +%Y.%m.%d)
RC_BRANCH="release-candidate-v${NEW_VERSION}"
echo ""
if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN: This script would:"
else
    echo "This script will (phase 1 of 2):"
fi
echo "  1. Update version ${CURRENT_VERSION} -> ${NEW_VERSION} in _version.py and setup.py"
echo "  2. Update CHANGELOG.md (${NEW_VERSION} - ${TODAY})"
echo "  3. Commit locally, push commit to branch '${RC_BRANCH}' on ${PUSH_REMOTE}"
echo "     (master is NOT pushed; no tag is created yet)"
echo "  4. Trigger the build wheels workflow on that branch (test-pypi)"
echo ""
echo "Once the workflow finishes, re-run this script:"
echo "  - success -> tag v${NEW_VERSION}, push master, delete RC branch"
echo "  - failure -> prompt to roll back (delete RC branch + reset local commit)"
echo ""
if [[ "$DRY_RUN" == false ]]; then
    read -rp "Proceed? [Y/n] " confirm
    [[ "${confirm:-Y}" =~ ^[Nn] ]] && exit 0
fi

# ============================================================
# From here on, everything runs without further prompts.
# If a step fails after commit, re-run this script: it detects the
# partial release commit at HEAD and prints recovery instructions.
# ============================================================

# --- Update version files ---

sed -i.bak "s/__version__.*=.*'.*'/__version__: str = '${NEW_VERSION}'/" "$VERSION_FILE"
if cmp -s "$VERSION_FILE" "${VERSION_FILE}.bak"; then
    echo "Error: failed to update version in $VERSION_FILE (pattern not found)"
    mv "${VERSION_FILE}.bak" "$VERSION_FILE"
    exit 1
fi
rm -f "${VERSION_FILE}.bak"

sed -i.bak "s/version=\"${CURRENT_VERSION}\"/version=\"${NEW_VERSION}\"/" "$SETUP_FILE"
if cmp -s "$SETUP_FILE" "${SETUP_FILE}.bak"; then
    echo "Error: failed to update version in $SETUP_FILE (pattern not found)"
    mv "${SETUP_FILE}.bak" "$SETUP_FILE"
    exit 1
fi
rm -f "${SETUP_FILE}.bak"

echo "Updated version: ${CURRENT_VERSION} -> ${NEW_VERSION}"

# --- Update changelog ---

sed -i.bak "s/^## Unreleased$/## ${NEW_VERSION} - ${TODAY}/" "$CHANGELOG"
if cmp -s "$CHANGELOG" "${CHANGELOG}.bak"; then
    echo "Error: failed to update changelog ('## Unreleased' heading not found)"
    mv "${CHANGELOG}.bak" "$CHANGELOG"
    exit 1
fi
rm -f "${CHANGELOG}.bak"

sed -i.bak "/^# CHANGELOG$/a\\
\\
## Unreleased\\
### Added\\
### Fixed\\
### Changed\\
### Removed\\
" "$CHANGELOG"
if cmp -s "$CHANGELOG" "${CHANGELOG}.bak"; then
    echo "Error: failed to insert fresh Unreleased section ('# CHANGELOG' heading not found)"
    mv "${CHANGELOG}.bak" "$CHANGELOG"
    exit 1
fi
rm -f "${CHANGELOG}.bak"

echo "Updated CHANGELOG.md"

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo "DRY RUN: planned file changes:"
    git --no-pager diff -- "$VERSION_FILE" "$SETUP_FILE" "$CHANGELOG"
    echo ""
    echo "DRY RUN: reverting local edits (no commit, push, or workflow trigger)."
    git checkout -- "$VERSION_FILE" "$SETUP_FILE" "$CHANGELOG"
    echo ""
    echo "DRY RUN: would have run:"
    echo "  git commit -m 'release v${NEW_VERSION}'"
    echo "  git push ${PUSH_REMOTE} HEAD:refs/heads/${RC_BRANCH}"
    echo "  gh workflow run build_wheels.yml --repo ${REPO} --ref ${RC_BRANCH} -f upload_to_pypi=true -f test_pypi=true"
    exit 0
fi

# --- Commit locally and push RC branch (no tag, no master push yet) ---

git add "$VERSION_FILE" "$SETUP_FILE" "$CHANGELOG"
git commit -m "release v${NEW_VERSION}"
git push "$PUSH_REMOTE" "HEAD:refs/heads/${RC_BRANCH}"

# --- Trigger test-pypi build against the RC branch ---

gh workflow run build_wheels.yml --repo "$REPO" --ref "$RC_BRANCH" -f upload_to_pypi=true -f test_pypi=true

echo ""
echo "Release candidate v${NEW_VERSION} started (phase 1 of 2):"
echo "  - Local:  release commit on ${CURRENT_BRANCH}, NOT pushed"
echo "  - Remote: branch '${RC_BRANCH}' has the release commit"
echo "  - Workflow: https://github.com/${REPO}/actions?query=branch%3A${RC_BRANCH}"
echo ""
echo "Re-run this script after the workflow finishes to finalize:"
echo "  - success -> tag v${NEW_VERSION}, push ${CURRENT_BRANCH}, delete ${RC_BRANCH}"
echo "  - failure -> prompt to roll back"
