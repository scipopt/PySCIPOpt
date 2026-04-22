#!/usr/bin/env bash
set -euo pipefail

VERSION_FILE="src/pyscipopt/_version.py"
SETUP_FILE="setup.py"
CHANGELOG="CHANGELOG.md"
REPO="scipopt/PySCIPOpt"

DRY_RUN=false
ACTION=start
NEW_VERSION_OVERRIDE=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --finalize) ACTION=finalize ;;
        --rollback) ACTION=rollback ;;
        --version=*) NEW_VERSION_OVERRIDE="${arg#--version=}" ;;
        -h|--help)
            cat <<USAGE
Usage: $0 [--dry-run] [--version=X.Y.Z] [--finalize | --rollback]

  (default)        Start a new release: bump version, push the staging branch,
                   trigger the test-pypi build.
  --version=X.Y.Z  Use this exact version instead of prompting for patch/minor/major.
                   Useful if test-pypi has already burnt the default next version.
  --finalize       Promote: tag vX.Y.Z, push master, delete the staging branch.
                   Requires the staging workflow to have succeeded.
  --rollback       Abandon: delete the staging branch and reset the local commit.
  --dry-run        Preview without side effects. Combinable with the flags above.
USAGE
            exit 0
            ;;
        *) echo "Error: unknown argument '$arg' (see --help)"; exit 1 ;;
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

# Promote a successful pending release: tag, push master, clean up staging branch.
finalize_release() {
    local version="$1"
    local staging_branch="staging-v${version}"

    echo "Pending release: v${version}"
    echo "Checking workflow status on ${staging_branch}..."

    local run_data run_count status conclusion url run_id
    run_data=$(gh run list --workflow=build_wheels.yml --repo "$REPO" \
        --branch "$staging_branch" --limit 1 \
        --json databaseId,status,conclusion,url 2>/dev/null || echo "[]")
    run_count=$(echo "$run_data" | jq 'length')

    if [[ "$run_count" == "0" ]]; then
        echo "Error: no workflow run found for branch '${staging_branch}'."
        echo "Either wait a moment and try again, or clean up manually:"
        echo "  git push ${PUSH_REMOTE} --delete ${staging_branch}"
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
            echo "DRY RUN: would prompt to roll back (delete ${staging_branch} + reset local commit)."
            exit 0
        fi
        read -rp "Roll back (delete ${staging_branch} and reset local release commit)? [Y/n] " confirm
        if [[ ! "${confirm:-Y}" =~ ^[Nn] ]]; then
            git push "$PUSH_REMOTE" --delete "$staging_branch" || echo "(note: remote staging branch delete failed; clean up manually)"
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
        echo "  git push ${PUSH_REMOTE} --delete ${staging_branch}"
        exit 0
    fi

    echo "Workflow succeeded. Promoting release v${version}..."
    git tag "v${version}"
    git push "$PUSH_REMOTE" "$CURRENT_BRANCH"
    git push "$PUSH_REMOTE" "v${version}"
    git push "$PUSH_REMOTE" --delete "$staging_branch"

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

# Abandon a pending release: delete staging branch, reset local release commit.
rollback_release() {
    local version="$1"
    local staging_branch="staging-v${version}"

    echo "Rolling back pending release v${version}..."
    if [[ "$DRY_RUN" == true ]]; then
        echo "DRY RUN: would have run:"
        echo "  git push ${PUSH_REMOTE} --delete ${staging_branch}"
        echo "  git reset --hard HEAD~1"
        exit 0
    fi

    if git ls-remote --heads --exit-code "$PUSH_REMOTE" "$staging_branch" &>/dev/null; then
        git push "$PUSH_REMOTE" --delete "$staging_branch"
    else
        echo "(note: ${staging_branch} already absent from ${PUSH_REMOTE})"
    fi
    git reset --hard HEAD~1
    echo "Rolled back. Repo is at pre-release state."
}

# Validate that HEAD is a local release commit; return the version via stdout.
require_pending_release() {
    local head_msg
    head_msg=$(git log -1 --format=%s)
    if [[ ! "$head_msg" =~ ^release\ v([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
        echo "Error: expected HEAD to be a 'release vX.Y.Z' commit, got: '${head_msg}'" >&2
        echo "Run without --finalize/--rollback to start a new release." >&2
        exit 1
    fi
    echo "${BASH_REMATCH[1]}"
}

# --- Dispatch finalize / rollback / start ---

if [[ "$ACTION" == "finalize" ]]; then
    PENDING_VERSION=$(require_pending_release)
    finalize_release "$PENDING_VERSION"
    exit 0
fi

if [[ "$ACTION" == "rollback" ]]; then
    PENDING_VERSION=$(require_pending_release)
    rollback_release "$PENDING_VERSION"
    exit 0
fi

# Start path: if HEAD is already a release commit, the user probably forgot a flag.
HEAD_MSG=$(git log -1 --format=%s)
if [[ "$HEAD_MSG" =~ ^release\ v([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
    echo "Error: HEAD is already 'release v${BASH_REMATCH[1]}' — a pending release exists."
    echo "Use:"
    echo "  ./release.sh --finalize   # promote (requires the staging workflow to have succeeded)"
    echo "  ./release.sh --rollback   # abandon (delete staging branch, reset local commit)"
    exit 1
fi

# --- Read current version ---

CURRENT_VERSION=$(sed -n "s/^__version__.*'\(.*\)'/\1/p" "$VERSION_FILE")
validate_version "$CURRENT_VERSION"
MAJOR=$(echo "$CURRENT_VERSION" | cut -d. -f1)
MINOR=$(echo "$CURRENT_VERSION" | cut -d. -f2)
PATCH=$(echo "$CURRENT_VERSION" | cut -d. -f3)

echo "Current version: ${CURRENT_VERSION}"

# --- Determine new version (prompt, or use --version=X.Y.Z override) ---

if [[ -n "$NEW_VERSION_OVERRIDE" ]]; then
    validate_version "$NEW_VERSION_OVERRIDE"
    NEW_VERSION="$NEW_VERSION_OVERRIDE"
    echo "Using version (from --version): ${NEW_VERSION}"
else
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
fi

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
STAGING_BRANCH="staging-v${NEW_VERSION}"
echo ""
if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN: This script would:"
else
    echo "This script will (phase 1 of 2):"
fi
echo "  1. Update version ${CURRENT_VERSION} -> ${NEW_VERSION} in _version.py and setup.py"
echo "  2. Update CHANGELOG.md (${NEW_VERSION} - ${TODAY})"
echo "  3. Commit locally, push commit to branch '${STAGING_BRANCH}' on ${PUSH_REMOTE}"
echo "     (master is NOT pushed; no tag is created yet)"
echo "  4. Trigger the build wheels workflow on that branch (test-pypi)"
echo ""
echo "Once the workflow finishes, re-run this script:"
echo "  - success -> tag v${NEW_VERSION}, push master, delete staging branch"
echo "  - failure -> prompt to roll back (delete staging branch + reset local commit)"
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
    echo "  git push ${PUSH_REMOTE} HEAD:refs/heads/${STAGING_BRANCH}"
    echo "  gh workflow run build_wheels.yml --repo ${REPO} --ref ${STAGING_BRANCH} -f upload_to_pypi=true -f test_pypi=true"
    exit 0
fi

# --- Commit locally and push staging branch (no tag, no master push yet) ---

git add "$VERSION_FILE" "$SETUP_FILE" "$CHANGELOG"
git commit -m "release v${NEW_VERSION}"
git push "$PUSH_REMOTE" "HEAD:refs/heads/${STAGING_BRANCH}"

# --- Trigger test-pypi build against the staging branch ---

gh workflow run build_wheels.yml --repo "$REPO" --ref "$STAGING_BRANCH" -f upload_to_pypi=true -f test_pypi=true

echo ""
echo "Release candidate v${NEW_VERSION} started (phase 1 of 2):"
echo "  - Local:  release commit on ${CURRENT_BRANCH}, NOT pushed"
echo "  - Remote: branch '${STAGING_BRANCH}' has the release commit"
echo "  - Workflow: https://github.com/${REPO}/actions?query=branch%3A${STAGING_BRANCH}"
echo ""
echo "Re-run this script after the workflow finishes to finalize:"
echo "  - success -> tag v${NEW_VERSION}, push ${CURRENT_BRANCH}, delete ${STAGING_BRANCH}"
echo "  - failure -> prompt to roll back"
