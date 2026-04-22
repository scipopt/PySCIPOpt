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

# --- Read current version ---

CURRENT_VERSION=$(sed -n "s/^__version__.*'\(.*\)'/\1/p" "$VERSION_FILE")
validate_version "$CURRENT_VERSION"
MAJOR=$(echo "$CURRENT_VERSION" | cut -d. -f1)
MINOR=$(echo "$CURRENT_VERSION" | cut -d. -f2)
PATCH=$(echo "$CURRENT_VERSION" | cut -d. -f3)

# Detect partial failure: if HEAD is already a release commit for this version,
# a previous run bumped+committed but did not finish. Re-bumping would double the version.
if [[ "$(git log -1 --format=%s)" == "release v${CURRENT_VERSION}" ]]; then
    echo "Error: HEAD is already 'release v${CURRENT_VERSION}' — a prior run did not finish."
    echo ""
    echo "Recovery options:"
    echo "  a) Nothing was pushed: git tag -d v${CURRENT_VERSION} 2>/dev/null; git reset --hard HEAD~1"
    echo "  b) Master pushed, tag not: git push ${PUSH_REMOTE} v${CURRENT_VERSION} && \\"
    echo "       gh workflow run build_wheels.yml --repo ${REPO} --ref v${CURRENT_VERSION} -f upload_to_pypi=true -f test_pypi=true"
    echo "  c) Everything pushed, workflow not triggered:"
    echo "       gh workflow run build_wheels.yml --repo ${REPO} --ref v${CURRENT_VERSION} -f upload_to_pypi=true -f test_pypi=true"
    exit 1
fi

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
echo ""
if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN: This script would:"
else
    echo "This script will:"
fi
echo "  1. Update version ${CURRENT_VERSION} -> ${NEW_VERSION} in _version.py and setup.py"
echo "  2. Update CHANGELOG.md (${NEW_VERSION} - ${TODAY})"
echo "  3. Commit, tag v${NEW_VERSION}, and push to ${PUSH_REMOTE}"
echo "  4. Trigger the build wheels workflow (test-pypi)"
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
    echo "DRY RUN: reverting local edits (no commit, tag, push, or workflow trigger)."
    git checkout -- "$VERSION_FILE" "$SETUP_FILE" "$CHANGELOG"
    echo ""
    echo "DRY RUN: would have run:"
    echo "  git commit -m 'release v${NEW_VERSION}'"
    echo "  git tag v${NEW_VERSION}"
    echo "  git push ${PUSH_REMOTE} ${CURRENT_BRANCH}"
    echo "  git push ${PUSH_REMOTE} v${NEW_VERSION}"
    echo "  gh workflow run build_wheels.yml --repo ${REPO} --ref v${NEW_VERSION} -f upload_to_pypi=true -f test_pypi=true"
    exit 0
fi

# --- Commit, tag, and push ---

git add "$VERSION_FILE" "$SETUP_FILE" "$CHANGELOG"
git commit -m "release v${NEW_VERSION}"
git tag "v${NEW_VERSION}"
git push "$PUSH_REMOTE" "$CURRENT_BRANCH"
git push "$PUSH_REMOTE" "v${NEW_VERSION}"

# --- Trigger test-pypi build ---
# --ref pins the build to the tag we just pushed so a race with a master push
# can't cause the wheel to be built from a different commit than the tag.

gh workflow run build_wheels.yml --repo "$REPO" --ref "v${NEW_VERSION}" -f upload_to_pypi=true -f test_pypi=true

echo ""
echo "Done! v${NEW_VERSION} committed, tagged, pushed, and test-pypi build triggered."
echo "Monitor at: gh run list --workflow=build_wheels.yml --repo ${REPO}"
echo ""
echo "Remaining manual steps:"
echo "  1. Test the test-pypi package:"
echo "       pip install -i https://test.pypi.org/simple/ PySCIPOpt==${NEW_VERSION}"
echo "  2. Release to production pypi:"
echo "       gh workflow run build_wheels.yml --repo ${REPO} --ref v${NEW_VERSION} -f upload_to_pypi=true -f test_pypi=false"
echo "  3. Create a GitHub release from tag v${NEW_VERSION}:"
echo "       gh release create v${NEW_VERSION} --repo ${REPO} --title v${NEW_VERSION} --generate-notes"
echo "  4. Update readthedocs: Builds -> Build version (latest and stable)"
