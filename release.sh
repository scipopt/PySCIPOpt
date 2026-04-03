#!/usr/bin/env bash
set -euo pipefail

VERSION_FILE="src/pyscipopt/_version.py"
SETUP_FILE="setup.py"
CHANGELOG="CHANGELOG.md"
REPO="scipopt/PySCIPOpt"

# --- Pre-flight checks ---

if ! command -v gh &>/dev/null; then
    echo "Error: gh CLI is not installed. Install it from https://cli.github.com"
    exit 1
fi

if ! gh auth status &>/dev/null; then
    echo "Error: gh CLI is not authenticated. Run 'gh auth login' first."
    exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: working directory has uncommitted changes. Commit or stash them first."
    exit 1
fi

CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" != "master" ]]; then
    echo "Error: must be on 'master' branch (currently on '${CURRENT_BRANCH}')."
    exit 1
fi

git pull --ff-only

# --- Read current version ---

CURRENT_VERSION=$(sed -n "s/^__version__.*'\(.*\)'/\1/p" "$VERSION_FILE")
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
    echo "Error: tag 'v${NEW_VERSION}' already exists."
    exit 1
fi

# --- Show changelog preview ---

echo ""
echo "Unreleased changelog entries:"
echo "-----------------------------"
# Print lines between "## Unreleased" and the next "## " header
sed -n '/^## Unreleased$/,/^## [0-9]/{/^## [0-9]/!p;}' "$CHANGELOG" | head -30
echo "-----------------------------"
echo ""

TODAY=$(date +%Y.%m.%d)
echo ""
echo "This script will:"
echo "  1. Update version ${CURRENT_VERSION} -> ${NEW_VERSION} in _version.py and setup.py"
echo "  2. Update CHANGELOG.md (${NEW_VERSION} - ${TODAY})"
echo "  3. Commit, tag v${NEW_VERSION}, and push to origin"
echo "  4. Trigger the build wheels workflow (test-pypi)"
echo ""
read -rp "Proceed? [Y/n] " confirm
[[ "${confirm:-Y}" =~ ^[Nn] ]] && exit 0

# --- Update version files ---

sed -i.bak "s/__version__.*=.*'.*'/__version__: str = '${NEW_VERSION}'/" "$VERSION_FILE"
rm -f "${VERSION_FILE}.bak"

sed -i.bak "s/version=\"${CURRENT_VERSION}\"/version=\"${NEW_VERSION}\"/" "$SETUP_FILE"
rm -f "${SETUP_FILE}.bak"

echo "Updated version: ${CURRENT_VERSION} -> ${NEW_VERSION}"

# --- Update changelog ---

UNRELEASED_HEADER="## Unreleased"
NEW_HEADER="## ${NEW_VERSION} - ${TODAY}"
EMPTY_UNRELEASED="## Unreleased\n### Added\n### Fixed\n### Changed\n### Removed\n"

sed -i.bak "s/^${UNRELEASED_HEADER}$/${NEW_HEADER}/" "$CHANGELOG"
rm -f "${CHANGELOG}.bak"

# Add empty Unreleased section at the top (after "# CHANGELOG" line)
sed -i.bak "/^# CHANGELOG$/a\\
\\
${EMPTY_UNRELEASED}" "$CHANGELOG"
rm -f "${CHANGELOG}.bak"

echo "Updated CHANGELOG.md"

# --- Commit, tag, and push ---

git add "$VERSION_FILE" "$SETUP_FILE" "$CHANGELOG"
git commit -m "release v${NEW_VERSION}"
git tag "v${NEW_VERSION}"
git push origin master
git push origin "v${NEW_VERSION}"

# --- Trigger test-pypi build ---

gh workflow run build_wheels.yml --repo "$REPO" -f upload_to_pypi=true -f test_pypi=true

echo ""
echo "Done! v${NEW_VERSION} committed, tagged, pushed, and test-pypi build triggered."
echo "Monitor at: gh run list --workflow=build_wheels.yml --repo ${REPO}"
echo ""
echo "Remaining manual steps:"
echo "  1. Test the test-pypi package:"
echo "       pip install -i https://test.pypi.org/simple/ PySCIPOpt==${NEW_VERSION}"
echo "  2. Release to production pypi:"
echo "       gh workflow run build_wheels.yml --repo ${REPO} -f upload_to_pypi=true -f test_pypi=false"
echo "  3. Create a GitHub release from tag v${NEW_VERSION}:"
echo "       gh release create v${NEW_VERSION} --repo ${REPO} --title v${NEW_VERSION} --generate-notes"
echo "  4. Update readthedocs: Builds -> Build version (latest and stable)"
