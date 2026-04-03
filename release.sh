#!/usr/bin/env bash
set -euo pipefail

VERSION_FILE="src/pyscipopt/_version.py"
SETUP_FILE="setup.py"
PYPROJECT="pyproject.toml"
CHANGELOG="CHANGELOG.md"
REPO="scipopt/PySCIPOpt"
DEPLOY_REPO="scipopt/scipoptsuite-deploy"

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

# --- Helper functions ---

validate_version() {
    if [[ ! "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Error: '$1' is not a valid version (expected X.Y.Z)"
        exit 1
    fi
}

prompt_version() {
    local label="$1" current="$2"
    read -rp "${label} [${current}]: " value
    value="${value:-$current}"
    validate_version "$value"
    echo "$value"
}

# --- Collect all inputs ---

# 1. New SCIP binaries?

CURRENT_DEPLOY_VERSION=$(grep -o 'scipoptsuite-deploy/releases/download/v[0-9.]*' "$PYPROJECT" | head -1 | sed 's|.*/||')

echo "Current scipoptsuite-deploy version: ${CURRENT_DEPLOY_VERSION}"
read -rp "Does this release need new SCIP binaries? [y/N] " need_deploy

NEED_DEPLOY=false
if [[ "${need_deploy:-N}" =~ ^[Yy] ]]; then
    NEED_DEPLOY=true

    echo ""
    echo "Enter component versions (press enter to keep current):"

    # Fetch current defaults from the deploy workflow
    DEPLOY_WORKFLOW=$(gh api repos/${DEPLOY_REPO}/contents/.github/workflows/build_binaries.yml --jq '.content' | base64 -d)
    current_deploy_default() {
        echo "$DEPLOY_WORKFLOW" | sed -n "/${1}:/,/default:/{s/.*default: \"\(.*\)\"/\1/p;}" | head -1
    }

    CUR_SCIP=$(current_deploy_default "scip_version")
    CUR_SOPLEX=$(current_deploy_default "soplex_version")
    CUR_GCG=$(current_deploy_default "gcg_version")
    CUR_IPOPT=$(current_deploy_default "ipopt_version")

    SCIP_VERSION=$(prompt_version "SCIP" "$CUR_SCIP")
    SOPLEX_VERSION=$(prompt_version "SoPlex" "$CUR_SOPLEX")
    GCG_VERSION=$(prompt_version "GCG" "$CUR_GCG")
    IPOPT_VERSION=$(prompt_version "IPOPT" "$CUR_IPOPT")

    # Bump deploy version (increment minor)
    DEPLOY_MAJOR=$(echo "$CURRENT_DEPLOY_VERSION" | sed 's/^v//' | cut -d. -f1)
    DEPLOY_MINOR=$(echo "$CURRENT_DEPLOY_VERSION" | sed 's/^v//' | cut -d. -f2)
    DEPLOY_PATCH=$(echo "$CURRENT_DEPLOY_VERSION" | sed 's/^v//' | cut -d. -f3)
    SUGGESTED_DEPLOY="v$((DEPLOY_MAJOR)).$((DEPLOY_MINOR + 1)).$((DEPLOY_PATCH))"

    read -rp "New deploy release tag [${SUGGESTED_DEPLOY}]: " NEW_DEPLOY_VERSION
    NEW_DEPLOY_VERSION="${NEW_DEPLOY_VERSION:-$SUGGESTED_DEPLOY}"
fi

# 2. PySCIPOpt version bump

CURRENT_VERSION=$(sed -n "s/^__version__.*'\(.*\)'/\1/p" "$VERSION_FILE")
MAJOR=$(echo "$CURRENT_VERSION" | cut -d. -f1)
MINOR=$(echo "$CURRENT_VERSION" | cut -d. -f2)
PATCH=$(echo "$CURRENT_VERSION" | cut -d. -f3)

echo ""
echo "Current PySCIPOpt version: ${CURRENT_VERSION}"
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

if git rev-parse "v${NEW_VERSION}" &>/dev/null; then
    echo "Error: tag 'v${NEW_VERSION}' already exists."
    exit 1
fi

# --- Show summary and confirm ---

echo ""
echo "Unreleased changelog entries:"
echo "-----------------------------"
sed -n '/^## Unreleased$/,/^## [0-9]/{/^## [0-9]/!p;}' "$CHANGELOG" | head -30
echo "-----------------------------"

TODAY=$(date +%Y.%m.%d)
echo ""
echo "This script will:"
if [[ "$NEED_DEPLOY" == true ]]; then
    echo "  1. Build new SCIP binaries (SCIP=${SCIP_VERSION} SoPlex=${SOPLEX_VERSION} GCG=${GCG_VERSION} IPOPT=${IPOPT_VERSION})"
    echo "  2. Create scipoptsuite-deploy release ${NEW_DEPLOY_VERSION}"
    echo "  3. Update deploy version ${CURRENT_DEPLOY_VERSION} -> ${NEW_DEPLOY_VERSION} in pyproject.toml"
    echo "  4. Update PySCIPOpt version ${CURRENT_VERSION} -> ${NEW_VERSION} in _version.py and setup.py"
    echo "  5. Update CHANGELOG.md (${NEW_VERSION} - ${TODAY})"
    echo "  6. Commit, tag v${NEW_VERSION}, and push to origin"
    echo "  7. Trigger the build wheels workflow (test-pypi)"
else
    echo "  1. Update version ${CURRENT_VERSION} -> ${NEW_VERSION} in _version.py and setup.py"
    echo "  2. Update CHANGELOG.md (${NEW_VERSION} - ${TODAY})"
    echo "  3. Commit, tag v${NEW_VERSION}, and push to origin"
    echo "  4. Trigger the build wheels workflow (test-pypi)"
fi
echo ""
read -rp "Proceed? [Y/n] " confirm
[[ "${confirm:-Y}" =~ ^[Nn] ]] && exit 0

# ============================================================
# From here on, everything runs without further prompts.
# ============================================================

# --- Build and release SCIP binaries (if needed) ---

if [[ "$NEED_DEPLOY" == true ]]; then
    echo ""
    echo "Triggering SCIP binary build..."
    gh workflow run build_binaries.yml --repo "$DEPLOY_REPO" \
        -f scip_version="$SCIP_VERSION" \
        -f soplex_version="$SOPLEX_VERSION" \
        -f gcg_version="$GCG_VERSION" \
        -f ipopt_version="$IPOPT_VERSION"

    # Wait for the run to appear
    sleep 5
    RUN_ID=$(gh run list --workflow=build_binaries.yml --repo "$DEPLOY_REPO" --limit 1 --json databaseId --jq '.[0].databaseId')

    echo "Waiting for build to complete (run ${RUN_ID})..."
    echo "  https://github.com/${DEPLOY_REPO}/actions/runs/${RUN_ID}"
    gh run watch "$RUN_ID" --repo "$DEPLOY_REPO" --exit-status

    # Download artifacts and create release
    TMPDIR=$(mktemp -d)
    echo "Downloading artifacts..."
    gh run download "$RUN_ID" --repo "$DEPLOY_REPO" --dir "$TMPDIR"

    RELEASE_NAME="SCIP ${SCIP_VERSION} SOPLEX ${SOPLEX_VERSION} GCG ${GCG_VERSION} IPOPT ${IPOPT_VERSION}"
    echo "Creating release ${NEW_DEPLOY_VERSION}..."
    gh release create "$NEW_DEPLOY_VERSION" \
        --repo "$DEPLOY_REPO" \
        --title "$RELEASE_NAME" \
        --notes "$RELEASE_NAME" \
        "$TMPDIR"/linux/*.zip \
        "$TMPDIR"/linux-arm/*.zip \
        "$TMPDIR"/macos-arm/*.zip \
        "$TMPDIR"/macos-intel/*.zip \
        "$TMPDIR"/windows/*.zip

    rm -rf "$TMPDIR"

    # Update deploy version in pyproject.toml
    sed -i.bak "s|scipoptsuite-deploy/releases/download/${CURRENT_DEPLOY_VERSION}|scipoptsuite-deploy/releases/download/${NEW_DEPLOY_VERSION}|g" "$PYPROJECT"
    rm -f "${PYPROJECT}.bak"
    echo "Updated pyproject.toml: ${CURRENT_DEPLOY_VERSION} -> ${NEW_DEPLOY_VERSION}"
fi

# --- Update version files ---

sed -i.bak "s/__version__.*=.*'.*'/__version__: str = '${NEW_VERSION}'/" "$VERSION_FILE"
rm -f "${VERSION_FILE}.bak"

sed -i.bak "s/version=\"${CURRENT_VERSION}\"/version=\"${NEW_VERSION}\"/" "$SETUP_FILE"
rm -f "${SETUP_FILE}.bak"

echo "Updated version: ${CURRENT_VERSION} -> ${NEW_VERSION}"

# --- Update changelog ---

sed -i.bak "s/^## Unreleased$/## ${NEW_VERSION} - ${TODAY}/" "$CHANGELOG"
rm -f "${CHANGELOG}.bak"

sed -i.bak "/^# CHANGELOG$/a\\
\\
## Unreleased\\
### Added\\
### Fixed\\
### Changed\\
### Removed\\
" "$CHANGELOG"
rm -f "${CHANGELOG}.bak"

echo "Updated CHANGELOG.md"

# --- Commit, tag, and push ---

FILES_TO_COMMIT=("$VERSION_FILE" "$SETUP_FILE" "$CHANGELOG")
[[ "$NEED_DEPLOY" == true ]] && FILES_TO_COMMIT+=("$PYPROJECT")

git add "${FILES_TO_COMMIT[@]}"
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
