#!/usr/bin/env bash
set -euo pipefail

PYPROJECT="pyproject.toml"
DEPLOY_REPO="scipopt/scipoptsuite-deploy"
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

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: working directory is not clean. Commit, stash, or remove changes first."
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

CURRENT_DEPLOY_VERSION=$(grep -o 'scipoptsuite-deploy/releases/download/v[0-9.]*' "$PYPROJECT" | head -1 | sed 's|.*/||')

echo "Current scipoptsuite-deploy version: ${CURRENT_DEPLOY_VERSION}"
echo ""
echo "Enter component versions (press enter to keep current):"

# Fetch current defaults from the deploy workflow
DEPLOY_WORKFLOW=$(gh api repos/${DEPLOY_REPO}/contents/.github/workflows/build_binaries.yml --jq '.content' | base64 --decode)
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

if [[ ! "$NEW_DEPLOY_VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: deploy tag must match vX.Y.Z"
    exit 1
fi

if gh api "repos/${DEPLOY_REPO}/git/ref/tags/${NEW_DEPLOY_VERSION}" &>/dev/null; then
    echo "Error: deploy tag ${NEW_DEPLOY_VERSION} already exists in ${DEPLOY_REPO}."
    exit 1
fi

# --- Show summary and confirm ---

BRANCH="upgrade-scip-${SCIP_VERSION}"

echo ""
echo "This script will:"
echo "  1. Build new SCIP binaries (SCIP=${SCIP_VERSION} SoPlex=${SOPLEX_VERSION} GCG=${GCG_VERSION} IPOPT=${IPOPT_VERSION})"
echo "  2. Create scipoptsuite-deploy release ${NEW_DEPLOY_VERSION}"
echo "  3. Create branch '${BRANCH}', update pyproject.toml, and open a PR"
echo ""
read -rp "Proceed? [Y/n] " confirm
[[ "${confirm:-Y}" =~ ^[Nn] ]] && exit 0

# ============================================================
# From here on, everything runs without further prompts.
# ============================================================

# --- Build SCIP binaries ---

echo ""
echo "Triggering SCIP binary build..."
gh workflow run build_binaries.yml --repo "$DEPLOY_REPO" \
    -f scip_version="$SCIP_VERSION" \
    -f soplex_version="$SOPLEX_VERSION" \
    -f gcg_version="$GCG_VERSION" \
    -f ipopt_version="$IPOPT_VERSION"

# Wait for the run to appear
DISPATCH_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
for i in {1..12}; do
    sleep 5
    RUN_ID=$(gh run list --workflow=build_binaries.yml --repo "$DEPLOY_REPO" --limit 1 --event workflow_dispatch --json databaseId,createdAt --jq "[.[] | select(.createdAt >= \"${DISPATCH_TIME}\")] | .[0].databaseId")
    [[ -n "$RUN_ID" && "$RUN_ID" != "null" ]] && break
done

if [[ -z "$RUN_ID" || "$RUN_ID" == "null" ]]; then
    echo "Error: could not find the triggered workflow run."
    exit 1
fi

echo "Waiting for build to complete (run ${RUN_ID})..."
echo "  https://github.com/${DEPLOY_REPO}/actions/runs/${RUN_ID}"
gh run watch "$RUN_ID" --repo "$DEPLOY_REPO" --exit-status

# --- Create deploy release ---

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

# --- Create PR with updated pyproject.toml ---

git checkout -b "$BRANCH"

sed -i.bak "s|scipoptsuite-deploy/releases/download/${CURRENT_DEPLOY_VERSION}|scipoptsuite-deploy/releases/download/${NEW_DEPLOY_VERSION}|g" "$PYPROJECT"
rm -f "${PYPROJECT}.bak"

git add "$PYPROJECT"
git commit -m "Update scipoptsuite-deploy to ${NEW_DEPLOY_VERSION} (SCIP ${SCIP_VERSION})"
git push -u origin "$BRANCH"

gh pr create --repo "$REPO" \
    --title "Upgrade to SCIP ${SCIP_VERSION}" \
    --body "Updates scipoptsuite-deploy ${CURRENT_DEPLOY_VERSION} -> ${NEW_DEPLOY_VERSION} (SCIP ${SCIP_VERSION}, SoPlex ${SOPLEX_VERSION}, GCG ${GCG_VERSION}, IPOPT ${IPOPT_VERSION}).

Fix any API incompatibilities, get CI green, then merge and run \`./release.sh\`."

echo ""
echo "Done! PR created on branch '${BRANCH}'."
echo "Fix any API incompatibilities, get CI green, then merge and run ./release.sh"
