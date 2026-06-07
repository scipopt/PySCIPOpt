#!/usr/bin/env bash
set -euo pipefail

PYPROJECT="pyproject.toml"
DEPLOY_REPO="scipopt/scipoptsuite-deploy"
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

validate_deploy_version() {
    if [[ ! "$1" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Error: '$1' is not a valid deploy version (expected vX.Y.Z)"
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

if [[ -z "$CURRENT_DEPLOY_VERSION" ]]; then
    echo "Error: could not find a scipoptsuite-deploy/releases/download/vX.Y.Z URL in ${PYPROJECT}."
    exit 1
fi
validate_deploy_version "$CURRENT_DEPLOY_VERSION"

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

# --- Check if a matching deploy release already exists ---

RELEASE_NAME="SCIP ${SCIP_VERSION} SOPLEX ${SOPLEX_VERSION} GCG ${GCG_VERSION} IPOPT ${IPOPT_VERSION}"
EXISTING_TAG=$(gh release list --repo "$DEPLOY_REPO" --limit 20 --json tagName,name \
    --jq ".[] | select(.name == \"${RELEASE_NAME}\") | .tagName" | head -1)

SKIP_DEPLOY=false
if [[ -n "$EXISTING_TAG" ]]; then
    echo "Found existing release '${EXISTING_TAG}' matching these versions. Skipping build."
    NEW_DEPLOY_VERSION="$EXISTING_TAG"
    SKIP_DEPLOY=true
else
    # Bump deploy version (increment minor)
    DEPLOY_MAJOR=$(echo "$CURRENT_DEPLOY_VERSION" | sed 's/^v//' | cut -d. -f1)
    DEPLOY_MINOR=$(echo "$CURRENT_DEPLOY_VERSION" | sed 's/^v//' | cut -d. -f2)
    DEPLOY_PATCH=$(echo "$CURRENT_DEPLOY_VERSION" | sed 's/^v//' | cut -d. -f3)
    SUGGESTED_DEPLOY="v$((DEPLOY_MAJOR)).$((DEPLOY_MINOR + 1)).$((DEPLOY_PATCH))"

    read -rp "New deploy release tag [${SUGGESTED_DEPLOY}]: " NEW_DEPLOY_VERSION
    NEW_DEPLOY_VERSION="${NEW_DEPLOY_VERSION:-$SUGGESTED_DEPLOY}"

    validate_deploy_version "$NEW_DEPLOY_VERSION"

    if gh release view "$NEW_DEPLOY_VERSION" --repo "$DEPLOY_REPO" &>/dev/null; then
        echo "Error: deploy tag ${NEW_DEPLOY_VERSION} already exists in ${DEPLOY_REPO} (with different versions)."
        exit 1
    fi
fi

if [[ "$CURRENT_DEPLOY_VERSION" == "$NEW_DEPLOY_VERSION" ]]; then
    echo "Error: new deploy version (${NEW_DEPLOY_VERSION}) matches current; nothing to upgrade."
    exit 1
fi

# --- Show summary and confirm ---

BRANCH="upgrade-scip-${SCIP_VERSION}"

echo ""
if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN: This script would:"
else
    echo "This script will:"
fi
if [[ "$SKIP_DEPLOY" == false ]]; then
    echo "  1. Build new SCIP binaries (SCIP=${SCIP_VERSION} SoPlex=${SOPLEX_VERSION} GCG=${GCG_VERSION} IPOPT=${IPOPT_VERSION})"
    echo "  2. Create scipoptsuite-deploy release ${NEW_DEPLOY_VERSION}"
    echo "  3. Create branch '${BRANCH}', update pyproject.toml, and open a PR"
else
    echo "  1. [skip] Build binaries — release ${NEW_DEPLOY_VERSION} already exists"
    echo "  2. [skip] Create release — already exists"
    echo "  3. Create branch '${BRANCH}', update pyproject.toml, and open a PR"
fi
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN: would have run:"
    if [[ "$SKIP_DEPLOY" == false ]]; then
        echo "  gh workflow run build_binaries.yml --repo ${DEPLOY_REPO} \\"
        echo "      -f scip_version=${SCIP_VERSION} -f soplex_version=${SOPLEX_VERSION} \\"
        echo "      -f gcg_version=${GCG_VERSION} -f ipopt_version=${IPOPT_VERSION}"
        echo "  (wait for run, download artifacts)"
        echo "  gh release create ${NEW_DEPLOY_VERSION} --repo ${DEPLOY_REPO} ..."
    fi
    echo "  git checkout -b ${BRANCH}"
    echo "  (sed) ${CURRENT_DEPLOY_VERSION} -> ${NEW_DEPLOY_VERSION} in ${PYPROJECT}"
    echo "  git commit -m 'Update scipoptsuite-deploy to ${NEW_DEPLOY_VERSION} (SCIP ${SCIP_VERSION})'"
    echo "  git push -u ${PUSH_REMOTE} ${BRANCH}"
    echo "  gh pr create --repo ${REPO} --title 'Upgrade to SCIP ${SCIP_VERSION}'"
    exit 0
fi

read -rp "Proceed? [Y/n] " confirm
[[ "${confirm:-Y}" =~ ^[Nn] ]] && exit 0

# ============================================================
# From here on, everything runs without further prompts.
# ============================================================

if [[ "$SKIP_DEPLOY" == false ]]; then

    ARTIFACT_DIR=""
    cleanup() { [[ -n "$ARTIFACT_DIR" ]] && rm -rf "$ARTIFACT_DIR"; }
    trap cleanup EXIT

    # --- Build SCIP binaries ---

    echo ""
    echo "Triggering SCIP binary build..."
    DISPATCH_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    gh workflow run build_binaries.yml --repo "$DEPLOY_REPO" \
        -f scip_version="$SCIP_VERSION" \
        -f soplex_version="$SOPLEX_VERSION" \
        -f gcg_version="$GCG_VERSION" \
        -f ipopt_version="$IPOPT_VERSION"

    # Wait for the run to appear. Filter by actor and earliest createdAt so a
    # concurrent workflow_dispatch (by us or someone else) can't hijack RUN_ID.
    MY_LOGIN=$(gh api user --jq .login)
    for i in {1..12}; do
        sleep 5
        RUN_ID=$(gh run list --workflow=build_binaries.yml --repo "$DEPLOY_REPO" \
            --limit 20 --event workflow_dispatch \
            --json databaseId,createdAt,actor \
            --jq "[.[] | select(.createdAt >= \"${DISPATCH_TIME}\") | select(.actor.login == \"${MY_LOGIN}\")] | sort_by(.createdAt) | .[0].databaseId")
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

    ARTIFACT_DIR=$(mktemp -d)
    echo "Downloading artifacts..."
    gh run download "$RUN_ID" --repo "$DEPLOY_REPO" --dir "$ARTIFACT_DIR"

    shopt -s nullglob
    for subdir in linux linux-arm macos-arm macos-intel windows; do
        zips=("$ARTIFACT_DIR/$subdir"/*.zip)
        if [[ ${#zips[@]} -eq 0 ]]; then
            echo "Error: no .zip files found in $ARTIFACT_DIR/$subdir/ — artifact layout may have changed."
            exit 1
        fi
    done
    shopt -u nullglob

    RELEASE_NAME="SCIP ${SCIP_VERSION} SOPLEX ${SOPLEX_VERSION} GCG ${GCG_VERSION} IPOPT ${IPOPT_VERSION}"
    echo "Creating release ${NEW_DEPLOY_VERSION}..."
    gh release create "$NEW_DEPLOY_VERSION" \
        --repo "$DEPLOY_REPO" \
        --title "$RELEASE_NAME" \
        --notes "$RELEASE_NAME" \
        "$ARTIFACT_DIR"/linux/*.zip \
        "$ARTIFACT_DIR"/linux-arm/*.zip \
        "$ARTIFACT_DIR"/macos-arm/*.zip \
        "$ARTIFACT_DIR"/macos-intel/*.zip \
        "$ARTIFACT_DIR"/windows/*.zip

    rm -rf "$ARTIFACT_DIR"
    ARTIFACT_DIR=""

fi

# --- Create PR with updated pyproject.toml ---

if git rev-parse --verify "$BRANCH" &>/dev/null; then
    read -rp "Branch '$BRANCH' already exists locally. Delete it? [y/N] " del_branch
    if [[ "${del_branch:-N}" =~ ^[Yy] ]]; then
        git branch -D "$BRANCH"
    else
        echo "Aborting. Delete the branch manually and re-run."
        exit 1
    fi
fi

if git ls-remote --heads --exit-code "$PUSH_REMOTE" "$BRANCH" &>/dev/null; then
    echo "Error: branch '$BRANCH' already exists on ${PUSH_REMOTE}."
    echo "Delete it remotely first: git push ${PUSH_REMOTE} --delete ${BRANCH}"
    exit 1
fi

git checkout -b "$BRANCH"

sed -i.bak "s|scipoptsuite-deploy/releases/download/${CURRENT_DEPLOY_VERSION}|scipoptsuite-deploy/releases/download/${NEW_DEPLOY_VERSION}|g" "$PYPROJECT"
if cmp -s "$PYPROJECT" "${PYPROJECT}.bak"; then
    echo "Error: failed to update ${CURRENT_DEPLOY_VERSION} -> ${NEW_DEPLOY_VERSION} in ${PYPROJECT} (pattern not found)."
    mv "${PYPROJECT}.bak" "$PYPROJECT"
    git checkout master
    git branch -D "$BRANCH"
    exit 1
fi
rm -f "${PYPROJECT}.bak"

git add "$PYPROJECT"
git commit -m "Update scipoptsuite-deploy to ${NEW_DEPLOY_VERSION} (SCIP ${SCIP_VERSION})"
git push -u "$PUSH_REMOTE" "$BRANCH"

gh pr create --repo "$REPO" \
    --title "Upgrade to SCIP ${SCIP_VERSION}" \
    --body "$(cat <<EOF
Updates scipoptsuite-deploy ${CURRENT_DEPLOY_VERSION} -> ${NEW_DEPLOY_VERSION} (SCIP ${SCIP_VERSION}, SoPlex ${SOPLEX_VERSION}, GCG ${GCG_VERSION}, IPOPT ${IPOPT_VERSION}).

## Checklist
- [ ] Fix any API incompatibilities
- [ ] CI is green
- [ ] Update [compatibility table](https://pyscipopt.readthedocs.io/en/latest/build.html#building-from-source) if needed
- [ ] Merge and run \`./release.sh\`
EOF
)"

echo ""
echo "Done! PR created on branch '${BRANCH}'."
echo "Note: you are now on branch '${BRANCH}'. Switch back with: git checkout master"
echo "Fix any API incompatibilities, get CI green, then merge and run ./release.sh"
