#!/usr/bin/env bash
set -euo pipefail

PYPROJECT="pyproject.toml"
WORKFLOWS=(.github/workflows/*.yml)
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

# --- Pick the scipoptsuite-deploy release ---
# Deploy releases are made in scipoptsuite-deploy (its release.sh); this only picks one.

CURRENT_DEPLOY_VERSION=$(grep -o 'scipoptsuite-deploy/releases/download/v[0-9.]*' "$PYPROJECT" | head -1 | sed 's|.*/||')

if [[ -z "$CURRENT_DEPLOY_VERSION" ]]; then
    echo "Error: could not find a scipoptsuite-deploy/releases/download/vX.Y.Z URL in ${PYPROJECT}."
    exit 1
fi
validate_deploy_version "$CURRENT_DEPLOY_VERSION"
echo "Current scipoptsuite-deploy release: ${CURRENT_DEPLOY_VERSION}"

LATEST_DEPLOY=$(gh release view --repo "$DEPLOY_REPO" --json tagName -q .tagName)
read -rp "scipoptsuite-deploy release to use [${LATEST_DEPLOY}]: " NEW_DEPLOY_VERSION
NEW_DEPLOY_VERSION="${NEW_DEPLOY_VERSION:-$LATEST_DEPLOY}"
validate_deploy_version "$NEW_DEPLOY_VERSION"

if [[ "$CURRENT_DEPLOY_VERSION" == "$NEW_DEPLOY_VERSION" ]]; then
    echo "Error: ${NEW_DEPLOY_VERSION} is already used; nothing to upgrade."
    echo "To build new SCIP binaries, make a release in ${DEPLOY_REPO} first (its release.sh)."
    exit 1
fi

# release titles look like "SCIP 10.1.0 SOPLEX 8.1.0 GCG 4.1.0 IPOPT 3.14.19"
RELEASE_NAME=$(gh release view "$NEW_DEPLOY_VERSION" --repo "$DEPLOY_REPO" --json name -q .name)
SCIP_VERSION=$(echo "$RELEASE_NAME" | sed -n 's/^SCIP \([0-9.]*\) .*/\1/p')
validate_version "$SCIP_VERSION"

# --- Show summary and confirm ---

BRANCH="upgrade-scip-${SCIP_VERSION}"

echo ""
echo "${NEW_DEPLOY_VERSION}: ${RELEASE_NAME}"
echo ""
if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN: would have run:"
    echo "  git checkout -b ${BRANCH}"
    echo "  (sed) ${CURRENT_DEPLOY_VERSION} -> ${NEW_DEPLOY_VERSION} in ${PYPROJECT}"
    echo "  (sed) 'version: ${SCIP_VERSION}' in $(grep -l '^  version: ' "${WORKFLOWS[@]}" | tr '\n' ' ')"
    echo "  git commit -m 'Update scipoptsuite-deploy to ${NEW_DEPLOY_VERSION} (SCIP ${SCIP_VERSION})'"
    echo "  git push -u ${PUSH_REMOTE} ${BRANCH}"
    echo "  gh pr create --repo ${REPO} --title 'Upgrade to SCIP ${SCIP_VERSION}'"
    exit 0
fi

echo "This script will create branch '${BRANCH}', update ${PYPROJECT} and the CI SCIP version, and open a PR."
read -rp "Proceed? [Y/n] " confirm
[[ "${confirm:-Y}" =~ ^[Nn] ]] && exit 0

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

for wf in "${WORKFLOWS[@]}"; do
    if grep -q '^  version: ' "$wf"; then
        sed -i.bak "s/^  version: .*/  version: ${SCIP_VERSION}/" "$wf"
        rm -f "${wf}.bak"
    fi
done

git add "$PYPROJECT" "${WORKFLOWS[@]}"
git commit -m "Update scipoptsuite-deploy to ${NEW_DEPLOY_VERSION} (SCIP ${SCIP_VERSION})"
git push -u "$PUSH_REMOTE" "$BRANCH"

gh pr create --repo "$REPO" \
    --title "Upgrade to SCIP ${SCIP_VERSION}" \
    --body "$(cat <<EOF
Updates scipoptsuite-deploy ${CURRENT_DEPLOY_VERSION} -> ${NEW_DEPLOY_VERSION} (${RELEASE_NAME}) and the SCIP version used by CI workflows.

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
