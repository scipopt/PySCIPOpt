# Release Checklist

## Upgrading SCIP

Run `./upgrade_scip.sh` from the `master` branch (use `--dry-run` first to preview without side effects). The script will:
1. Prompt for SCIP, SoPlex, GCG, and IPOPT versions
2. Build new binaries via [scipoptsuite-deploy](https://github.com/scipopt/scipoptsuite-deploy) (skipped if a matching release already exists)
3. Create a branch, update `pyproject.toml`, and open a PR

On the PR:
- [ ] Fix any API incompatibilities
- [ ] Get CI green
- [ ] Update the [compatibility table](https://pyscipopt.readthedocs.io/en/latest/build.html#building-from-source) if needed
- [ ] Merge into `master`

## Releasing PySCIPOpt

Releases run in two phases from `master`, driven by `./release.sh`. The tag and master push only happen in phase 2, so an aborted release leaves no semantic public trace — just a deletable `release-candidate-vX.Y.Z` branch.

Use `--dry-run` with any command to preview without side effects.

### Phase 1 — start

```bash
./release.sh
```

Prompts for the version bump (patch/minor/major), updates `_version.py`, `setup.py`, and `CHANGELOG.md`, commits **locally**, pushes the commit to `release-candidate-vX.Y.Z` on origin, and triggers the build-wheels workflow on that branch (uploads to test-pypi). **Master is not pushed, no tag is created.** The script exits as soon as the workflow is dispatched.

To skip the bump prompt (e.g., when test-pypi has already burnt the default next version and you need to jump ahead):

```bash
./release.sh --version=X.Y.Z
```

### Manual verification

Once the release-candidate workflow finishes, install from test-pypi and smoke-test:

```bash
pip install -i https://test.pypi.org/simple/ PySCIPOpt==X.Y.Z
```

### Phase 2 — finalize or roll back

If the smoke test **passes**:

```bash
./release.sh --finalize
```

Checks the release-candidate workflow succeeded, then tags `vX.Y.Z`, pushes master, and deletes the release-candidate branch.

If the smoke test **fails** (or you change your mind):

```bash
./release.sh --rollback
```

Deletes the release-candidate branch and resets the local release commit. test-pypi has already burnt the uploaded version string, so the next attempt must use `--version=` to pick a different one.

### After finalize

- [ ] Release to production pypi:
  ```bash
  gh workflow run build_wheels.yml --repo scipopt/PySCIPOpt --ref vX.Y.Z -f upload_to_pypi=true -f test_pypi=false
  ```
- [ ] Create a GitHub release:
  ```bash
  gh release create vX.Y.Z --repo scipopt/PySCIPOpt --title vX.Y.Z --generate-notes
  ```
- [ ] Update readthedocs: Builds -> Build version (latest and stable)
