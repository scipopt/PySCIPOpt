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

Run `./release.sh` from the `master` branch (use `--dry-run` first to preview without side effects). The script will:
1. Prompt for the version bump type (patch/minor/major)
2. Update `_version.py`, `setup.py`, and `CHANGELOG.md`
3. Commit, tag, push, and trigger a test-pypi build

After the script completes:
- [ ] Test the package from test-pypi:
  ```bash
  pip install -i https://test.pypi.org/simple/ PySCIPOpt==X.Y.Z
  ```
- [ ] Release to production pypi:
  ```bash
  gh workflow run build_wheels.yml --repo scipopt/PySCIPOpt --ref vX.Y.Z -f upload_to_pypi=true -f test_pypi=false
  ```
- [ ] Create a GitHub release:
  ```bash
  gh release create vX.Y.Z --repo scipopt/PySCIPOpt --title vX.Y.Z --generate-notes
  ```
- [ ] Update readthedocs: Builds -> Build version (latest and stable)
