# Release Guide
The following are the steps to follow to make a new PySCIPOpt release.
1. Check if [scipoptsuite-deploy](https://github.com/scipopt/scipoptsuite-deploy) needs a new release, if a new SCIP version is released for example, or new dependencies (change symmetry dependency, add support for papilo/ parallelization.. etc). And Update release links in `pyproject.toml`
2. Update version number according to semantic versioning [rules](https://semver.org/) in `_version.py`. 
3. Update `CHANGELOG.md`; Change the `Unlreased` to the new version number and add an empty unreleased section.
4. Create a release candidate on test-pypi by running the workflow “Build wheels” in Actions->build wheels, with these parameters `upload:true, test-pypi:true` 
5. If the pipeline passes, test the released pip package on test-pypi by running and checking that it works
```bash
pip install -i https://test.pypi.org/simple/ PySCIPOpt
```
6. If it works, release on pypi.org with running the same workflow but with `test-pypi:false`.
7. Then create a tag wit the new version (from the master branch)
```bash
git tag vX.X.X
git push origin vX.X.X
```
8. Then make a github [release](https://github.com/scipopt/PySCIPOpt/releases/new) from this new tag. 
9. Update documentation by running the `Generate Docs` workflow in Actions->Generate Docs.