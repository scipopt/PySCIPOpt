## Reviewer checklist

Ticked by the reviewer, nothing to do here for the author.

### Code

- [ ] The code change is correct.
- [ ] New methods are named after their SCIP function (e.g. `SCIPgetVars()` becomes `getVars()`) and are placed next to related methods, **or** no new methods were added.
- [ ] The added/modified behavior is identical to SCIP's, **or** deviations were justified.
- [ ] If methods were added to `BaseModel`, they work in exact mode.
- [ ] If methods were added to `Model`, the exact counterpart was added to `ExactModel` with `Fraction` i/o, **or** the method has no meaning in exact mode or SCIP has no exact equivalent.

### Documentation

- [ ] The code is documented (numpydoc docstrings with parameter and return types).
- [ ] `CHANGELOG.md` is updated, including API changes.
- [ ] The type stubs (`scip.pyi`) are updated for new or changed methods.
- [ ] Docs and examples are updated, **or** a new docs section was added for new features (e.g. plugin types), **or** no user-facing behavior changed.

### Testing

- [ ] New code is covered by tests, including error paths.
- [ ] If this touches expressions, model building or callbacks, `/benchmark` was run and shows no regression.
  `/benchmark [filter] [timelimit=seconds]`, e.g. `/benchmark expr or plugin`. Locally: `pytest benchmarks`.

### Does this PR introduce an API change? ⚠️

- [ ] No, **or** it is backwards compatible, **or** the old behavior goes through a deprecation warning first.
