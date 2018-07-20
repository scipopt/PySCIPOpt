Contributing to PySCIPOpt
=========================

Code contributions are very welcome and should comply to a few rules:

0. Read `Design principles of PySCIPOpt`_.

1. Compatibility with both Python-2 and Python-3.

2. All tests defined in the Continuous Integration setup need to pass:

   - `.travis.yml <.travis.yml>`__
   - `appveyor.yml <appveyor.yml>`__

3. New features should be covered by tests *and* examples. Please extend `tests <tests>`__ and `examples <examples>`__. Tests uses pytest and examples are meant to be accessible for PySCIPOpt newcomers (even advanced examples).

4. New code should be documented in the same style as the rest of the code.

5. New code should be `pep8-compliant <https://www.python.org/dev/peps/pep-0008/>`__. Help yourself with the `style guide checker <https://pypi.org/project/pep8/>`__.

6. Before implementing a new PySCIPOpt feature, check whether the feature exists in SCIP. If so, implement it as a pure wrapper, mimicking SCIP whenever possible. If the new feature does not exist in SCIP but it is close to an existing one, consider if implementing that way is substantially convenient (e.g. Pythonic). If it does something completely different, you are welcome to pull your request and discuss the implementation.

For general reference, we suggest:

- `PySCIPOpt README <README.rst>`__;
- `SCIP documentation <http://scip.zib.de/doc/html/>`__;
- `SCIP mailing list <https://listserv.zib.de/mailman/listinfo/scip/>`__ which can be easily searched with search engines (e.g. `Google <http://www.google.com/#q=site:listserv.zib.de%2Fpipermail%2Fscip>`__);
- `open and closed PySCIPOpt issues <https://github.com/SCIP-Interfaces/PySCIPOpt/issues?utf8=%E2%9C%93&q=is%3Aissue>`__;
- `SCIP/PySCIPOpt Stack Exchange <https://stackoverflow.com/questions/tagged/scip>`__.

If you find this contributing guide unclear, please open an issue! :)

Design principles of PySCIPOpt
==============================

PySCIPOpt is meant to be a fast-prototyping interface of the pure SCIP C API. By design, we distinguish different functions in PySCIPOPT:

- pure wrapping functions of SCIP;
- convenience functions.

**PySCIPOpt wrappers of SCIP functions** should act:

- with an expected behavior - and parameters, returns, attributes, ... - as close to SCIP as possible
- without *"breaking"* Python and the purpose for what the language it is meant.

Ideally speaking, we want every SCIP function to be wrapped in PySCIPOpt.  

**Convenience functions** are additional, non-detrimental features meant to help prototyping the Python way. Since these functions are not in SCIP, we wish to limit them to prevent difference in features between SCIP and PySCIPOPT, which are always difficult to maintain. A few convenience functions survive in PySCIPOpt when keeping them is doubtless beneficial.  

Admittedly, *there is a middle ground where functions are not completely wrappers or just convenient*. That is the case, for instance, of fundamental :code:`Model` methods like :code:`addCons` or :code:`writeProblem`. We want to leave their development to negotiation.
