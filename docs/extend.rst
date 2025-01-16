########################
Extending the Interface
########################

PySCIPOpt covers many of the SCIP callable library methods, however it is not complete.
One can extend the interface to encompass even more functionality that is possible with
SCIP. The following will provide some directions on how this can be achieved.

.. contents:: Contents

General File Structure
======================

The two most important files in PySCIPOpt are the ``scip.pxd`` and
``scip.pxi``. These two files specify the public functions of SCIP that
can be accessed from your python code.

To make PySCIPOpt aware of the public functions that one can access,
one must add them to ``scip.pxd``. There are two things that must
be done in order to properly add the functions:

- Ensure any ``enum``, ``struct`` or SCIP variable types are included in ``scip.pxd``
- Add the prototype of the public function you wish to access to ``scip.pxd``

After following the previous two steps, it is then possible to create
functions in python that reference the SCIP public functions included in
``scip.pxd``. This is achieved by modifying the `scip.pxi` file to add the
functionality you require.

Contribution Guidelines
=======================

Code contributions are very welcome and should comply to a few rules:

- Read Design Principles of PySCIPOpt in the section below
- New features should be covered by tests and examples. Please extend
  the tests and example appropriately. Tests uses pytest
  and examples are meant to be accessible for PySCIPOpt newcomers
  (even advanced examples).
- New code should be documented in the same style as the rest of the code.
- New features or bugfixes have to be documented in the CHANGELOG.
- New code should be `pep8-compliant <https://www.python.org/dev/peps/pep-0008/>`_. Help
  yourself with the `style guide checker <https://pypi.org/project/pep8/>`_.
- Before implementing a new PySCIPOpt feature, check whether the
  feature exists in SCIP. If so, implement it as a pure wrapper,
  mimicking SCIP whenever possible. If the new feature does not exist
  in SCIP but it is close to an existing one, consider if implementing
  that way is substantially convenient (e.g. Pythonic). If it does
  something completely different, you are welcome to pull your request
  and discuss the implementation.
- PySCIPOpt uses `semantic versioning <https://semver.org/>`_. Version
  number increase only happens on master and must be tagged to build a new PyPI release.

For general reference, we suggest:

- `SCIP documentation <http://scip.zib.de/doc/html/>`_
- `SCIP mailing list <https://listserv.zib.de/mailman/listinfo/scip/>`_
- `open and closed PySCIPOpt issues <https://github.com/scipopt/PySCIPOpt/issues>`_
- `SCIP/PySCIPOpt Stack Exchange <https://stackoverflow.com/questions/tagged/scip>`_

If you find this contributing guide unclear, please open an issue! :)

Design Principles of PySCIPOpt
==============================

PySCIPOpt is meant to be a fast-prototyping interface of the pure SCIP C
API. By design, we distinguish different functions in PySCIPOPT:

- pure wrapping functions of SCIP;
- convenience functions.

**PySCIPOpt wrappers of SCIP functions** should act:

- with an expected behavior - and parameters, returns, attributes, ... - as close to SCIP as possible
- without **"breaking"** Python and the purpose for what the language it is meant.

Ideally speaking, we want every SCIP function to be wrapped in PySCIPOpt.

**Convenience functions** are additional, non-detrimental features meant
to help prototyping the Python way. Since these functions are not in
SCIP, we wish to limit them to prevent difference in features between
SCIP and PySCIPOPT, which are always difficult to maintain. A few
convenience functions survive in PySCIPOpt when keeping them is
doubtless beneficial.

Admittedly, there is a middle ground where functions are not completely
wrappers or just convenient. That is the case, for instance, of
fundamental ``Model`` methods like ``addCons`` or
``writeProblem``. We want to leave their development to negotiation.