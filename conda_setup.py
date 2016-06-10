# simplified version of setup.py for conda recipe
from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext
from Cython.Build import cythonize

extensions = [Extension('pyscipopt.scip', ['pyscipopt/scip.pyx'],
                        extra_compile_args=['-UNDEBUG'],
                        include_dirs=['lib/scip-src'],
                        library_dirs=['lib'],
                        libraries=['m', 'scipopt', 'z', 'ipopt'])]

setup(
    name = 'pyscipopt',
    version = '0.0',
    description = 'wrapper for SCIP in Python',
    author = 'Zuse Institute Berlin',
    author_email = 'scip@zib.de',
    license = 'ZIB',
    ext_modules = cythonize(extensions),
    packages=['pyscipopt']
)
