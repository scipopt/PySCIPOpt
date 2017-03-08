from setuptools import setup, Extension
import os

cythonize = True
dllname = 'libscip-4.0.0.mingw.x86_64.msvc.opt.spx2'
includedir = 'include'
libdir = 'lib'

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    cythonize = False

if not os.path.exists(os.path.join('pyscipopt', 'scip.pyx')):
    cythonize = False

extensions = []
ext = '.pyx' if cythonize else '.c'

extensions = [Extension('pyscipopt.scip', [os.path.join('pyscipopt', 'scip'+ext)],
                          include_dirs=[includedir],
                          library_dirs=[libdir],
                          libraries=[dllname])]

if cythonize:
    extensions = cythonize(extensions)

setup(
    name = 'pyscipopt',
    version = '1.0',
    description = 'wrapper for SCIP in Python',
    author = 'Zuse Institute Berlin',
    author_email = 'scip@zib.de',
    license = 'ZIB',
    ext_modules = extensions,
    packages=['pyscipopt']
)
