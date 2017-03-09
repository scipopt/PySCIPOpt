from setuptools import setup, Extension
import os

# look for environment variable that specifies path to SCIP Opt lib and headers
scipoptdir = os.environ.get('SCIPOPTDIR', '')

includedir = os.path.join(scipoptdir, 'include')
libdir = os.path.join(scipoptdir, 'lib')

libname = 'libscipopt' if os.name == 'nt' else 'scipopt'

cythonize = True

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
                          libraries=[libname])]

if cythonize:
    extensions = cythonize(extensions)

setup(
    name = 'PySCIPOpt',
    version = '1.0.0',
    description = 'Python interface and modeling environment for SCIP',
    url = 'https://github.com/SCIP-Interfaces/PySCIPOpt',
    author = 'Zuse Institute Berlin',
    author_email = 'scip@zib.de',
    license = 'MIT',
    classifiers=[
    'Development Status :: 4 - Beta',
     'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3'
    ],
    ext_modules = extensions,
    packages = ['pyscipopt']
)
