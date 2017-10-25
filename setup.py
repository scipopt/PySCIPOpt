from setuptools import setup, Extension
import os, platform, sys

# look for environment variable that specifies path to SCIP Opt lib and headers
scipoptdir = os.environ.get('SCIPOPTDIR', '/usr/local')  # assume SCIP shared library is in /usr/local if SCIOPTDIR not set
includedir = os.path.abspath(os.path.join(scipoptdir, 'include'))
libdir = os.path.abspath(os.path.join(scipoptdir, 'lib'))
libname = 'scip'

cythonize = True

packagedir = os.path.join('src', 'pyscipopt')

try:
    from Cython.Build import cythonize
except ImportError:
    if not os.path.exists(os.path.join(packagedir, 'scip.c')):
        print('Cython is required')
        quit(1)
    cythonize = False

if not os.path.exists(os.path.join(packagedir, 'scip.pyx')):
    cythonize = False

ext = '.pyx' if cythonize else '.c'

# set runtime libraries
runtime_library_dirs = []
extra_link_args = []

if platform.system() in ['Darwin', 'Linux']:  # now both Linux and Darwin work like this:
    extra_link_args.append('-Wl,-rpath,'+libdir)

# enable debug mode if requested
extra_compile_args = []
if "--debug" in sys.argv:
    extra_compile_args.append('-UNDEBUG')
    sys.argv.remove("--debug")

extensions = [Extension('pyscipopt.scip', [os.path.join(packagedir, 'scip'+ext)],
                          include_dirs=[includedir],
                          library_dirs=[libdir],
                          libraries=[libname],
                          runtime_library_dirs=runtime_library_dirs,
                          extra_compile_args = extra_compile_args,
                          extra_link_args=extra_link_args
                          )]

if cythonize:
    extensions = cythonize(extensions)

setup(
    name = 'PySCIPOpt',
    version = '1.2.0',
    description = 'Python interface and modeling environment for SCIP',
    url = 'https://github.com/SCIP-Interfaces/PySCIPOpt',
    author = 'Zuse Institute Berlin',
    author_email = 'scip@zib.de',
    license = 'MIT',
    classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Education',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Programming Language :: Cython',
    'Topic :: Scientific/Engineering :: Mathematics'],
    ext_modules = extensions,
    packages = ['pyscipopt'],
    package_dir = {'pyscipopt': packagedir},
    package_data = {'pyscipopt': ['scip.pyx', 'scip.pxd', '*.pxi']}
)
