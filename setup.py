from setuptools import setup, Extension
import os, platform, sys, re

# look for environment variable that specifies path to SCIP
scipoptdir = os.environ.get('SCIPOPTDIR', '').strip('"')

extra_compile_args = []
extra_link_args = []

# determine include directory
if os.path.exists(os.path.join(scipoptdir, 'src')):
    # SCIP seems to be installed in place
    includedir = os.path.abspath(os.path.join(scipoptdir, 'src'))
else:
    # assume that SCIP is installed on the system
    includedir = os.path.abspath(os.path.join(scipoptdir, 'include'))

print('Using include path <%s>.' % includedir)

# determine library
if os.path.exists(os.path.join(scipoptdir, 'lib/shared/libscipsolver.so')):
    # SCIP seems to be created with make
    libdir = os.path.abspath(os.path.join(scipoptdir, 'lib/shared'))
    libname = 'scipsolver'
    extra_compile_args.append('-DNO_CONFIG_HEADER')
else:
    # assume that SCIP is installed on the system
    libdir = os.path.abspath(os.path.join(scipoptdir, 'lib'))
    libname = 'scip'
    if platform.system() in ['Windows']:
        libname = 'libscip'

print('Using SCIP library <%s> at <%s>.' % (libname,libdir))

# set runtime libraries
if platform.system() in ['Linux', 'Darwin']:
    extra_link_args.append('-Wl,-rpath,'+libdir)

# enable debug mode if requested
if "--debug" in sys.argv:
    extra_compile_args.append('-UNDEBUG')
    sys.argv.remove("--debug")

use_cython = True

packagedir = os.path.join('src', 'pyscipopt')

with open(os.path.join(packagedir, '__init__.py'), 'r') as initfile:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        initfile.read(), re.MULTILINE).group(1)

try:
    from Cython.Build import cythonize
except ImportError as err:
    # if cython is not found _and_ src/pyscipopt/scip.c does not exist then we cannot do anything.
    if not os.path.exists(os.path.join(packagedir, 'scip.c')):
        print('Cython is required')
        quit(1)
    use_cython = False

# if src/pyscipopt/scip.pyx does not exist then there is no need for using cython
if not os.path.exists(os.path.join(packagedir, 'scip.pyx')):
    use_cython = False

ext = '.pyx' if use_cython else '.c'

extensions = [Extension('pyscipopt.scip', [os.path.join(packagedir, 'scip'+ext)],
                          include_dirs=[includedir],
                          library_dirs=[libdir],
                          libraries=[libname],
                          extra_compile_args = extra_compile_args,
                          extra_link_args=extra_link_args
                          )]

if use_cython:
    extensions = cythonize(extensions, compiler_directives={'language_level': 3})

with open('README.md') as f:
    long_description = f.read()

setup(
    name='PySCIPOpt',
    version=version,
    description='Python interface and modeling environment for SCIP',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/SCIP-Interfaces/PySCIPOpt',
    author='Zuse Institute Berlin',
    author_email='scip@zib.de',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Cython',
        'Topic :: Scientific/Engineering :: Mathematics'],
    ext_modules=extensions,
    install_requires=['wheel'],
    packages=['pyscipopt'],
    package_dir={'pyscipopt': packagedir},
    package_data={'pyscipopt': ['scip.pyx', 'scip.pxd', '*.pxi']}
)
