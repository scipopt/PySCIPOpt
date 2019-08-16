from setuptools import setup, Extension
import os, platform, sys, re

# look for environment variable that specifies path to SCIP
scipoptdir = os.environ.get('SCIPOPTDIR', '').strip('"')

# now check whether SCIP is installed on the system or whether a source code directory has been specified
if os.path.exists(os.path.join(scipoptdir, 'include')):
    includedir = os.path.abspath(os.path.join(scipoptdir, 'include'))
else:
    if os.path.exists(os.path.join(scipoptdir, 'src')):
        includedir = os.path.abspath(os.path.join(scipoptdir, 'src'))
    else:
        print('Neither directory \'src\' nor \'include\' exists in the path <%s>' % scipoptdir)
        quit(1)

# determine library
libdir = os.path.abspath(os.path.join(scipoptdir, 'lib'))
libname = 'scip'
if (not os.path.exists(os.path.join(libdir, 'libscip.so'))) and (not os.path.exists('/usr/lib/libscip.so')):
    if os.path.exists(os.path.join(libdir, 'shared/libscipsolver.so')):
        print('SCIP library <%s> does not exist.\nUsing <%s> instead.' % (os.path.join(libdir, 'libscip.so'), os.path.join(libdir, 'shared/libscipsolver.so')))
        libdir = os.path.abspath(os.path.join(scipoptdir, 'lib/shared'))
        libname = 'scipsolver'
    else:
        print('SCIP library does not exist - tried <lib/libscip.so> and <lib/shared/libscipsolver.so>.')
        quit(1)


cythonize = True

packagedir = os.path.join('src', 'pyscipopt')

with open(os.path.join(packagedir, '__init__.py'), 'r') as initfile:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        initfile.read(), re.MULTILINE).group(1)

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
if platform.system() in ['Linux', 'Darwin']:
    extra_link_args.append('-Wl,-rpath,'+libdir)

# enable debug mode if requested
extra_compile_args = []
if "--debug" in sys.argv:
    extra_compile_args.append('-UNDEBUG')
    sys.argv.remove("--debug")

# avoid errors if SCIP is build with make:
extra_compile_args.append('-DNO_CONFIG_HEADER')

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
#     extensions = cythonize(extensions, compiler_directives={'linetrace': True})

with open('README.md') as f:
    long_description = f.read()

setup(
    name = 'PySCIPOpt',
    version = version,
    description = 'Python interface and modeling environment for SCIP',
    long_description = long_description,
    long_description_content_type='text/markdown',
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
