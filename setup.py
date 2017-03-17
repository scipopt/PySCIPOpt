from setuptools import setup, Extension
import os, platform

# look for environment variable that specifies path to SCIP Opt lib and headers
scipoptdir = os.environ.get('SCIPOPTDIR', '')

includedir = os.path.abspath(os.path.join(scipoptdir, 'include'))
libdir = os.path.abspath(os.path.join(scipoptdir, 'lib'))

libname = 'libscipopt' if os.name == 'nt' else 'scipopt'

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
if platform.system() == 'Linux':
    runtime_library_dirs.append(libdir)
elif platform.system() == 'Darwin':
    extra_link_args.append('-Wl,-rpath,'+libdir)

extensions = [Extension('pyscipopt.scip', [os.path.join(packagedir, 'scip'+ext)],
                          include_dirs=[includedir],
                          library_dirs=[libdir],
                          libraries=[libname],
                          runtime_library_dirs=runtime_library_dirs,
                          extra_link_args=extra_link_args
                          )]

if cythonize:
    extensions = cythonize(extensions)

setup(
    name = 'PySCIPOpt',
    version = '1.1.0',
    description = 'Python interface and modeling environment for SCIP',
    url = 'https://github.com/SCIP-Interfaces/PySCIPOpt',
    author = 'Zuse Institute Berlin',
    author_email = 'scip@zib.de',
    license = 'MIT',
    classifiers=[
    'Development Status :: 4 - Beta',
     'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3'],
    ext_modules = extensions,
    packages = ['pyscipopt'],
    package_dir = {'pyscipopt': packagedir},
    package_data = {'pyscipopt': ['scip.pyx', 'scip.pxd', '*.pxi']}
)
