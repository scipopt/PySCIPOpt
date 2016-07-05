import sys, os, readline, glob
from distutils.core import setup
from distutils.extension import Extension

cythonize = True

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    cythonize = False

pathToSCIPOpt = os.path.abspath('../../../')
pathToSCIPsrc = os.path.abspath('../../src')
lib = 'lib'
include = 'include'

# try to find library directory automatically
if os.path.exists(os.path.join(pathToSCIPOpt, lib, 'libscipopt.so')):
    # create symbolic link to SCIPOptSuite lib
    if not os.path.lexists(lib):
        os.symlink(os.path.join(pathToSCIPOpt, lib), lib)

# try to find header directory automatically
if os.path.exists(os.path.join(pathToSCIPsrc, 'scip', 'scip.h')):
    # create symbolic link to SCIP src
    if not os.path.lexists(include):
        os.symlink(pathToSCIPsrc, include)

def complete(text, state):
    return (glob.glob(text+'*')+[None])[state]

readline.set_completer_delims(' \t\n;')
readline.parse_and_bind("tab: complete")
readline.set_completer(complete)

args = sys.argv[1:]

# Python 2/3 compatibility
if sys.version_info >= (3, 0):
    my_input = input
else:
    my_input = raw_input

# always use build_ext --inplace
if args.count("build_ext") > 0 and args.count("--inplace") == 0:
    sys.argv.insert(sys.argv.index("build_ext")+1, "--inplace")

# check for missing library directory and link it
if not os.path.lexists(lib):
    pathToLib = os.path.abspath(my_input('Please enter path to scipopt library (scipoptsuite/lib):\n'))
    print(pathToLib)
    if not os.path.exists(pathToLib):
        print('Sorry, the path to scipopt library does not exist')
        quit()
    os.symlink(pathToLib, lib)

# check for missing scip src directory and link it
if not os.path.lexists(include):
    pathToSrc = os.path.abspath(my_input('Please enter path to scip src directory (scipoptsuite/scip/src):\n'))
    print(pathToSrc)
    if not os.path.exists(pathToSrc):
        print('Sorry, the path to SCIP src/ directory does not exist')
        quit()
    os.symlink(pathToSrc, include)

# verify links
if not os.path.exists(os.path.join(lib, 'libscipopt.so')):
    print("ERROR: invalid path to libscipopt.so")
    quit()
if not os.path.exists(os.path.join(include, 'scip', 'scip.h')):
    print("ERROR: invalid path to SCIP src directory")
    quit()

extensions = []
ext = '.pyx' if cythonize else '.c'

extensions = [Extension('pyscipopt.scip', [os.path.join('pyscipopt', 'scip'+ext)],
                         include_dirs=[include],
                         library_dirs=[lib],
                         runtime_library_dirs=[os.path.abspath(lib)],
                         libraries=['scipopt'])]

if cythonize:
    extensions = cythonize(extensions)
    os.system("sed -i 's/[ ]*$//' pyscipopt/*.c")

setup(
    name = 'pyscipopt',
    version = '1.0',
    description = 'Python interface and modeling environment for SCIP',
    author = 'Zuse Institute Berlin',
    author_email = 'scip@zib.de',
    license = 'ZIB',
    ext_modules = extensions,
    packages=['pyscipopt']
)
