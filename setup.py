import sys, os, readline, glob
from distutils.core import setup
from distutils.extension import Extension

cythonize = True

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    cythonize = False

# defines whether the python interface should link againt a static (.a)
# or a shared (.so) scipopt library
usesharedlib = True
pathToScipoptsuite = os.path.abspath('../../../')
if usesharedlib:
    libscipopt = 'lib/libscipopt.so'
else:
    libscipopt = 'lib/libscipopt.a'

# create lib directory if necessary
if not os.path.exists('lib'):
    os.makedirs('lib')

# try to find library automatically
if os.path.exists(os.path.join(pathToScipoptsuite,libscipopt)):
    # create symbolic links to SCIP
    if not os.path.lexists(libscipopt):
        os.symlink(os.path.join(pathToScipoptsuite,libscipopt), libscipopt)

includescip = os.path.abspath('../../src')

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

# remove links to lib and include
if 'cleanlib' in args:
    if os.path.lexists(libscipopt):
        print('removing '+libscipopt)
        os.remove(libscipopt)
    quit()

# completely remove compiled code
if 'clean' in args:
    compiledcode = 'pyscipopt/scip.c'
    if os.path.exists(compiledcode):
        print('removing '+compiledcode)
        os.remove(compiledcode)

# always use build_ext --inplace
if args.count("build_ext") > 0 and args.count("--inplace") == 0:
    sys.argv.insert(sys.argv.index("build_ext")+1, "--inplace")

# check for missing scipopt library
if not os.path.lexists(libscipopt):
    pathToLib = os.path.abspath(my_input('Please enter path to scipopt library (scipoptsuite/lib/libscipopt.so or .a):\n'))
    print(pathToLib)
    if not os.path.exists(pathToLib):
        print('Sorry, the path to scipopt library does not exist')
        quit()

# check for missing scip src directory
if not os.path.lexists(includescip):
    includescip = os.path.abspath(my_input('Please enter path to scip src directory (scipoptsuite/scip/src):\n'))
    print(includescip)
    if not os.path.exists(includescip):
        print('Sorry, the path to SCIP src/ directory does not exist')
        quit()

# create symbolic links to SCIP
if not os.path.lexists(libscipopt):
    os.symlink(pathToLib, libscipopt)

extensions = []
ext = '.pyx' if cythonize else '.c'

if usesharedlib:
   extensions = [Extension('pyscipopt.scip', [os.path.join('pyscipopt', 'scip'+ext)],
                         extra_compile_args=['-UNDEBUG'],
                         include_dirs=[includescip],
                         library_dirs=['lib'],
                         runtime_library_dirs=[os.path.abspath('lib')],
                         libraries=['scipopt', 'readline', 'z', 'gmp', 'ncurses', 'm'])]
else:
   extensions = [Extension('pyscipopt.scip', [os.path.join('pyscipopt', 'scip'+ext)],
                         extra_compile_args=['-UNDEBUG'],
                         include_dirs=[includescip],
                         extra_objects=[libscipopt],
                         libraries=['readline', 'z', 'gmp', 'ncurses', 'm'])]

if cythonize:
    extensions = cythonize(extensions)
    os.system("sed -i 's/[ ]*$//' pyscipopt/*.c")

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
