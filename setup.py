import sys, os, readline, glob
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

def complete(text, state):
    return (glob.glob(text+'*')+[None])[state]

readline.set_completer_delims(' \t\n;')
readline.parse_and_bind("tab: complete")
readline.set_completer(complete)

libscipopt = 'lib/libscipopt.so'
includescip = 'include/scip'

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
    if os.path.lexists(includescip):
        print('removing '+includescip)
        os.remove(includescip)
    quit()

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
    pathToLib = os.path.abspath(my_input('Please enter path to scipopt library (scipoptsuite/lib/libscipopt.so):\n'))
    print(pathToLib)

    # create lib directory if necessary
    if not os.path.exists('lib'):
        os.makedirs('lib')

    if not os.path.exists(pathToLib):
        print('Sorry, the path to scipopt library does not exist')
        quit()

# check for missing scip src directory
if not os.path.lexists(includescip):
    pathToScip = os.path.abspath(my_input('Please enter path to scip src directory (scipoptsuite/scip/src):\n'))
    print(pathToScip)

    # create lib directory if necessary
    if not os.path.exists('include'):
        os.makedirs('include')

    if not os.path.exists(pathToScip): 
        print('Sorry, the path to scip src directory does not exist')
        quit()

# create symbolic links to SCIP
if not os.path.lexists(libscipopt):
    os.symlink(pathToLib, libscipopt)
if not os.path.lexists(includescip):
    os.symlink(pathToScip, includescip)


ext_modules = []

ext_modules += [Extension('pyscipopt.scip', [os.path.join('pyscipopt', 'scip.pyx')],
                          #extra_compile_args=['-g', '-O0', '-UNDEBUG'],
                          include_dirs=[includescip],
                          library_dirs=['lib'],
                          runtime_library_dirs=[os.path.abspath('lib')],
                          libraries=['scipopt', 'readline', 'z', 'gmp', 'ncurses', 'm'])]


#ext_modules += cythonize(extensions)

setup(
    name = 'pyscipopt',
    version = '0.2',
    description = 'wrapper for SCIP in Python',
    author = 'Zuse Institute Berlin',
    author_email = 'scip@zib.de',
    license = 'MIT',
    cmdclass = {'build_ext' : build_ext},
    ext_modules = ext_modules,
    packages=['pyscipopt']
)
