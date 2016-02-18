import sys, os, readline, glob, platform
from distutils.core import setup
from distutils.extension import Extension

cythonize = True

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    cythonize = False

BASEDIR = os.path.dirname(os.path.abspath(__file__))
INCLUDEDIR = os.path.join(BASEDIR,'..\..\src')

#identify compiler version
prefix = "MSC v."
i = sys.version.find(prefix)
if i == -1:
    raise Exception('cannot determine compiler version')
i = i + len(prefix)
s, rest = sys.version[i:].split(" ", 1)
majorVersion = int(s[:-2]) - 6
minorVersion = int(s[2:3]) / 10.0

if platform.architecture()[0].find('64')>=0:
    LIBDIR  = os.path.join(BASEDIR,'..','..','build','Release')
else:
    LIBDIR  = os.path.join(BASEDIR,'..','..','build','Release')
print('BASEDIR='+ BASEDIR)
print('INCLUDEDIR='+ INCLUDEDIR)
print('LIBDIR='+ LIBDIR)

def complete(text, state):
    return (glob.glob(text+'*')+[None])[state]

readline.set_completer_delims(' \t\n;')
readline.parse_and_bind("tab: complete")
readline.set_completer(complete)

extensions = []
ext = '.pyx' if cythonize else '.c'

extensions = [Extension('pyscipopt.scip', [os.path.join('pyscipopt', 'scip'+ext)],
                          extra_compile_args=['-UNDEBUG'],
                          include_dirs=[INCLUDEDIR],
                          library_dirs=[LIBDIR],
                          #runtime_library_dirs=[os.path.abspath('lib')],
                          libraries=['soplex', 'scipopt'])]
                          #libraries=['scipopt', 'readline', 'z', 'gmp', 'ncurses', 'm'])]

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
