import sys, os, readline, glob, platform
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

BASEDIR = os.path.dirname(os.path.abspath(__file__))
BASEDIR = os.path.dirname(BASEDIR)
BASEDIR = os.path.dirname(BASEDIR)
INCLUDEDIR = os.path.join(BASEDIR,'src')
BASEDIR = os.path.dirname(BASEDIR)

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
    LIBDIR  = os.path.join(BASEDIR,'vc'+str(majorVersion),'scip_spx','x64','Release')
else:
    LIBDIR  = os.path.join(BASEDIR,'vc'+str(majorVersion),'scip_spx','Release')
print('BASEDIR='+ BASEDIR)
print('INCLUDEDIR='+ INCLUDEDIR)
print('LIBDIR='+ LIBDIR)

def complete(text, state):
    return (glob.glob(text+'*')+[None])[state]

readline.set_completer_delims(' \t\n;')
readline.parse_and_bind("tab: complete")
readline.set_completer(complete)

libscipopt = 'lib/libscipopt.so'
includescip = 'include/scip'


ext_modules = []

#extensions =  [Extension('pyscipopt.scip', [os.path.join('pyscipopt', 'scip.pyx')],
                          ## extra_compile_args=['-g', '-O0', '-UNDEBUG'],
                          #include_dirs=[includescip],
                          #library_dirs=['lib'],
                          #runtime_library_dirs=[os.path.abspath('lib')],
                          #libraries=['scipopt', 'readline', 'z', 'gmp', 'ncurses', 'm'])]

ext_modules += [Extension('pyscipopt.scip', [os.path.join('pyscipopt', 'scip.pyx')],
                          #extra_compile_args=['-g', '-O0', '-UNDEBUG'],
                          include_dirs=[INCLUDEDIR],
                          library_dirs=[LIBDIR],
                          #runtime_library_dirs=[os.path.abspath('lib')],
                          libraries=['spx', 'scip_spx'])]
                          #libraries=['scipopt', 'readline', 'z', 'gmp', 'ncurses', 'm'])]


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
