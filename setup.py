import sys, os, glob
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

lib = 'lib'
include = 'include'
error = False
args = sys.argv[1:]

def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except:
        os.remove(link_name)
        os.symlink(target, link_name)
    print("created link '"+link_name+"' pointing to '"+target+"'")

# always use build_ext --inplace
if args.count("build_ext") > 0 and args.count("--inplace") == 0:
    sys.argv.insert(sys.argv.index("build_ext")+1, "--inplace")

# check for SCIPOPTDIR variable and link it accordingly
scipoptdir = os.environ.get('SCIPOPTDIR')

if not scipoptdir is None:
    scipsrcdir = glob.glob(os.path.join(scipoptdir,'scip-*','src'))
    scipsrcdir = sorted(scipsrcdir)[-1]  # use the latest version
    print("setting up links to SCIP Optimization Suite installation based on environment variable 'SCIPOPTDIR:'\n"+scipoptdir)
    if not os.path.exists(os.path.join(scipoptdir,lib,'libscipopt.so')):
        print("ERROR: invalid path to library 'libscipopt.so'")
        error = True
    else:
        symlink_force(os.path.join(scipoptdir,lib), lib)
    if not os.path.exists(os.path.join(scipsrcdir,'scip','scip.h')):
        print("ERROR: invalid path to SCIP src directory")
        error = True
    else:
        symlink_force(os.path.join(scipsrcdir), include)

# verify exiting links
if not os.path.exists(os.path.join(lib, 'libscipopt.so')):
    print("ERROR: 'libscipopt.so' not found")
    print("Please create symbolic link (named 'lib') to 'lib/' directory of SCIP Opt Suite, containing 'libscipopt.so'")
    error = True
if not os.path.exists(os.path.join(include, 'scip', 'scip.h')):
    print("ERROR: SCIP headers not found")
    print("Please create symbolic link (named 'include') to 'src/' directory of SCIP in SCIP Opt Suite")
    error = True

if error:
    print("You may also set environment variable 'SCIPOPTDIR' to point to the installation directory of the SCIP Optimization Suite.")
    quit()

extensions = [Extension(
    'pyscipopt.scip',
    [os.path.join('pyscipopt', 'scip.pyx')],
    #extra_compile_args=['-UNDEBUG'],   # use this when linking to a dbg libscipopt.so
    include_dirs=[include],
    library_dirs=[lib],
    runtime_library_dirs=[os.path.abspath(lib)],
    libraries=['scipopt']
)]

setup(
    name = 'PySCIPOpt',
    version = '1.0',
    description = 'Python interface and modeling environment for SCIP',
    author = 'Zuse Institute Berlin',
    author_email = 'scip@zib.de',
    license = 'ZIB academic license',
    ext_modules = cythonize(extensions),
    packages=['pyscipopt']
)
