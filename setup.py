from setuptools import find_packages, setup, Extension
import os, platform, sys

# look for environment variable that specifies path to SCIP
scipoptdir = os.environ.get("SCIPOPTDIR", "").strip('"')

extra_compile_args = []
extra_link_args = []

# if SCIPOPTDIR is not set, we assume that SCIP is installed globally
if not scipoptdir:
    if platform.system() == "Darwin":
        includedir = "/usr/local/include/"
        libdir = "/usr/local/lib/"
    else:
        includedir = "."
        libdir = "."
    libname = "libscip" if platform.system() in ["Windows"] else "scip"
    print("Assuming that SCIP is installed globally, because SCIPOPTDIR is undefined.\n")

else:

    # check whether SCIP is installed in the given directory
    if os.path.exists(os.path.join(scipoptdir, "include")):
        includedir = os.path.abspath(os.path.join(scipoptdir, "include"))
    else:
        print(f"SCIPOPTDIR={scipoptdir} does not contain an include directory; searching for include files in src or ../src directory.")

        if os.path.exists(os.path.join(scipoptdir, "src")):
            # SCIP seems to be installed in-place; check whether it was built using make or cmake
            if os.path.exists(os.path.join(scipoptdir, "src", "scip")):
                # assume that SCIPOPTDIR pointed to the main source directory (make)
                includedir = os.path.abspath(os.path.join(scipoptdir, "src"))
            else:
                # assume that SCIPOPTDIR pointed to a cmake build directory; try one level up (this is just a heuristic)
                if os.path.exists(os.path.join(scipoptdir, "..", "src", "scip")):
                    includedir = os.path.abspath(os.path.join(scipoptdir, "..", "src"))
                else:
                    sys.exit(f"Could neither find src/scip nor ../src/scip directory in SCIPOPTDIR={scipoptdir}. Consider installing SCIP in a separate directory.")
        else:                    
            sys.exit(f"Could not find a src directory in SCIPOPTDIR={scipoptdir}; maybe it points to a wrong directory.")

    # determine library
    if os.path.exists(os.path.join(scipoptdir, "lib", "shared", "libscip.so")):
        # SCIP seems to be created with make
        libdir = os.path.abspath(os.path.join(scipoptdir, "lib", "shared"))
        libname = "scip"
        extra_compile_args.append("-DNO_CONFIG_HEADER")
        # the following is a temporary hack to make it compile with SCIP/make:
        extra_compile_args.append("-DTPI_NONE")  # if other TPIs are used, please modify
    else:
        # assume that SCIP is installed on the system
        libdir = os.path.abspath(os.path.join(scipoptdir, "lib"))
        libname = "libscip" if platform.system() in ["Windows"] else "scip"

    print(f"Using include path {includedir}.")
    print(f"Using SCIP library {libname} at {libdir}.\n")

# set runtime libraries
if platform.system() in ["Linux", "Darwin"]:
    extra_link_args.append(f"-Wl,-rpath,{libdir}")

# enable debug mode if requested
if "--debug" in sys.argv:
    extra_compile_args.append("-UNDEBUG")
    sys.argv.remove("--debug")

use_cython = True

packagedir = os.path.join("src", "pyscipopt")

try:
    from Cython.Build import cythonize
except ImportError as err:
    # if cython is not found _and_ src/pyscipopt/scip.c does not exist then we cannot do anything.
    if not os.path.exists(os.path.join(packagedir, "scip.c")):
        sys.exit("Cython is required.")
    use_cython = False

# if src/pyscipopt/scip.pyx does not exist then there is no need for using cython
if not os.path.exists(os.path.join(packagedir, "scip.pyx")):
    use_cython = False

ext = ".pyx" if use_cython else ".c"


on_github_actions = os.getenv('GITHUB_ACTIONS') == 'true'
release_mode = os.getenv('RELEASE') == 'true'
compile_with_line_tracing = on_github_actions and not release_mode    

extensions = [
    Extension(
        "pyscipopt.scip",
        [os.path.join(packagedir, f"scip{ext}")],
        include_dirs=[includedir],
        library_dirs=[libdir],
        libraries=[libname],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros= [("CYTHON_TRACE_NOGIL", 1), ("CYTHON_TRACE", 1)] if compile_with_line_tracing else []
    )
]

if use_cython:
    extensions = cythonize(extensions, compiler_directives={"language_level": 3, "linetrace": compile_with_line_tracing})

with open("README.md") as f:
    long_description = f.read()

setup(
    name="PySCIPOpt",
    version="5.4.1",
    description="Python interface and modeling environment for SCIP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SCIP-Interfaces/PySCIPOpt",
    author="Zuse Institute Berlin",
    author_email="scip@zib.de",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    ext_modules=extensions,
    packages=find_packages(where="src"),
    package_dir={"pyscipopt": packagedir},
    package_data={"pyscipopt": ["scip.pyx", "scip.pxd", "*.pxi", "scip/lib/*"]},
    include_package_data=True,
)
