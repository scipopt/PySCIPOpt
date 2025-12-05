from setuptools import find_packages, setup, Extension
import os, platform, sys

# look for environment variable that specifies path to SCIP
scipoptdir = os.environ.get("SCIPOPTDIR", "").strip('"')

extra_compile_args = []
extra_link_args = []

# if SCIPOPTDIR is not set, try to detect conda environment, otherwise assume global installation
if not scipoptdir:
    # check if we're in a conda environment
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip('"')

    if conda_prefix and os.path.exists(os.path.join(conda_prefix, "include")):
        if platform.system() == "Windows":
            includedirs = [os.path.join(conda_prefix, "Library/include")]
            libdir = os.path.join(conda_prefix, "Library/lib")
            libname = "libscip"
        else:
            includedirs = [os.path.join(conda_prefix, "include")]
            libdir = os.path.join(conda_prefix, "lib")
            libname = "scip"
        print(f"Detected conda environment at {conda_prefix}.")
        print(f"Using include path {includedirs}.")
        print(f"Using library directory {libdir}.\n")
    else:
        # fall back to global installation
        if platform.system() == "Darwin":
            includedirs = ["/usr/local/include"]
            libdir = "/usr/local/lib"
        else:
            includedirs = ["."]
            libdir = "."
        libname = "libscip" if platform.system() == "Windows" else "scip"
        print("Assuming that SCIP is installed globally, because SCIPOPTDIR is undefined.\n")

else:

    # check whether SCIP is installed in the given directory
    if os.path.exists(os.path.join(scipoptdir, "include")):
        includedirs = [os.path.abspath(os.path.join(scipoptdir, "include"))]
    else:
        print("SCIPOPTDIR=%s does not contain an include directory; searching for include files in src or ../src directory.\n" % scipoptdir)

        if os.path.exists(os.path.join(scipoptdir, "src")):
            # SCIP seems to be installed in-place; check whether it was built using make or cmake
            if os.path.exists(os.path.join(scipoptdir, "src", "scip")):
                # assume that SCIPOPTDIR pointed to the main source directory (make)
                includedirs = [os.path.abspath(os.path.join(scipoptdir, "src")), os.path.abspath(os.path.join(scipoptdir, "lib/shared/include"))]
            else:
                # assume that SCIPOPTDIR pointed to a cmake build directory; try one level up (this is just a heuristic)
                if os.path.exists(os.path.join(scipoptdir, "..", "src", "scip")):
                    includedirs = [os.path.abspath(os.path.join(scipoptdir, "..", "src"))]
                else:
                    sys.exit("Could neither find src/scip nor ../src/scip directory in SCIPOPTDIR=%s. Consider installing SCIP in a separate directory." % scipoptdir)
        else:                    
            sys.exit("Could not find a src directory in SCIPOPTDIR=%s; maybe it points to a wrong directory." % scipoptdir)

    # determine library
    if os.path.exists(os.path.join(scipoptdir, "lib", "shared", "libscip.so")):
        # SCIP seems to be created with make
        libdir = os.path.abspath(os.path.join(scipoptdir, "lib", "shared"))
        libname = "scip"
    else:
        # assume that SCIP is installed on the system
        # check for lib64 first (newer SCIP tarballs), then lib
        if os.path.exists(os.path.join(scipoptdir, "lib64")):
            libdir = os.path.abspath(os.path.join(scipoptdir, "lib64"))
        elif os.path.exists(os.path.join(scipoptdir, "lib")):
            libdir = os.path.abspath(os.path.join(scipoptdir, "lib"))
        else:
            sys.exit("Could not find lib or lib64 directory in SCIPOPTDIR=%s" % scipoptdir)
        libname = "libscip" if platform.system() == "Windows" else "scip"

    print("Using include path %s." % includedirs)
    print("Using SCIP library %s at %s.\n" % (libname, libdir))

# set runtime libraries
if platform.system() in ["Linux", "Darwin"]:
    extra_compile_args.append("-I/opt/homebrew/include")
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
        [os.path.join(packagedir, "scip%s" % ext)],
        include_dirs=includedirs,
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
    version="6.0.0",
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
