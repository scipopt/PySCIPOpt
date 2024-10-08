# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import pyscipopt

# sys.path.insert(0, os.path.abspath('.'))


sys.path.insert(0, os.path.abspath("../src/"))
# -- Project information -----------------------------------------------------

project = "PySCIPOpt"
copyright = "2024, Zuse Institute Berlin"
author = "Zuse Institute Berlin <scip@zib.de>"
html_logo = "_static/skippy_logo_blue.png"
html_favicon = '_static/skippy_logo_blue.png'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.jquery",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinxcontrib.jquery",
]

# You can define documentation here that will be repeated often
# e.g. XXX = """WARNING: This method will only work if your model has status=='optimal'"""
# rst_epilog = f""" .. |XXX| replace:: {XXX}"""
# Then in the documentation of a method you can put |XXX|

autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The intersphinx mapping dictionary is used to automatically reference
# python object types to their respective documentation.
# As PySCIPOpt has few dependencies this is not done.

intersphinx_mapping = {}

extlinks = {
    "pypi": ("https://pypi.org/project/%s/", "%s"),
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

autosummary_generate = False
napoleon_numpy_docstring = True
napoleon_google_docstring = False

pygments_style = "sphinx"

bibtex_bibfiles = ["ref.bib"]
