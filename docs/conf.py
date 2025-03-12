# Configuration file for the Sphinx documentation builder.
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('..'))



# -- Project information -----------------------------------------------------

project = 'tedi'
copyright = '2025, João Camacho'
author = 'João Camacho'

# The full version, including alpha/beta/rc tags
release = '3.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinxcontrib.autodoc_pydantic'
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = "_images/logo_tedi.png"
html_theme_options = {
    'logo_only': True,
    'display_version': True,
}

# -- Extension configuration -------------------------------------------------

# Enable parsing of both Google and NumPy-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
autosummary_generate = True
