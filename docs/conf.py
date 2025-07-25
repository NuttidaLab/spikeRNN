# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the parent directory to the path so we can import the packages
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'spikeRNN'
copyright = '2025, NuttidaLab'
author = 'NuttidaLab'

# The full version, including alpha/beta/rc tags
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'myst_nb'
]

# MyST-NB settings
nb_execution_mode = "off"  # Temporarily disable execution while debugging
nb_execution_timeout = 300
nb_execution_allow_errors = True
nb_execution_raise_on_error = False

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.md': 'myst-nb',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_title = "spikeRNN"

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

# Autodoc settings
autodoc_mock_imports = ['matplotlib']  # Only mock visualization dependencies
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

html_theme_options = {
    'repository_url': 'https://github.com/NuttidaLab/spikeRNN',
    "use_repository_button": True,
    "use_download_button": False,
    'repository_branch': 'main',
    'launch_buttons': {
        'colab_url': 'https://colab.research.google.com',
        'binderhub_url': 'https://mybinder.org'
    },
}

# Clean up any existing build artifacts
import shutil
if os.path.exists('_build'):
    shutil.rmtree('_build')
    
