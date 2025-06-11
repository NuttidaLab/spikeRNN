# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the parent directory to the path so we can import the packages
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../rate'))
sys.path.insert(0, os.path.abspath('../spiking'))

# -- Project information -----------------------------------------------------
project = 'SpikeRNN'
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

autosummary_generate = True
autosummary_imported_members = True

# Napoleon settings (for better docstring parsing)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_mock_imports = ['torch', 'numpy', 'scipy', 'matplotlib'] 
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
} 

templates_path = ['_templates']
exclude_patterns = []

html_title = "SpikeRNN"
html_theme = 'sphinx_book_theme'
html_theme_options = {
    'repository_url': 'https://github.com/NuttidaLab/SpikeRNN',
    "use_repository_button": True,
    "use_download_button": False,
    'repository_branch': 'main',
    "path_to_docs": 'docs/source',
    'launch_buttons': {
        'colab_url': 'https://colab.research.google.com',
        'binderhub_url': 'https://mybinder.org'
    },
}