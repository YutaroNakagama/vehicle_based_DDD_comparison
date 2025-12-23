# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project     = "Vehicle-based DDD Comparison"
html_title  = "Vehicle-based DDD Docs"
copyright   = '2025, ynakagama'
author      = 'ynakagama'
release     = '0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys

# Add project source to path for autodoc
sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('..'))

extensions = [
    "sphinx.ext.autodoc",         # docstringを拾う
    "sphinx.ext.napoleon",        # Google/Numpyスタイルdocstring対応
    "sphinx.ext.viewcode",        # ソースコードリンク
    "sphinx.ext.intersphinx",     # Cross-referencing external docs
    "sphinx_autodoc_typehints",   # 型ヒントを自動表示
    "myst_parser",                # Markdown support
    "sphinx_copybutton",          # Copy button for code blocks
    "sphinxcontrib.mermaid",      # Mermaid diagrams
]

# MyST-Parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]
myst_heading_anchors = 3

# conf.py
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# Source directory configuration
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Use source/ as the master document location
master_doc = 'source/index'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
}
#html_static_path = ['_static']

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

autodoc_mock_imports = [
    "tensorflow", "tensorflow.keras",
    "optuna", "numpy", "pandas", "scipy",
    "matplotlib", "seaborn", "joblib", "numba", "sklearn",
    "tslearn", "pyswarm", "xgboost", "lightgbm", "catboost",
    "tqdm", "imblearn",   
]

