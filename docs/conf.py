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

sys.path.insert(0, os.path.abspath('../project/src'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

extensions = [
    "sphinx.ext.autodoc",         # docstringを拾う
    "sphinx.ext.napoleon",        # Google/Numpyスタイルdocstring対応
    "sphinx.ext.viewcode",        # ソースコードリンク
    "sphinx_autodoc_typehints",   # 型ヒントを自動表示
]

# conf.py
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
#html_static_path = ['_static']

autodoc_mock_imports = [
    "tensorflow",
    "tensorflow.keras",
    "optuna",
]

