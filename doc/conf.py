# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'isoext'
copyright = '2025, Guangyan Cai'
author = 'Guangyan Cai'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For Google/NumPy style docstrings
]

# MyST-NB settings
nb_execution_mode = "off"  # Don't execute notebooks during build (requires GPU)

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_docstring_signature = True  # Extract signatures from nanobind docstrings

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
