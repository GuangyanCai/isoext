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
    "sphinxcontrib.bibtex",
]

# MyST-NB settings
nb_execution_mode = "off"  # Don't execute notebooks during build (requires GPU)

# Bibliography. Pages cite with {cite:t}`key` and end with a local
# {bibliography} directive filtered to their own citations; the references
# page lists everything.
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "unsrt"
# The per-page lists restart their numeric labels, which is intended.
suppress_warnings = ["bibtex.duplicate_label"]

# ACM and Taylor & Francis serve 403 to non-browser agents; the DOIs
# behind these were verified against Crossref when references.bib was
# written.
linkcheck_ignore = [
    r"https://doi\.org/10\.1145/.*",
    r"https://doi\.org/10\.1080/.*",
]

# Copy viser's static client into _static so the interactive scene embeds in
# the executed notebooks keep working in the built documentation.
from pathlib import Path

from isoext.viewer import copy_client

copy_client(Path(__file__).parent / "_static" / "viser")

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
html_css_files = ['custom.css']

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#d95f43",
        "color-brand-content": "#d95f43",
    },
    "dark_css_variables": {
        "color-brand-primary": "#ff8a65",
        "color-brand-content": "#ff8a65",
    },
    "source_repository": "https://github.com/GuangyanCai/isoext",
    "source_branch": "master",
    "source_directory": "doc/",
}
