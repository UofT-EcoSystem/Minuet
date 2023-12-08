# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Minuet'
copyright = '2023, Jiacheng Yang, Christina Giannoula, Jun Wu, Mostafa Elhoushi, James Gleeson, Gennady Pekhimenko'
author = 'Jiacheng Yang, Christina Giannoula, Jun Wu, Mostafa Elhoushi, James Gleeson, Gennady Pekhimenko'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.ifconfig',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    'show_prev_next': False,
    "show_nav_level": 0,
    "logo": {
        "text": "🎶 Minuet",
    },
    "footer_start": [],
    "footer_end": [],
    "navbar_end": [],
    "navbar_persistent": [],
}
html_sidebars = {
    "**": ["search-field.html", "sidebar-nav-bs.html"]
}
html_show_sourcelink = False

html_css_files = ['custom.css']
html_static_path = ['_static']

autodoc_preserve_defaults = True
autodoc_default_options = {
    'members': True,
    'no-undoc-members': True,
}
autodoc_inherit_docstrings = False
# autodoc_typehints = "description"

add_module_names = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}
