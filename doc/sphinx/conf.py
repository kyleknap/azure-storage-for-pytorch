# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Azure Storage for PyTorch'
copyright = '2025, Microsoft'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'Python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'azure-core': ('https://azuresdkdocs.z19.web.core.windows.net/python/azure-core/latest/', None),
    'azure-identity': ('https://azuresdkdocs.z19.web.core.windows.net/python/azure-identity/latest/', None),
}

autodoc_typehints = 'both'
