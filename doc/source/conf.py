from datetime import datetime
from sphinx.application import Sphinx
from sphinx.util.docfields import Field
import os


def setup(app: Sphinx):
    app.add_object_type(
        'confval',
        'confval',
        objname='configuration value',
        indextemplate='pair: %s; configuration value',
        doc_field_types=[
            Field('type', label='Type', has_arg=False, names=('type',)),
            Field('default', label='Default', has_arg=False, names=('default',)),
            Field('units', label='Units', has_arg=False, names=('units',)),
        ]
    )
    app.add_config_value('package_version', release, 'env')


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
# sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'Ladim'
# noinspection PyShadowingBuiltins
copyright = f'{datetime.now().year}, Institute of Marine Research'
author = 'Bjørn Ådlandsvik'
source_suffix = '.rst'


# The full version, including alpha/beta/rc tags
def getversion():
    version_file = os.path.abspath('../../ladim/__init__.py')
    version_line = ''
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                version_line = line
                break
    
    if not version_line:
        raise RuntimeError("Could not find __version__ in ladim/__init__.py")
    
    # Extract version string between quotes
    import re
    match = re.search(r"['\"]([^'\"]+)['\"]", line)
    if match:
        return match.group(1)
    else:
        raise RuntimeError("Could not find __version__ in ladim/__init__.py")


release = getversion()
# Add to the substitutions
rst_prolog = f"""
.. |package_version| replace:: {release}
"""

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'autoapi.extension',
    'matplotlib.sphinxext.plot_directive',
]

nitpicky = True
html_css_files = [
    'css/custom.css',
]

# Matplotlib extension options
plot_html_show_source_link = False
plot_formats = ['png']
plot_html_show_formats = False
plot_pre_code = ""

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

locale_dirs = []
language = 'en'
gettext_compact = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for Sphinx AutoAPI -----------------------------------------------

autoapi_dirs = ['../../ladim']
autoapi_ignore = [
    '*/build/*',
    '*/gridforce/*',
]
autoapi_add_toctree_entry = True
autoapi_member_order = 'groupwise'
autoapi_template_dir = '_templates/autoapi'
autoapi_keep_files = False
autoapi_generate_api_docs = True
autoapi_own_page_level = 'module'
autoapi_options = [
    'members',
    'show-module-summary',
    'imported-members',
]
autodoc_typehints = 'description'


# -- Options for intersphinx ------------------------

intersphinx_mapping = {
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}
