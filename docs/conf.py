# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os, sys

sys.path.insert(0, os.path.abspath('../src'))


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]
if os.getenv('SPELLCHECK'):
    extensions += 'sphinxcontrib.spelling',
    spelling_show_suggestions = True
    spelling_lang = 'en_US'

source_suffix = '.rst'
master_doc = 'index'
project = 'Crunchers'
year = '2016'
author = 'Gus Dunn'
copyright = '{0}, {1}'.format(year, author)
version = release = '0.0.1'

# on_rtd is whether we are on readthedocs.org, this line of code grabbed from docs.readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
# only import and set the theme if we're building docs locally
if not on_rtd:
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
    # html_theme_options = {
    #     'githuburl': 'https://github.com/xguse/crunchers/'}
# otherwise, readthedocs.org uses their theme by default, so no need to specify it

# import sphinx_rtd_theme
# html_theme = "sphinx_rtd_theme"
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# html_theme_options = {
#     'githuburl': 'https://github.com/xguse/crunchers/'
# }

pygments_style = 'trac'
templates_path = ['.']
html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = True
html_sidebars = {
   '**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html'],
}
html_short_title = '%s-%s' % (project, version)
