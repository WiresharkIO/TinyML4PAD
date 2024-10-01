# conf.py

# -- Project information -----------------------------------------------------

project = 'Tiny-ML for chew detection using non-contact proximity sensors'
author = 'Vishal Sivakumar'
release = '1.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',  # Automatically generate docs from docstrings
    'sphinx.ext.napoleon', # Support for Google-style and NumPy-style docstrings
    'sphinx.ext.viewcode', # Add links to highlighted source code
]

templates_path = ['_templates']
exclude_patterns = [
                    '_build', 'Thumbs.db', '.DS_Store',
                    '*.c', '*.h', '*.pyc', '*.o',
                    '.git/', '.github/', '.gitignore',
                    'node_modules/', 'vendor/',
                    'tests/', 'examples/'
                    ]

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'  # Use the Read the Docs theme
html_theme_options = {
    'collapse_navigation': False,  # Don't collapse navigation
    'sticky_navigation': True,     # Stick the navigation to the top
}
html_static_path = ['_static']

# -- Options for autodoc -----------------------------------------------------

autodoc_mock_imports = ["numpy", "pandas"]  # If you need to mock imports during doc generation

language = 'en'  # Set the language of your documentation
extensions.append('sphinx.ext.intersphinx')
intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}
