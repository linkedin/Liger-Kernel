import os
import sys
from unittest.mock import MagicMock

class MockModule(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        if name == '__all__':
            return []
        return MagicMock()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__all__ = []

# Configure paths
docs_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(os.path.join(docs_dir, '..'))
sys.path.insert(0, root_dir)

# Source and Build directories
source_dir = os.path.join(docs_dir, 'source')
output_dir = os.path.join(docs_dir, '_build')

# List all modules to mock
MOCK_MODULES = [
    'liger_kernel',
    'liger_kernel.transformers',
    'liger_kernel.transformers.experimental',
    'liger_kernel.nn',
    'liger_kernel.transformers.AutoLigerKernelForCausalLM',
    'liger_kernel.transformers.LigerFusedLinearCrossEntropyLoss',
    'liger_kernel.transformers.KLDivergence',
    'liger_kernel.transformers.JSD',
    'liger_kernel.transformers.GeneralizedJSD',
    'liger_kernel.transformers.FusedLinearJSD',
    'liger_kernel.nn.Module',
]
sys.modules.update((mod_name, MockModule()) for mod_name in MOCK_MODULES)

# Project information
project = 'Liger-Kernel'
copyright = '2024'
author = 'LinkedIn'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.autosummary',
    'sphinx.ext.linkcode',
    'sphinx.ext.intersphinx'
]

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}

# Source and build configuration
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = []

# HTML output options
html_theme = 'alabaster'
html_static_path = [os.path.join(source_dir, '_static')]
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
}

# Output directory configuration
html_baseurl = ''
html_output_dir = output_dir

# autodoc configuration
autodoc_member_order = 'bysource'
autoclass_content = 'both'

# Enable todo
todo_include_todos = True

# Generate autosummary
autosummary_generate = True