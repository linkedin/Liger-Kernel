import os
import sys
import importlib
import inspect

# Increase the recursion limit (optional)
sys.setrecursionlimit(10000)

# Configure paths
docs_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(os.path.join(docs_dir, '..'))
sys.path.insert(0, root_dir)

# Source and build directories
source_dir = os.path.join(docs_dir, 'source')
output_dir = os.path.join(docs_dir, '_build')

# List modules to mock (as simple strings)
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

# Mock modules by replacing them in sys.modules
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = None  # Simple mock; replace with 'None' or a basic object as needed

# Project information
project = 'Liger-Kernel'
copyright = '2024'
author = 'LinkedIn'

# General Sphinx configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.autosummary',
    'sphinx.ext.linkcode',
    'sphinx.ext.intersphinx',
]

# Mocked imports for autodoc
autodoc_mock_imports = ['numpy', 'tensorflow', 'torch', 'triton']

# Intersphinx mapping for cross-referencing external documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'triton': ('https://triton-lang.org/main/', None),
}

# Source file configuration
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = []

# HTML output configuration
html_theme = 'alabaster'
html_static_path = [os.path.join(source_dir, '_static')]
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
}

# Output directory for HTML files
html_output_dir = output_dir

# Autodoc configuration
autodoc_member_order = 'bysource'
autoclass_content = 'both'

# Enable todo extension
todo_include_todos = True

# Autosummary settings
autosummary_generate = True

def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != 'py':
        return None

    try:
        mod = importlib.import_module(info['module'])
        if 'class' in info:
            obj = getattr(mod, info['class'])
            if 'fullname' in info:
                obj = getattr(obj, info['fullname'])
        else:
            obj = getattr(mod, info['fullname'])
    except Exception:
        return None

    try:
        filepath = inspect.getsourcefile(obj)
        if filepath:
            filepath = os.path.relpath(filepath, start=root_dir)
        else:
            return None
    except Exception:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        return None

    # Modify these values based on your repository
    github_url = "https://github.com/LinkedIn/liger-kernel"
    branch = "main"  # or whatever your default branch is
    
    return f"{github_url}/blob/{branch}/{filepath}#L{lineno}-L{lineno + len(source) - 1}"