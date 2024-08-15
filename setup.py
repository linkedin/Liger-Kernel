from setuptools import find_namespace_packages, setup

__version__ = "0.0.1"

setup(
    name="liger_kernel",
    version=__version__,
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "torch>=2.1.2",
        "triton>=2.3.0",
        "transformers>=4.40.1",
    ],
    extras_require={
        "dev": [
            "matplotlib>=3.7.2",
            "flake8>=4.0.1.1",
            "black>=24.4.2",
            "isort>=5.13.2",
            "pre-commit>=3.7.1",
            "torch-tb-profiler>=0.4.1",
        ]
    },
)
