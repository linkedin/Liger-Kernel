from setuptools import find_namespace_packages, setup

__version__ = "0.1.1"

setup(
    name="liger_kernel",
    version=__version__,
    description="Efficient Triton kernels for LLM Training",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="BSD-2-Clause",
    url="https://github.com/linkedin/Liger-Kernel",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="triton,kernels,LLM training,deep learning,Hugging Face,PyTorch,GPU optimization",
    include_package_data=True,
    install_requires=[
        "torch>=2.1.2",
        "triton>=2.3.0",
        "transformers>=4.42.0",
    ],
    extras_require={
        "dev": [
            "matplotlib>=3.7.2",
            "flake8>=4.0.1.1",
            "black>=24.4.2",
            "isort>=5.13.2",
            "pytest>=7.1.2",
            "datasets>=2.19.2",
        ]
    },
)
