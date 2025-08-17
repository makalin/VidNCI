#!/usr/bin/env python3
"""
Setup script for VidNCI - A Python Library for Defining AI-Generated Videos
"""

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import numpy as np
import sys
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Check if Cython is available
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext as cython_build_ext
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

# Define Cython extensions
extensions = []
if USE_CYTHON:
    extensions = [
        Extension(
            "vidnci.core._code_extractor",
            ["vidnci/core/_code_extractor.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-march=native", "-ffast-math"],
            extra_link_args=["-O3"],
        )
    ]

# Setup configuration
setup(
    name="vidnci",
    version="0.1.0",
    author="VidNCI Team",
    author_email="contact@vidnci.org",
    description="A Python Library for Defining AI-Generated Videos using Noise-Coded Illumination",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/makalin/VidNCI",
    project_urls={
        "Bug Reports": "https://github.com/makalin/VidNCI/issues",
        "Source": "https://github.com/makalin/VidNCI",
        "Documentation": "https://vidnci.readthedocs.io/",
    },
    packages=find_packages(include=["vidnci", "vidnci.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Legal Industry",
        "Intended Audience :: Media",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: Multimedia :: Video",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "pytest-mock>=3.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "ai": [
            "torch>=1.9.0",
            "tensorflow>=2.6.0",
            "scikit-learn>=1.0",
        ],
        "full": [
            "torch>=1.9.0",
            "tensorflow>=2.6.0",
            "scikit-learn>=1.0",
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "pytest-mock>=3.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    ext_modules=cythonize(extensions) if USE_CYTHON else [],
    cmdclass={"build_ext": cython_build_ext if USE_CYTHON else build_ext},
    include_package_data=True,
    package_data={
        "vidnci": ["py.typed", "*.pyi"],
    },
    zip_safe=False,
    keywords=[
        "video",
        "forensics",
        "ai-detection",
        "deepfake",
        "noise-coded-illumination",
        "nci",
        "computer-vision",
        "digital-forensics",
        "video-analysis",
        "authentication",
    ],
)
