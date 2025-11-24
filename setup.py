#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for Neural Network Efficiency Analyzer
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="nn-efficiency-analyzer",
    version="0.1.0",
    author="Neural Network Efficiency Analyzer Team",
    author_email="",
    description="A comprehensive toolkit for analyzing and optimizing neural network efficiency",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # Note: Repository name contains hyphen prefix (original repository naming)
    url="https://github.com/fryfry33/-Neural-Network-Efficiency-Analyzer",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "tensorflow": ["tensorflow>=2.4.0"],
        "pytorch": ["torch>=1.7.0"],
        "all": ["tensorflow>=2.4.0", "torch>=1.7.0"],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "flake8>=3.8.0",
            "black>=20.8b1",
        ],
    },
    keywords="neural-network deep-learning pruning optimization efficiency tensorflow pytorch",
    project_urls={
        "Bug Reports": "https://github.com/fryfry33/-Neural-Network-Efficiency-Analyzer/issues",
        "Source": "https://github.com/fryfry33/-Neural-Network-Efficiency-Analyzer",
        "Documentation": "https://github.com/fryfry33/-Neural-Network-Efficiency-Analyzer/tree/main/docs",
    },
)
