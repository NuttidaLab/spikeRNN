#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for the SpikeRNN package - Unified Rate and Spiking Neural Networks

This setup script allows both the rate and spiking packages to be installed together
as a unified Python package, enabling users to import both modules after a single install.

Based on Kim et al. (2019) framework for implementing continuous-variable rate RNNs 
that can be mapped to spiking neural networks.

Authors: NuttidaLab
License: MIT
"""

from setuptools import setup, find_packages
import os

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
def parse_requirements():
    """Parse requirements from requirements.txt file."""
    with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    # Package metadata
    name="spikeRNN",
    version="0.1.0",
    author="Sally Liu",
    author_email="bl3092@columbia.edu",
    description="Unified Rate and Spiking Recurrent Neural Networks for cognitive tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NuttidaLab/spikeRNN",
    
    # Package discovery
    packages=find_packages(),
    python_requires=">=3.7",
    
    # Dependencies
    install_requires=parse_requirements(),
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov',
            'black',
            'flake8',
            'mypy',
        ],
        'docs': [
            'sphinx>=3.0',
            'sphinx-book-theme',
            'sphinxcontrib-napoleon',
        ],
        'parallel': [
            'joblib>=1.0',
            'multiprocess>=0.70',
        ],
    },
    
    # Package data
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.md', '*.rst'],
    },
    
    # Entry points (if any)
    entry_points={
        'console_scripts': [
            # Add command-line scripts here if needed
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Keywords
    keywords=[
        "neural networks", 
        "RNN", 
        "rate models", 
        "spiking neural networks",
        "recurrent neural networks", 
        "cognitive tasks", 
        "neuroscience",
        "computational neuroscience",
        "leaky integrate-and-fire",
        "cognitive modeling",
        "pytorch",
        "machine learning",
        "neuromorphic computing"
    ],
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/NuttidaLab/spikeRNN/issues",
        "Source": "https://github.com/NuttidaLab/spikeRNN",
        "Documentation": "https://nuttidalab.github.io/spikeRNN/",
    },
    
    # Zip safety
    zip_safe=False,
) 