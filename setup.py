"""Setup script for poison_detection package.

Note: The repository URL is currently set to an anonymous placeholder.
This will be updated to the actual repository URL upon publication.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="poison_detection",
    version="1.0.0",
    author="Anonymous",
    description="A toolkit for detecting poisoned data in instruction-tuned language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anonymous/Poison-Detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "safetensors>=0.3.0",
        "pandas>=1.3.0",
        "pyyaml>=6.0",
        "kronfluence>=0.1.0",
        "sentencepiece>=0.1.96",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "poison-detect=examples.detect_poisons:main",
        ],
    },
)