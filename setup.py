#!/usr/bin/env python3
"""Setup script for EpixVanity."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="epix-vanity",
    version="1.0.0",
    author="Mud",
    description="High-performance vanity address generator for Epix blockchain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EpixZone/EpixVanity",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Security :: Cryptography",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": ["pycuda>=2023.1", "pyopencl>=2023.1.4"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "black>=23.12.0", "flake8>=6.1.0", "mypy>=1.8.0"],
    },
    entry_points={
        "console_scripts": [
            "epix-vanity=epix_vanity.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
