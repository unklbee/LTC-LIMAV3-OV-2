# setup.py
"""
Setup script for LIMA Traffic Counter
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="lima-traffic-counter",
    version="2.0.0",
    author="Lintas Mediatama",
    author_email="dev@lintasmediatama.com",
    description="Advanced traffic counting system with AI-powered vehicle detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lintasmediatama/lima-traffic-counter",
    project_urls={
        "Bug Tracker": "https://github.com/lintasmediatama/lima-traffic-counter/issues",
        "Documentation": "https://lima-traffic-counter.readthedocs.io",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-qt>=4.2.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
            "pre-commit>=3.3.0",
        ],
        "tensorrt": [
            "tensorrt>=8.6.0",
            "pycuda>=2022.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "lima-counter=src.main:run",
            "lima-counter-cli=src.main:run_cli",
            "lima-counter-server=src.main:run_server",
            "lima-counter-benchmark=src.main:run_benchmark",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml"],
    },
)
