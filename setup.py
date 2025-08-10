"""
Setup script for PyRefine framework.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read DOCUMENTATION for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "DOCUMENTATION.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="pyrefine",
    version="1.0.0",
    author="Siva Sandeep Reddy",
    author_email="sivasandeep@example.com",
    description="PyRefine: Iterative refinement engine with Change-of-Thought capture and LangGraph orchestration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/pyrefine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pyrefine=pyrefine.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pyrefine": ["config/*.yaml"],
    },
    keywords=[
        "pyrefine",
        "self-refine",
        "change-of-thought",
        "llm",
        "refinement",
        "langgraph",
        "openai",
        "gemini",
        "iterative",
        "ai",
        "nlp"
    ],
)
