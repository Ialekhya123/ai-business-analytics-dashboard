"""
Setup script for AI Business Analytics Dashboard
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-business-analytics-dashboard",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive, real-time business intelligence platform with machine learning algorithms",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-business-analytics-dashboard",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Topic :: Office/Business :: Financial :: Accounting",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-dash>=2.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-dashboard=BV:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.csv", "*.json", "*.yaml", "*.yml"],
    },
    keywords=[
        "business intelligence",
        "analytics",
        "dashboard",
        "machine learning",
        "recommendation system",
        "data visualization",
        "real-time",
        "e-commerce",
        "python",
        "dash",
        "plotly",
        "scikit-learn",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ai-business-analytics-dashboard/issues",
        "Source": "https://github.com/yourusername/ai-business-analytics-dashboard",
        "Documentation": "https://github.com/yourusername/ai-business-analytics-dashboard/docs",
    },
) 