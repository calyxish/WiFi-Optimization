from setuptools import setup, find_packages

setup(
    name="wifi-optimization",
    version="1.0.0",
    description="WiFi network optimization using Graph Theory and Linear Algebra",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    authors=[
        "Kwakye Ishmael",
        "Kwame Adjei Amoah",
        "Adwoa Pokua",
        "Kofi Sintim",
        "Melchi",
    ],
    url="https://github.com/calyxish/WiFi-Optimization.git",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "matplotlib>=3.4",
    ],
    extras_require={
        "dev": ["pytest", "jupyter", "notebook"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Education",
    ],
)
