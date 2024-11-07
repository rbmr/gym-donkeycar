#!/usr/bin/env python
import os

from setuptools import find_packages, setup

with open(os.path.join("gym_donkeycar", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()

description = "Modified OpenAI Gym Environments for Donkey Car"


with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

# gym 0.23 introduces breaking changes
requirements = [
    "gym==0.22.0",                    
    "numpy",            
    "pillow",       
    "cloudpickle",     
    "scipy",  
    "matplotlib",  
    "opencv-python-headless",
    "tensorflow==1.2.0", 
    "torch",
    "stable_baselines3",
    "onnx",
    'shimmy'
]

setup(
    name="gym_donkeycar",
    author="Some Handsome People",
    author_email="somehandsomepeople@student.uts.edu.au",
    python_requires=">=3.11,<3.12",
    classifiers=[],
    description=description,
    install_requires=requirements,
    extras_require={
        "tests": [
            # Run tests and coverage
            "pytest",
            "pytest-mock",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            # Type check
            "pytype",
            # Lint code
            "ruff",
            # Sort imports
            "isort>=5.0",
            # Reformat
            "black",
        ],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            # For spelling
            "sphinxcontrib.spelling",
            # Type hints support
            "sphinx-autodoc-typehints",
        ],
    },
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="donkeycar, environment, agent, rl, openaigym, openai-gym, gym",
    packages=find_packages(),
    url="https://github.com/tawnkramer/gym-donkeycar",
    version=__version__,
    zip_safe=False,
)
