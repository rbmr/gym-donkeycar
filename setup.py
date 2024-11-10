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

# works for py 3.6 with cuda install
# requirements = [
#     "stable_baselines3==1.3.0",
#     "torch==1.10.2",
#     "pillow==8.3.1", 
# ]
# # In setup.py
# requirements = [
#     "gym==0.17.3",
#     "opencv-python==4.5.5.64",
#     "tensorflow-gpu==1.15.0",
#     "stable_baselines==2.10.2",
#     "torch==1.8.0",
#     "pillow==8.3.1", 
# ]

setup(
    name="gym_donkeycar",
    author="Some Handsome People",
    author_email="somehandsomepeople@student.uts.edu.au",
    python_requires=">=3.6, <3.7", # ">=3.11,<3.12",
    classifiers=[],
    description=description,
    # install_requires=requirements,
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
