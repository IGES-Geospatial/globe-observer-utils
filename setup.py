from datetime import date
from os.path import dirname
from os.path import exists
from os.path import join
from os.path import abspath
from setuptools import find_packages
from setuptools import setup
import subprocess

setup_dir = dirname(abspath(__file__))
git_dir = join(setup_dir, ".git")
version_file = join(setup_dir, "version.py")
package_name = "go_utils"

# Automatically generate a version.py based on the git version
if exists(git_dir):
    proc = subprocess.run(
        [
            "git",
            "rev-list",
            "--count",
            # Includes previous year's commits in case one was merged after the
            # year incremented. Otherwise, the version wouldn't increment.
            '--after="main@{' + str(date.today().year - 1) + '-01-01}"',
            "main",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # If there is no main branch, the commit count defaults to 0
    if proc.returncode:
        commit_count = "0"
    else:
        commit_count = proc.stdout.decode("utf-8")

    # Version number: <year>.<# commits on main>
    version = str(date.today().year) + "." + commit_count.strip()

    # Create the version.py file
    with open(version_file, "w+") as fp:
        fp.write('# Autogenerated by setup.py\n__version__ = "{0}"'.format(version))

if exists(version_file):
    with open(version_file, "r") as fp:
        exec(fp.read(), globals())
else:
    __version__ = "main"

with open(join(setup_dir, "README.md"), "r") as readme_file:
    long_description = readme_file.read()

setup(
    name=package_name,
    version=__version__,
    description="Utilities for interfacing with GLOBE Observer Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="IGES",
    maintainer="Matteo Kimura, Prachi Ingle, Pratham Babaria",
    maintainer_email="mateus.sakata@gmail.com, phna14@gmail.com, prathambabaria1@gmail.com",
    project_urls={
        "Source Code": "https://github.com/IGES-Geospatial/globe-observer-utils",
        "Documentation": "https://iges-geospatial.github.io/globe-observer-utils-docs/go_utils.html",
        "Bug Tracker": "https://github.com/IGES-Geospatial/globe-observer-utils/issues",
    },
    entry_points={
        "console_scripts": [
            "mhm-download=go_utils._cli:mhm_data_download",
            "lc-download=go_utils._cli:lc_data_download",
            "mhm-photo-download=go_utils._cli:mhm_photo_download",
            "lc-photo-download=go_utils._cli:lc_photo_download",
        ]
    },
    keywords="GlobeObserver GLOBE mosquito landcover",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=True,
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.5",
        "requests>=2.23.0",
        "arcgis>=1.8.4",
        "seaborn>=0.11.1",
    ],
    python_requires=">=3.6",
    license="MIT License",
    classifiers=[
        "Intended Audience :: Education",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: GIS",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
)
