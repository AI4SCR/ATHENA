"""Install package."""
import io
import re

from pkg_resources import parse_requirements
from setuptools import find_packages, setup


def read_version(filepath: str) -> str:
    """Read the __version__ variable from the file.

    Args:
        filepath: probably the path to the root __init__.py

    Returns:
        the version
    """
    match = re.search(
        r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        io.open(filepath, encoding="utf_8_sig").read(),
    )
    if match is None:
        raise SystemExit("Version number not found.")
    return match.group(1)

# ease installation during development
vcs = re.compile(r"(git|svn|hg|bzr)\+")
try:
    with open("requirements.txt") as fp:
        VCS_REQUIREMENTS = [
            str(requirement)
            for requirement in parse_requirements(fp)
            if vcs.search(str(requirement))
        ]
except FileNotFoundError:
    # requires verbose flags to show
    print("requirements.txt not found.")
    VCS_REQUIREMENTS = []

# TODO: Update these values according to the name of the module.
setup(
    name="ai4scr-athena",
    version=read_version("athena/__init__.py"),  # single place for version
    description="ATHENA package provides methods to analyse spatial heterogeneity in spatial omics data",
    long_description=open("README.md").read(),
    url="https://github.com/AI4SCR/ATHENA",
    author="Adriano Martinelli",
    author_email="art@zurich.ibm.com",
    # the following exclusion is to prevent shipping of tests.
    # if you do include them, add pytest to the required packages.
    packages=find_packages(".", exclude=["*tests*"]),
    package_data={"spatialHeterogeneity": ["py.typed"]},
    # entry_points='',
    # scripts=["bin/brief_salutation", "bin/a_shell_script"],
    extras_require={
        "vcs": VCS_REQUIREMENTS,
        "test": ["pytest", "pytest-cov"],
        "dev": [
            # tests
            'pytest==6.2.4',
            'pytest-cov==2.11.1',
            # checks
            'black==21.5b0',
            'flake8==3.9.1',
            'mypy==0.812',
            # docs
            'sphinx==3.5.4',
            'sphinx-autodoc-typehints==1.12.0',
            'better-apidoc==0.3.1',
            'six==1.16.0',
            'sphinx_rtd_theme==0.5.2',
            'myst-parser==0.14',
            #
            'nbconvert==6.5.0',
            'Jinja2<3.1',
            'jupyterlab==3.3.4',
            'colorcet==3.0.0',
            'twine'
        ]
    },
    # versions should be very loose here, just exclude unsuitable versions
    # because your dependencies also have dependencies and so on ...
    # being too strict here will make dependency resolution harder
    install_requires=[
        'scanpy>=1.9.1',
        'scikit-image>=0.19.2',
        'scikit-learn>=1.0.2',
        'scipy>=1.8.0',
        'numpy>=1.21.6',
        'pandas>=1.2',
        'networkx>=2.8',
        'h5py>=3.6.0',
        'tables>=3.7.0',
        'astropy>=5.0.4',
        'tqdm>=4.64.0',
        'napari[all]>=0.4.15',
        'seaborn>=0.11.2',
        'squidpy>=1.2.0',
    ]
)
