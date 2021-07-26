# spatial-heterogeneity
[![Build Status](https://travis.ibm.com/art-zurich/spatial-heterogeneity.svg?token=bmUqdLriQp1g3yv7TJC6&branch=master)](https://travis.ibm.com/art-zurich/spatial-heterogeneity)
[![GitHub Pages](https://img.shields.io/badge/docs-sphinx-blue)](https://pages.github.ibm.com/art-zurich/spatial-heterogeneity)

Blueprint for a python package to create reproducible assets in a collaboration friendly manner.

## Usage

Click the "Use this template" button (on GitHub Enterprise) to create a new repo (if needed, dashes are recommended over underscores for this name).

Your package is installable for anyone with access to your repo.

### Adapting the spatialHeterogeneity for you: TODOs

Of course, clone your repo.

- [ ] [.travis.yml](.travis.yml) notifications: delete the section if undesired, else set up a slack channel and set the token.
- [ ] If you do not want to build documentation same as last step but also delete the `docs` folder. Else consider deleting the example markdown file.
- [ ] change the name "spatialHeterogeneity" to your desired package name. If really needed, underscores are valid. Note that the repo name is independent. Don't forget to change it in [docs/conf.py](docs/conf.py) and [docs/index.md](docs/index.md).
- [ ] Update author information in [setup.py](setup.py) and [docs/conf.py](docs/conf.py).
- [ ] If you decide against some checks, remove them from .travis.yml and the respective tools from the development requirements ([setup.py](setup.py) "dev" extras and [dev_requirements.txt](dev_requirements.txt)).
- [ ] Set up Travis, including a github token in case of pushing docs.
- [ ] If not building a docker image, remove the Dockerfile, the .travis directory and the "Docker" build stage in .travis.yml. Else you will have to implement .travis/deploy.sh at some point.
- [ ] Make this README file your own, and update the different banners.

Happy developing and documenting! 
### Install
```sh
# assuming you have a ssh key set up on GitHub
pip install "git+ssh://git@github.ibm.com/USERNAME_OR_ORGANIZATION/NEW_REPOSITORY.git@master"
```

see the [VCS](#vcs) paragraph how to handle requirements from GitHub or other version control system (VCS)

### Suggested setup for development

Create a `virtualenv`:

```sh
python -m venv venv
```

Activate it:

```sh
source venv/bin/activate
```

Install the package (including required packages) as "editable", meaning changes to your code do not require another installation to update.

```sh
pip install -e ".[dev]"
# or `pip install -e ".[vcs,dev]"`  # if you rely on other packages from github
```

To ensure exact versions:
```sh
pip install -r requirements.txt
pip install -r dev_requirements.txt
pip install -e .
```

#### Python version

Newer versions are generally to be preferred, but consider if your package will be used in some cases that require backward compatibility.
For example on power pc architecture (e.g. on ZHC2) you are stuck with python 3.7.

A good way to install and manage multiple versions of python itself is `pyenv`. The other sane way is using `conda`, where you should create a conda virtual environment instead via `python -m venv`.

On older systems where python 2 is default, using the preinstalled python 3 you might need to use explicitly `python3` and `pip3`.
## Features

### Tests

The spatialHeterogeneity contains unit tests in `tests/` directories in each subfolder.
This keeps the tests in close proximity to what they test. See the `setup.py` to include/exclude them with installation.

The test file spatialHeterogeneity/tests/test_module.py contains examples using the recommended `pytest`.
`pytest` will also run unittests
```sh
# also prints a coverage report that requires a percentage
python -m pytest -sv --cov=spatialHeterogeneity --cov-fail-under=65
```

The test file `complex_module/tests/test_core.py` contains an example using `unittests` that is in the standard library.

Run the tests with: 

```sh 
python -m unittest discover -s spatialHeterogeneity -p "test_*" -v
```

You can also use additional scripts to reproducibly test other functionalities beyond unit tests. Consider keep such in a root level tests directory without `__init__.py`.