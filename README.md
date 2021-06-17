# blueprint-python-package
[![Build Status](https://travis.ibm.com/CHCLS/blueprint-python-package.svg?token=xpL9AKxpNBFpJzLTfSTv&branch=master)](https://travis.ibm.com/CHCLS/blueprint-python-package)
[![GitHub Pages](https://img.shields.io/badge/docs-sphinx-blue)](https://pages.github.ibm.com/CHCLS/blueprint-python-package)

Blueprint for a python package to create reproducible assets in a collaboration friendly manner.

## Usage

Click the "Use this template" button (on GitHub Enterprise) to create a new repo (if needed, dashes are recommended over underscores for this name).

Your package is installable for anyone with access to your repo.

### Adapting the blueprint for you: TODOs

Of course, clone your repo.

- [ ] [.travis.yml](.travis.yml) notifications: delete the section if undesired, else set up a slack channel and set the token.
- [ ] If you do not want to build documentation same as last step but also delete the `docs` folder. Else consider deleting the example markdown file.
- [ ] change the name "blueprint" to your desired package name. If really needed, underscores are valid. Note that the repo name is independent. Don't forget to change it in [docs/conf.py](docs/conf.py) and [docs/index.md](docs/index.md).
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

The blueprint contains unit tests in `tests/` directories in each subfolder.
This keeps the tests in close proximity to what they test. See the `setup.py` to include/exclude them with installation.

The test file blueprint/tests/test_module.py contains examples using the recommended `pytest`.
`pytest` will also run unittests
```sh
# also prints a coverage report that requires a percentage
python -m pytest -sv --cov=blueprint --cov-fail-under=65
```

The test file `complex_module/tests/test_core.py` contains an example using `unittests` that is in the standard library.

Run the tests with: 

```sh 
python -m unittest discover -s blueprint -p "test_*" -v
```

You can also use additional scripts to reproducibly test other functionalities beyond unit tests. Consider keep such in a root level tests directory without `__init__.py`.

### CLI commands
With the package, here are also three exemplary command line commands installed in your environment.

`salutation` calls a function defined as an entrypoint in `setup.py`:
```console
(venv) ubuntu@www:~$ salutation --help                             
Usage: salutation [OPTIONS] NAME SURNAME

  Introduce yourself and salute with full name.

Options:
  --pronouns TEXT  Be an ally.
  --help           Show this message and exit.
```
`brief_salutation` was installed as a separate script.  

Both cases use the `click` library to parse command line input arguments.

You can also install other scripts in this fashion, e.g. `a_shell_script` is a bash script:
```console
(venv) ubuntu@www:~$ a_shell_script  
We can also ship bash scripts with the python package if we have to.
What matters is the shebang.
```

### Experimental files
Have a look at the [bench](bench/README.md) to keep your work that is not part of a
package under version control.

### Checks
The following are external tools that analyse your code. Ideally they are integrated into your workflow via your IDE.

These tools can be configured to various degrees. The setup here ensures that there are no conflicting settings.

#### Formatter `black`

When writing code, you should **not** have to worry about how to format it best.
When committing code, it should be formatted in one specific way that reduces meaningless diff changes.  
This is achieved by using an agreed upon formatter in the project. Here, [`black`](https://github.com/psf/black) is recommended (with fixed version).
You can [set up your IDE](https://black.readthedocs.io/en/stable/editor_integration.html) to format your code on save, or add a [pre-commit hook](https://black.readthedocs.io/en/stable/version_control_integration.html).

to check for changes:
```sh
black blueprint --check --diff --color
```
to apply changes:
```sh
black blueprint
```

#### Linter `flake8`
flake8 for one checks code style (mostly a non issue when using black), but also detects various errors by checking the source file syntax trees.

example usage:
```sh
flake blueprint
```

#### Static typing with `mypy`
Modern python allows annotations with type information. These do not affect runtime, and are therefore "optional". However typing will
1. provide documentation for people using the code (including you)
2. force you to reason better about your code
3. find hard to catch errors without running any tests

But yes, it is an investment, especially if you are not used to do it.

example usage:
```sh
mypy blueprint
```

The `py.typed` file is only there to mark the package to support typing.

### Documentation with `sphinx`

In the docs/source directory you can write and add additional documentation on your project. Thanks to [MyST](https://myst-parser.readthedocs.io/en/latest/index.html), you are free to write them in Markdown or reStructuredText.

But actually, a crucial place for documentation is your code itself:
Docstrings and type annotations are used to create an API reference.

Google style docstrings are recommended, see [Example](https://github.com/sphinx-contrib/napoleon/blob/83bf1963096490dd666f93ef5a9ed1cb229fc3ec/docs/source/example_google.py#L66).

To build it locally:
```sh
cd docs && make html && cd ..
```
now open [docs/_build/html/index.html](docs/_build/html/index.html) in your browser.

For examples have a look at
- the [API page](https://pages.github.ibm.com/CHCLS/blueprint-python-package/api/blueprint.module.html) for [blueprint/module.py](blueprint/module.py).
- some [example](https://pages.github.ibm.com/CHCLS/blueprint-python-package/source/wealth_dynamics_md.html) additional documentation. Note that it can refer into the API.

Unlike usual sphinx setups, here
- the api files are automatically regenerated each time instead of once. So refactoring is not an issue at all (if old files stick around, just call `make clean`).
- type hints are taken from the actual type annotations and need not be added to the docstrings. 

### Continuous integration using Travis CI

There is a working `.travis.yml` script in the root of the repository.
Once you activate the [IBM Travis CI service](https://travis.ibm.com) for your repository, 
it will take care of installing the package, running tests, code checks and building a docker image using a clean VM.

On the travis page you will see the build status banner. Click it for the url to replace the banner at the top of this README.

You can enable notifications for build outcomes to a slack channel of your choice (e.g. create a dedicated "project-bots" channel):

```yaml
notifications:
  slack:
    rooms:
      - ibm-research:<TOKEN>#<CHANNEL>
    on_success: always
    on_failure: always
```

See [Travis CI documentation](https://docs.travis-ci.com/user/notifications/#configuring-slack-notifications) for more info.

Have a look a the example [slack channel for the blueprint bots](https://ibm-research.slack.com/archives/C02167RH2LW)


### Docker support

The blueprint contains a `Dockerfile` that builds an image containing the python package.
At the moment, it is based on the image `python:3.7` and this can be adapted to your needs. 
Docker images can be stored in a docker registry for later use.
IBM TaaS offers the possibility to create an enterprise docker registry on Artifactory. 
See [here](https://pages.github.ibm.com/TAAS/tools_guide/artifactory/getting-started.html).

#### Deployment example on IBM Artifactory

Assuming the following environment variables are set:
- `DOCKER_USER`: your w3id email.
- `DOCKER_PASSWORD`: token (obtained from IBM Artifactory).
- `DOCKER_REGISTRY`: url of Artifactory registry. E.g.: `blueprint-docker-local.artifactory.swg-devops.com`.
- `DOCKER_IMAGE`: name of the image. 
- `DOCKER_TAG`: version, tag information. E.g.: `latest`, `test`.

```sh 
docker login -u ${DOCKER_USER} -p ${DOCKER_PASSWORD}  ${DOCKER_REGISTRY}
docker build -t ${DOCKER_REGISTRY}/${DOCKER_IMAGE}:${DOCKER_TAG} .
docker push ${DOCKER_REGISTRY}/${DOCKER_IMAGE}:${DOCKER_TAG} 
```

Note that this can also be automated on travis with a [deploy script](https://docs.travis-ci.com/user/deployment) like [.travis/deploy.sh](.travis/deploy.sh). Remember to define secret variables only in the travis repo settings.

### VCS
As seen above the [VCS url](https://pip.pypa.io/en/stable/cli/pip_install/#vcs-support) for installation can be used to install the package directly from GitHub. It might happen that your package itself depends on other repos in development on GitHub Enterprise, that are not on a pypi like server (that allows proper versioning of python packages). We can add such VCS urls also to the `requirements.txt`. However, these should not be added to the `install_requires` list in `setup.py `, it will lead to conflicts at some point (been there, done that).
Instead -to enable easy installation of all requirements during development- there is an extra tag to include the VCS urls from the `requirements.txt`:

```sh
pip install "NEW_REPOSITORY[vcs] @ git+ssh://git@github.ibm.com/USERNAME_OR_ORGANIZATION/NEW_REPOSITORY.git@master"
```

Or you can just install from the requirements file first to ensure exact versions...


Note that if the VCS dependencies use the ssh protocol, the environment requires a valid ssh key. For example
- you might have to add an ssh key on the travis repo settings
- you might require it during the docker build. Know that doing this wrong might expose the ssh key. Here some advice (also see [docker docs](https://docs.docker.com/develop/develop-images/build_enhancements/#using-ssh-to-access-private-data-in-builds))

```dockerfile
# add credentials on build
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.ibm.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh pip install --no-cache-dir -r requirements.txt
```

and run it with the `--ssh` argument (requires docker>=18.09 and enabled BuildKit)
```
DOCKER_BUILDKIT=1 docker build --ssh default -t blueprint:test .
```
If you can't use BuildKit, have a look at multi-stage builds or the --sqash flag.
