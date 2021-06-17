#!/usr/bin/env bash

set -e

# run this with current directory being a clone of a repo derived from the https://github.ibm.com/CHCLS/blueprint-python-package template

# example to look for occurences yourself
# export FIND="blueprint"; grep -rn . -I --exclude-dir={venv,_build,api,.mypy_cache,.pytest_cache} --exclude="*.egg-info/*" --exclude="*$0" -e "${FIND}"

export REPO_SLUG="$(git config --get remote.origin.url | rev | cut -c 5- | rev | cut -d ':' -f2)"
export USER_OR_ORG=$(cut -d "/" -f1 <<<"${REPO_SLUG}")  # was CHCLS
export REPO_NAME=$(cut -d "/" -f2 <<<"${REPO_SLUG}")  # was blueprint-python-package

if [[ -n "$1" ]]
then
    export PACKAGE_NAME=$1

    # replace text references to github repo
    export FIND="CHCLS"; export REPLACE=${USER_OR_ORG}; grep -rnl . -I --exclude-dir={venv,_build,api,.mypy_cache,.pytest_cache} --exclude="*.egg-info/*" --exclude="*$0" -e "${FIND}" | xargs -n 1 sed -i "" "s/${FIND}/${REPLACE}/g"
    export FIND="blueprint-python-package"; export REPLACE=${REPO_NAME}; grep -rnl . -I --exclude-dir={venv,_build,api,.mypy_cache,.pytest_cache} --exclude="*.egg-info/*" --exclude="*$0" -e "${FIND}" | xargs -n 1 sed -i "" "s/${FIND}/${REPLACE}/g"

    # rename the top level package
    mv blueprint ${PACKAGE_NAME}
    # replace text references to top level package (this is done after changing the repo name that also contained "blueprint")
    export FIND="blueprint"; export REPLACE=${PACKAGE_NAME}; grep -rnl . -I --exclude-dir={venv,_build,api,.mypy_cache,.pytest_cache} --exclude="*.egg-info/*" --exclude="*$0" -e "${FIND}" | xargs -n 1 sed -i "" "s/${FIND}/${REPLACE}/g"
else
    echo "PACKAGE_NAME argument was not passed, did not rename 'blueprint'."
    echo "The following file."
    # list all files that need changes, and check that they should be changed or are under version control
    # in other words have a good look at any files/directories that are in the .gitignore
    grep -rnl . -I --exclude-dir={venv,_build,api,.mypy_cache,.pytest_cache} --exclude="*.egg-info/*" --exclude="*$0" -e "CHCHLS" -e "blueprint-python-package" -e "blueprint"
fi