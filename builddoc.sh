#!/bin/bash

#                                                                               
# generate doxygen documentation for PySCIPOpt 
#                                  

# stop on error                                                                 
set -e

SOURCE_BRANCH="master"
TARGET_BRANCH="gh-pages"

GH_REPO_ORG=bzfmuelh
GH_REPO_NAME=PySCIPOpt
GH_REPO_REF="github.com/$GH_REPO_ORG/$GH_REPO_NAME.git"
DOXYFILE=$TRAVIS_BUILD_DIR/docs/doxy

# Get the current gh-pages branch
git clone -b gh-pages git@github.com:$GH_REPO_ORG/$GH_REPO_NAME.git code_docs
cd code_docs

##### Configure git.
# Set the push default to simple i.e. push only the current branch.
git config --global push.default simple
# Pretend to be an user called Travis CI.
git config user.name "Travis CI"
git config user.email "travis@travis-ci.org"

# go back to first commit
git reset --hard `git rev-list --max-parents=0 --abbrev-commit HEAD`

# Need to create a .nojekyll file to allow filenames starting with an underscore
# to be seen on the gh-pages site. Therefore creating an empty .nojekyll file.
# Presumably this is only needed when the SHORT_NAMES option in Doxygen is set
# to NO, which it is by default. So creating the file just in case.
echo "" > .nojekyll

################################################################################
##### Generate the Doxygen code documentation and log the output.          #####
echo 'Generating Doxygen code documentation...'
# Redirect both stderr and stdout to the log file AND the console.
doxygen $DOXYFILE 2>&1 | tee doxygen.log

################################################################################
##### Upload the documentation to the gh-pages branch of the repository.   #####
# Only upload if Doxygen successfully created the documentation.
# Check this by verifying that the html directory and the file html/index.html
# both exist. This is a good indication that Doxygen did it's work.
if [ -d "docs/html" ] && [ -f "docs/html/index.html" ]; then
    echo 'Uploading documentation to the gh-pages branch...'
    # Add everything in this directory (the Doxygen code documentation) to the
    # gh-pages branch.
    # GitHub is smart enough to know which files have changed and which files have
    # stayed the same and will only update the changed files.
    git add --all

    # Commit the added files with a title and description containing the Travis CI
    # build number and the GitHub commit reference that issued this build.
    git commit -m "Deploy code docs to GitHub Pages Travis build: ${TRAVIS_BUILD_NUMBER}" -m "Commit: ${TRAVIS_COMMIT}"

    # Force push to the remote gh-pages branch.
    # The ouput is redirected to /dev/null to hide any sensitive credential data
    # that might otherwise be exposed.
    git push --force "git@github.com:$GH_REPO_ORG/$GH_REPO_NAME.git" > /dev/null 2>&1
else
    echo '' >&2
    echo 'Warning: No documentation (html) files have been found!' >&2
    echo 'Warning: Not going to push the documentation to GitHub!' >&2
    exit 1
fi
