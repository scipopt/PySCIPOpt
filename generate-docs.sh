#!/bin/bash -e

GITHUB_TOKEN=$1
if [ $GITHUB_TOKEN == "" ]; then
    echo "Fatal: Missing access token, exiting."
    exit 1
fi

# for live documentation branch set DOCBRANCH=gh-pages
DOCBRANCH=$2
if [ $2 == "" ]; then
    echo "Fatal: Missing branch, exiting."
    exit 1
fi

# get repo info
REPO_SLUG=$GITHUB_REPOSITORY
BUILD_COMMIT=$GITHUB_SHA
BUILD_NUMBER=$GITHUB_RUN_ID

GH_REPO_ORG=`echo $REPO_SLUG | cut -d "/" -f 1`
GH_REPO_NAME=`echo $REPO_SLUG | cut -d "/" -f 2`
GH_REPO_REF="github.com/$GH_REPO_ORG/$GH_REPO_NAME.git"

#get SCIP TAGFILE
echo "Downloading SCIP tagfile to create links to SCIP docu"
wget -q -O docs/scip.tag https://scip.zib.de/doc/scip.tag

#get version number for doxygen
export VERSION_NUMBER=$(grep "__version__" src/pyscipopt/__init__.py | cut -d ' ' -f 3 | tr --delete \')

# generate html documentation in docs/html
echo "Generating documentation"
doxygen docs/doxy

# fix broken links to SCIP online documentation
# If you set `HTML_FILE_EXTENSION    = .php` in doc/doxy you don't need the following sed commands
sed -i "s@\.php\.html@.php@g" docs/html/*.* docs/html/search/*.*
sed -i -E "s@(scip.zib.de.*)\.html@\1.php@g" docs/html/*.* docs/html/search/*.*

# clone the docu branch
echo "cloning branch ${DOCBRANCH} from repo git@github.com:${GH_REPO_ORG}/${GH_REPO_NAME}"
git clone -b ${DOCBRANCH} https://${GH_REPO_ORG}:${GITHUB_TOKEN}@github.com/${GH_REPO_ORG}/${GH_REPO_NAME} code_docs
cd code_docs

##### Configure git.
# Set the push default to simple i.e. push only the current branch.
git config --global push.default simple
# Pretend to be an user called SCIP CI Bot
git config user.name "SCIP CI Bot"
git config user.email "timo-admin@zib.de"

git remote set-url --push origin https://${GH_REPO_ORG}:${GITHUB_TOKEN}@github.com/${GH_REPO_ORG}/${GH_REPO_NAME}

# go back to first commit
git reset --hard `git rev-list --max-parents=0 --abbrev-commit HEAD`

# copy new docu files to gh-pages
mkdir -p docs/html
mv ../docs/html/* docs/html/
git add --all
git commit -m "Deploy docs to GitHub Pages, GitHub Actions build: ${BUILD_NUMBER}" -m "Commit: ${BUILD_COMMIT}"

# Force push to the remote gh-pages branch.
# The ouput is redirected to /dev/null to hide any sensitive credential data
# that might otherwise be exposed.
git push --force git@github.com:${GH_REPO_ORG}/${GH_REPO_NAME} > /dev/null 2>&1
