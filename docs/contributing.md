# Contributing

This document describes how to contribute to the development of this software.

## Bug Reports

If you find a but we need to know about it so we can fix it. Please report your [bugs on our GitHub Issues
page][gh-bug].

## Feature Requests

If you find AFMReader useful but think it can be improved you can [make a feature request][gh-feature].

## Code Contributions

If you would like to fix a bug or add a new feature that is great, Pull Requests are very welcome.

However, we have adopted a number of good software development practises that ensure the code and documentation is
linted and that unit and regression tests pass both locally and on Continuous Integration. The rest of this page helps
explain how to set yourself up with these various tools.

## Virtual Environments

Use of [virtual environments][venv], particularly during development of Python packages, is encouraged. There are lots of
options out there for you to choose from including...

- [Miniconda][miniconda]
- [venv][venv]
- [virtualenvwrapper][virtualenvwrapper]

Which you choose is up to you, although you should be wary of using the [Miniconda][miniconda] distribution from
[Anaconda][anaconda] if any of your work is carried out for or in conjunction with a commercial entity.

### uv

Developers are using the [uv][uv] package manager to setup and control environments to which end a [`uv.lock`][uv-lock]
file is included in the repository. [uv][uv] supports managing virtual environments so you may wish to install and use
this tool at the system level to manage your virtual environments for this package.

## Cloning the Repository

Once you have setup your virtual environment you should clone the repository from GitHub

```bash
cd ~/path/you/want/to/clone/to
git clone https://github.com/AFM-SPM/AFMReader
```

### Install development

Once you have cloned the AFMReader repository you should install all the package along with all development and
documentation dependencies in "editable" mode. This means you can test the changes you make in real time.

```bash
cd AFMReader
pip install --no-cache-dir -e .[docs,dev,tests]
```

## Git

[Git][git] is used to version control development of the package. The `main` branch on GitHub and the [pre-commit
hooks](#pre-commit) have protections in place that prevent committing/pushing directly to the `main`
branch. This means you should create a branch to undertake development or fix bugs.

### Issues

Ideally an [issue][afmreader-issue] should have been created detailing the feature request. If it is a large amount of
work this should be captured in an issue labelled "_Epic_" and the steps taken to achieve all work broken down into
smaller issues.

### Branch nomenclature

When undertaking work on a particular issue it is useful to use informative branch names. These convey information about
what the branch is for beyond simply "`adding-feature-x`". We ask that you create the branch using your GitHub username,
followed by the issue number and a short description of the work being undertaken. For example
`ns-rse/1021-add-xyz-support` as this allows others to know who has been undertaking the work, what issue the work
relates to and has an informative name as to the nature of that work.

### Conventional Commits

We also ask that you try and follow the [Conventional Commits][conventional-commits] pattern for titling your commits
and where required include additional information on _why_ you have made the changes you are committing.

## Linting

[Linting][linting] is the practice of following a consistent coding style. For Python that style is defined in
[PEP8][pep8]. By following a consistent style across a code base it is easier to read and understand the code written by
others (including your past self!). We use the following linters implemented as [pre-commit](#pre-commit) hooks

- Python
  - [Black][black]
  - [Blacken-docs][blacken-docs]
  - [flake8][flake8]
  - [Numpydoc][numpydoc-validation]
  - [Ruff][ruff]
- Other
  - [Codespell][codespell] (Spelling across all filesyy)
  - [markdownlint-cli2][markdownlint-cli2] (Markdown)
  - [prettier][prettier] (Markdown, YAML)

## Pre-commit

Style checks are made using the [pre-commit][pre-commit] framework which is one of the development dependencies and
should have been installed in the previous step. You can check if its is installed in your virtual environment with `pip
show pre-commit`. If you have pre-commit installed install the hook using...

```bash
pre-commit install
```

This adds a file to `.git/hooks/pre-commit` that will run all of the hooks specified in `.pre-commit-config.yaml`. The
first time these are run it will take a little while as a number of virtual environments are downloaded for the first
time. It might be a good time to run these manually on the code base you have just cloned which should pass all checks.

```bash
pre-commit run --all-files
```

These should all pass. Now whenever you try to make a `git commit` these checks will run _before_ the commit is made and
if any fail you will be advised of what has failed. Some of the linters such as [Black][black] and [Ruff][ruff] will
automatically correct any errors that they find and you will have to stage the files that have changed again. Not all
errors can be automatically corrected (e.g. [Numpydoc validation][numpydoc-validation] and [Pylint][pylint]) and you
will have to manually correct these.

## Docstrings

It is sensible and easiest to write informative docstrings when first defining your modules, classes and
methods/functions. Doing so is a useful _adie-memoire_ not only for others but your future self and with modern Language
Servers that will, on configuration, show you the docstrings when using the functions it helps save time.

You will find your commits fail the [numpydoc-validation][numpydoc-validation] pre-commit hook if you do not write
docstrings and will be prompted to add one.

## Testing

We use the [pytest][pytest] framework with various plugins in our testing suite. When correcting bugs and adding
features at a bare minimum the existing tests should not fail. Where possible we would be grateful of contributions to
the test suite. This means if an edge case has been identified and a solution derived a test is added that checks the
edge case is correctly handled. For new features would ideally mean writing [unit-tests][unit-tests] to ensure each
function or method works as intended and for larger classes that behaviour is as expected. Sometimes tests will need
updating in light of bug fixes and features which is to be expected, but remember to commit updates to tests as well as
to code to ensure the Continuous Integration tests pass.

### Pytest-testmon

To shorten the feedback loop during development the [pytest-testmon][pytest-testmon] plugin is used as a
[pre-commit](#pre-commit) hook so that only the tests affected by the changes that are being committed are run. This
requires that on first installing the package you create a local database of the state of the tests by running the
following...

```bash
pytest --test-mon
```

This creates the files `.testmondata` which stores the current state of tests. Once created commits will only run
affected tests. However if your environment has changed, such as adding new packages or updating installed packages you
will have to recreate the database.

## Pull Requests

Once you have made your changes and committed them you will at some point wish to make a [Pull Request][gh-pr] to merge
them into the `main` branch.

In order to keep Git history clean and easier to understand you can perform an interactive [`git rebase -i`][git-rebase]
on your feature branch to squash related commits and tidy up your commit history.

When your branch is ready for merging with `main` open a [Pull Request][gh-pr]. You can use the [GitHub
keywords][gh-keywords] of `close[s|d]`/`fix[es|ed]` / `resolve[s|d]` followed by the issue number in the body of your
commit message which will change the status of the issue to "_Closed_" when the Pull Request is merged.

Pull Requests will be reviewed in a timely and hopefully constructive manner.

[anaconda]: https://www.anaconda.com/blog/update-on-anacondas-terms-of-service-for-academia-and-research
[black]: https://black.readthedocs.io/en/stable/index.html
[blacken-docs]: https://github.com/adamchainz/blacken-docs
[codespell]: https://github.com/codespell-project/codespell
[conventional-commits]: https://www.conventionalcommits.org/en/v1.0.0/
[flake8]: https://flake8.pycqa.org/en/latest/
[gh-bug]: https://github.com/AFM-SPM/AFMReader/issues/new?assignees=&labels=bug&projects=AFM-SPM%2F1&template=bug_report.yaml&title=%5Bbug%5D%3A+
[gh-feature]: https://github.com/AFM-SPM/AFMReader/issues/new?assignees=&labels=enhancement&projects=AFM-SPM%2F1&template=feature_request.yaml&title=%5Bfeature%5D+%3A+
[gh-keywords]: https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/using-keywords-in-issues-and-pull-requests
[gh-pr]: https://github.com/AFM-SPM/AFMReader/pulls
[afmreader-issue]: https://github.com/AFM-SPM/AFMReader/issues/
[git]: https://git-scm.com
[git-rebase]: https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History
[linting]: https://en.wikipedia.org/wiki/Lint_(software)
[markdownlint-cli2]: https://github.com/DavidAnson/markdownlint-cli2
[miniconda]: https://docs.anaconda.com/miniconda/
[numpydoc-validation]: https://numpydoc.readthedocs.io/en/latest/validation.html
[pep8]: https://peps.python.org/pep-0008/
[pre-commit]: https://pre-commit.com/
[prettier]: https://prettier.io/docs/en/
[pylint]: https://www.pylint.org/
[pytest]: https://pytest.org
[pytest-testmon]: https://pypi.org/project/pytest-testmon/
[ruff]: https://docs.astral.sh/ruff
[unit-tests]: https://en.wikipedia.org/wiki/Unit_testing
[uv]: https://docs.astral.sh/uv/
[uv-lock]: https://docs.astral.sh/uv/concepts/projects/#project-lockfile
[venv]: https://docs.python.org/3/library/venv.html
[virtualenvwrapper]: https://virtualenvwrapper.readthedocs.io/en/latest/
