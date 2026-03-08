# Contributing to CODLING

First off, thank you for considering contributing to CODLING! It's people like you that make CODLING such a great tool.

## Where do I go from here?

If you've noticed a bug or have a feature request, please make sure to check our issue tracker to see if someone else in the community has already created a ticket. If not, go ahead and make one!

## Fork & create a branch

If this is something you think you can fix, then fork CODLING and create a branch with a descriptive name.

A good branch name would be (where issue #325 is the ticket you're working on):

```sh
git checkout -b feature/#325-add-new-activation-function
```

## Get the test suite running

Make sure you have installed the project requirements. You can do this by running:

```sh
pip install -r codling/requirements.txt
```

Before committing your changes, please run the test suite to make sure nothing is broken:

```sh
pytest
```

## Implement your fix or feature

At this point, you're ready to make your changes. Feel free to ask for help; everyone is a beginner at first. 

* Please adhere to Google Coding patterns and write very clean, optimized code.
* Avoid adding excessive comments; let the code speak for itself where possible.
* Ensure all files and variables have meaningful names.

## Make a Pull Request

At this point, you should switch back to your master branch and make sure it's up to date with CODLING's master branch:

```sh
git remote add upstream https://github.com/RaheesAhmed/codling.git
git checkout main
git pull upstream main
```

Then update your feature branch from your local copy of master, and push it!

```sh
git checkout feature/#325-add-new-activation-function
git rebase main
git push --set-upstream origin feature/#325-add-new-activation-function
```

Finally, go to GitHub and make a Pull Request!

## Keeping your Pull Request updated

If a maintainer asks you to "rebase" your PR, they're saying that a lot of code has changed, and that you need to update your branch so it's easier to merge.

Thank you for contributing!
