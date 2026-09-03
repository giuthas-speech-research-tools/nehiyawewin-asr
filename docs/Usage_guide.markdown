# Usage guide

To use the nehiyawewin-asr scripts you will need a couple of basic command line
tools: `git` and `uv`. `Git` is used for version control and for cloning the
repository (that is where the code is kept) and `uv` is used for setting up the
Python environment and running the scripts. 

In principle, if you are not going to modify the scripts, you could skip
installing `uv`, but in practice using standard tools like Python's `venv` will
be slower and potentially more error prone. The instructions below will
occasionally give instructions for how to use tools other than `uv`, but those
are untested and not recommended. They are only provided in case somebody needs
a starting point for setting up a system where `uv` is not an option for some
reason.

If you would like to or need to modify the scripts, you should also get a IDE
such as VSCodium or whatever is your favourite Python development environment.
Going this route will be referred to as 'installing for development' or similar
in the rest of the documentation.

Git also has various GUI implementations, but those are operating system
dependent and so you will have to look them up and select the one you would
like yourself. Git can be used from just the commandline without problems
though.

Here are some potentially useful links to these tools:
- [Getting `git`](https://git-scm.com/install/)
- [Getting `uv`](https://docs.astral.sh/uv/getting-started/installation/)
- [Getting VSCodium](https://vscodium.com/)
- [Getting Python](https://python.org)

In addition to the above, you should also create a
[github](https://github.com/) account and do any setup you need to do on your
local system to be able to clone repositories from github such as setting
access tokens and so on.

Getting everything in place to train ASR models or run them, takes a few steps.
These are explained below.

## Installations

- [Clone the code repository](Clone-the-code-repository.markdown)
- [Setup the Python environment](Setup-the-Python-environment.markdown)

## Data preparation

- [Prepare the Data](Prepare-the-data.markdown)
- [Upload files to HPC](Upload-files-to-HPC.markdown)

## Running trainings and checking the results

- [Run the training](Run-the-training.markdown)
- [Analyse the results](Analyse-the-results.markdown)

