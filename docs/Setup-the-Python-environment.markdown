# Setup the Python environment

## Install Python [if not using `uv`]

If you are not using `uv`, you will need to install Python. How to get this
done is again dependent on your system. See [python.org](https://python.org)
to get started.

## Setup the Python package environment

You can use standard `venv` or a faster manager like `uv`. Latter is
recommended.

### Using uv:

Within the `nehiyawewin-asr` directory run

```
uv tool install .
```

This will setup the necessary Python virtual package environment and install
most of the scripts in nehiyawewin-asr as commandline tools. This means that
they can be run with simply the names given in the documentation as if they
were regular commandline tools like `git`. 

#### Installing for development

If you are installing/setting up for development, you may not want to install
the nehiyawewin-asr scripts as commandline tools. If this is the case, after
installing `uv`, your IDE, and cloning the repository locally, you are done
with setup. Just remember to replace all script calls with `uv run
[path-to-script-file] [and-arguments-to-script]` in the running instructions.

If you do install the scripts as tools with `uv tool install .` you will
probably have to uninstall them and then reinstall them after any updates to
the code. It maybe more convenient to just run them with

```
uv run python [path-to-script-file] [and-arguments-to-script]
```
while making changes and only do the uninstall-reinstall dance when done with
editing. To uninstall the scripts run

```
uv tool uninstall nehiyawewin-asr
```
instead of using `.` to refer to the current directory. A bit unsymmetric, but
this is how `uv` works.


### Using pip:
```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install required dependencies
pip install torch transformers datasets evaluate pandas pyyaml

```

---
Next: [Prepare the Data](Prepare-the-data.markdown)