# Point cloud and pose visualizer

This project is a visualizer for multiple point clouds and their poses. It is designed to be usable both as a standalone GUI and a Python library.

It is based on the Open3D visualization tutorial scripts.

## User setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .
```

Conda also works just as well as virtualenv.

Additional instructions for the CLI are found in: `pviz --help`.

If you experience segfault errors when opening the GUI, try adding an environment variable: `export XDG_SESSION_TYPE=x11`.

### Library usage

Follow the `cli()` function for an example on how to use the library in your own code.

## Dev setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --editable .[dev]
pre-commit install
```

Conda also works just as well as virtualenv.

### Linting, formatting, and pre-commit checks

This project template uses the following tools for autoformatting, linting and checks:

* pre-commit-hooks: small checks, e.g. for file names and end-of-file lines
* ruff: linting and formatting

Pre-commit is used to run these tools automatically when committing to git.

#### Pre-commit hooks

Pre-commit hooks check and fix formatting before committing. To use them, after installing `requirements-dev.txt`, install them with `pre-commit install`. The next time you commit (or when you run `pre-commit run`), the hooks will run. If any hooks fail, `git add` the failed and reformatted files again and try `git commit` again. If the hooks still fail, you might need to manually fix errors in the files.

Update hook versions easily with `pre-commit autoupdate`.
