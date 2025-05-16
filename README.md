# Point cloud and pose visualizer

This project is a visualizer for multiple point clouds and their poses. It is designed to be usable both as a standalone GUI and a Python library.



## Project setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

## Linting, formatting, and typechecks

This project template uses the following tools for autoformatting, linting and typechecking:

* ruff: linting and formatting

Pre-commit hooks are used to run these tools automatically when committing to git.

### Pre-commit hooks

Pre-commit hooks check and fix formatting before committing. To use them, after installing `requirements-dev.txt`, install them with `pre-commit install`. The next time you commit (or when you run `pre-commit run`), the hooks will run. If any hooks fail, `git add` the failed and reformatted files again and try `git commit` again. If the hooks still fail, you might need to manually fix errors in the files.

Update hook versions easily with `pre-commit autoupdate`.
