# ut.ils

![Last update](https://img.shields.io/badge/last_update-Jun_13,_2024-green)
![Last update](https://img.shields.io/badge/version-v.0.2.0-blue)

## Project description

A small collection of utility functions for Python-based research projects.

## Project structure

## Install

`ut.ils` is optimized for Python versions  `>=3.8` and can be installed via `pip`:

```shell
pip install -U git+https://github.com/SHEscher/ut.ils.git[vizual]
```

Drop the package in a `requirements.txt` file, `setup.cfg`, or `pyproject.toml` of your project as:

```text
ut @ git+https://github.com/SHEscher/ut.ils.git[vizual]
```

Note the optional dependencies `[vizual]` need to be installed,
when you want to use the visualization utilities in `ut.viz`.

## Get started

```python
# Import the utilities you need
from ut.ils import cprint  # add more

# Use utilities
cprint("Hello! Import all utilities you need from `ut.ils`!", col="g", fm="b")
```

Check out [`src/ut/ils.py`](src/ut/ils.py) for all available utilities.

(*docs will be added at a later times point.*)

### Visualization add-ons

There are some first steps towards functions that help to visualize your data in the `ut.viz` submodule.

```python
from ut.viz import plot_poly_fit

# visualize your data
...
```

Note that you need to install additional dependencies to use the visualization utilities (see above).

## Contributions

*Open for suggestions, and pull requests.*
