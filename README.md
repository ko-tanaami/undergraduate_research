# Hapke and Shkuratov reflectance models

Python implementations for forward reflectance calculations and Shkuratov spectrum fitting.

[日本語](README_ja.md)

## Setup

Install Git and uv, then run:

```sh
git clone https://github.com/ko-tanaami/undergraduate_research.git
cd undergraduate_research
uv sync --locked
```

## Usage

```sh
uv run --locked python hapke_model.py --help
uv run --locked python shkuratov_python/shkuratov_model.py --help
```

Supply optical constants, observations, and configuration files locally. Research datasets, calculated results, figures, and presentation materials are not distributed in this repository. Some examples and validation scripts require local data and cannot run from a clone alone.

The optical-constant validation scripts use `shkuratov_python/data/opcon/`
by default, independently of the working directory. To use another directory,
set `OPCON_DIR` before running them; relative overrides use the current working
directory. The example JSON files use `../data/opcon`, resolved relative to the
JSON file. For another location, edit their `opcon_dir` field explicitly;
`OPCON_DIR` applies to the validation scripts, not the model's JSON loader.
The `data/` directories are ignored by Git; do not commit private input data.
