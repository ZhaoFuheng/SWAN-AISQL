#!/bin/bash
# LOTUS (PyPI `lotus-ai`) gets its own interpreter, aisql-bench/.venv-lotus (Python 3.12), with the versions its
# recorded runs replay under (../requirements-lotus.txt: lotus-ai 1.2.4, litellm 1.92.0). Run the LOTUS runners
# with it, for example:  .venv-lotus/bin/python MOVIE/lotus_movie.py
# Needs uv (https://docs.astral.sh/uv/).
set -eu
cd "$(dirname "$0")"
uv venv --python 3.12 .venv-lotus
uv pip install --python .venv-lotus/bin/python -r ../requirements-lotus.txt
echo "LOTUS ready in aisql-bench/.venv-lotus"
