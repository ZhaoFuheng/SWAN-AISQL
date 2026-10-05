#!/bin/bash
# ThalamusDB (PyPI `thalamusdb`, the version SemBench evaluated) pins its dependencies, so it gets its own
# interpreter: sembench/.venv-thalamusdb (Python 3.12). The ThalamusDB runners find it there, or through
# THALAMUSDB_PYTHON. Needs uv (https://docs.astral.sh/uv/).
set -eu
cd "$(dirname "$0")"
uv venv --python 3.12 .venv-thalamusdb
uv pip install --python .venv-thalamusdb/bin/python "thalamusdb==0.1.15"
echo "ThalamusDB ready in sembench/.venv-thalamusdb"
