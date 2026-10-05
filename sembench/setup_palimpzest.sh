#!/bin/bash
# Palimpzest (PyPI `palimpzest`; MIT DSG's semantic-operator system with the Abacus cost-based optimizer) pulls
# in torch and sentence-transformers, so it gets its own interpreter: sembench/.venv-palimpzest (Python 3.12).
# The Palimpzest runners find it there, or through PALIMPZEST_PYTHON. On an Intel Mac the last torch wheel
# (2.2) needs numpy < 2, which in turn needs scipy < 1.13. Needs uv (https://docs.astral.sh/uv/).
set -eu
cd "$(dirname "$0")"
uv venv --python 3.12 .venv-palimpzest
uv pip install --python .venv-palimpzest/bin/python "palimpzest==1.5.3"
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "x86_64" ]; then
	uv pip install --python .venv-palimpzest/bin/python "numpy<2" "scipy<1.13"
fi
echo "Palimpzest ready in sembench/.venv-palimpzest"
