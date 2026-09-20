#!/usr/bin/env bash
# Run the full test suite (Python + front-end).
set -euo pipefail

PY=python
if [ -x "./myenv/bin/python" ]; then
  PY="./myenv/bin/python"
fi

echo "== Python tests =="
"$PY" -m pytest -q

echo "== Front-end tests =="
node --test web/tests
