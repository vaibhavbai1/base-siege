#!/bin/zsh
set -eu
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  print 'Install the project environment first: uv sync --frozen'
  exit 1
fi
exec .venv/bin/python -m streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8512
