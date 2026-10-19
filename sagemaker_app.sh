#!/usr/bin/env bash
# Base Siege — launch the Streamlit app inside SageMaker Studio JupyterLab.
#
# The app registers a custom Streamlit component (ui/frontend, served at
# /component/siege_board) and opens a websocket at /_stcore/stream. Both are
# absolute paths, so the base path Streamlit thinks it is serving under has to
# match the path your browser actually requests.
#
# Studio's JupyterLab exposes two proxy forms:
#   /jupyterlab/default/proxy/<port>/            strips the prefix before forwarding
#   /jupyterlab/default/proxy/absolute/<port>/   forwards the full path
#
#   bash sagemaker_app.sh                  # absolute form (default)
#   bash sagemaker_app.sh --v2             # V2 app only (no TRPO condition)
#   bash sagemaker_app.sh --mode plain     # stripping form, no baseUrlPath
#   bash sagemaker_app.sh --port 8502
#   bash sagemaker_app.sh --background     # detach, log to streamlit.log
#
# If the page loads but the game board is blank, switch modes.

set -euo pipefail

PROJECT_DIR="${BASE_SIEGE_ROOT:-$HOME/base-siege}"
PORT=8501
MODE=absolute
BACKGROUND=0
# V3 entrypoint: runs ui/app.py through runpy with the V3 registry added,
# so the TRPO condition appears without any V2 file being modified.
ENTRY=ui/app_v3.py
JL_APP="${BASE_SIEGE_JL_APP:-default}"   # JupyterLab app name in the proxy path

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)        PROJECT_DIR="$2"; shift 2 ;;
    --port)       PORT="$2"; shift 2 ;;
    --mode)       MODE="$2"; shift 2 ;;
    --background) BACKGROUND=1; shift ;;
    --v2)         ENTRY=ui/app.py; shift ;;
    -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

case "$MODE" in
  absolute|plain) ;;
  *) echo "--mode must be absolute or plain" >&2; exit 2 ;;
esac

export PATH="$HOME/.local/bin:$PATH"
cd "$PROJECT_DIR"
[[ -f "$ENTRY" ]] || { echo "$PROJECT_DIR has no $ENTRY; is this the project root?" >&2; exit 1; }

# Fail early rather than showing a half-working app at the viva.
if find runs -name '*.zip' -size -1024c 2>/dev/null | grep -q .; then
  printf '\033[33m! Some checkpoints under runs/ are still Git LFS pointer stubs.\033[0m\n'
  printf '\033[33m  Policy replay will not work. Run the bootstrap LFS pull first.\033[0m\n'
fi

if command -v lsof >/dev/null 2>&1 && lsof -i :"$PORT" >/dev/null 2>&1; then
  echo "Port $PORT is already in use. Pass --port with a free port." >&2; exit 1
fi

# Studio writes the space identity here; used only to print a clickable URL.
space_host=""
# Deliberately NOT derived from /opt/ml/metadata/resource-metadata.json. The
# browser-facing host is an opaque app id (e.g. k8jaqloqxnlbn7t.studio...),
# which is not SpaceName, ResourceName or anything else in that file. Guessing
# it produces a confident, wrong URL. The path below is the part that matters;
# the host comes from the address bar.
space_host=""

if [[ "$MODE" == "absolute" ]]; then
  BASE_PATH="/jupyterlab/${JL_APP}/proxy/absolute/${PORT}"
  BASE_ARG=(--server.baseUrlPath "$BASE_PATH")
else
  BASE_PATH="/jupyterlab/${JL_APP}/proxy/${PORT}"
  BASE_ARG=()
fi

printf '\n\033[1mOpen this in the browser\033[0m\n'
if [[ -n "$space_host" ]]; then
  printf '  https://%s%s/\n' "$space_host" "$BASE_PATH"
else
  printf '  Take this tab'\''s Studio URL, which looks like\n'
  printf '    https://<id>.studio.<region>.sagemaker.aws/jupyterlab/default/lab/...\n'
  printf '  and replace everything from /lab onwards with:\n\n'
  printf '    proxy%s/%s/\n\n' "$([[ "$MODE" == absolute ]] && echo /absolute)" "$PORT"
  printf '  giving:\n'
  printf '    https://<id>.studio.<region>.sagemaker.aws%s/\n' "$BASE_PATH"
fi
printf '  mode: %s — if the board is blank, re-run with --mode %s\n\n' \
  "$MODE" "$([[ "$MODE" == absolute ]] && echo plain || echo absolute)"

# XSRF/CORS are relaxed because the only route in is the authenticated Studio
# proxy, and the repo default blocks the component handshake through it.
# Error details are turned back on; .streamlit/config.toml hides them, which
# makes a failure here look like an empty page.
ARGS=(
  --server.address 0.0.0.0
  --server.port "$PORT"
  --server.headless true
  --server.enableXsrfProtection false
  --server.enableCORS false
  --server.fileWatcherType none
  --browser.gatherUsageStats false
  --client.showErrorDetails true
  "${BASE_ARG[@]}"
)

if [[ "$BACKGROUND" -eq 1 ]]; then
  nohup uv run --frozen streamlit run "$ENTRY" "${ARGS[@]}" \
    > streamlit.log 2>&1 &
  echo "Started in the background (pid $!). Log: $PROJECT_DIR/streamlit.log"
  echo "Stop it with: kill $!"
else
  echo "Ctrl+C to stop."
  exec uv run --frozen streamlit run "$ENTRY" "${ARGS[@]}"
fi
