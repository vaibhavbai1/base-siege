#!/usr/bin/env bash
# Base Siege — SageMaker Studio bootstrap.
#
# Clones (or reuses) the repo, hydrates only the Git LFS objects you asked for,
# builds the locked Python 3.12 environment, registers a Jupyter kernel and
# verifies the twenty approved V2 checkpoints.
#
# Safe to re-run: every step is skipped if it is already done.
#
#   bash sagemaker_bootstrap.sh                   # models-only LFS (~91 MiB)
#   bash sagemaker_bootstrap.sh --lfs all         # every LFS object (~5.3 GiB)
#   bash sagemaker_bootstrap.sh --lfs none        # code only, no checkpoints
#   bash sagemaker_bootstrap.sh --skip-clone      # repo already on disk
#   bash sagemaker_bootstrap.sh --tests           # also run the test suite

set -euo pipefail

REPO_URL="${BASE_SIEGE_REPO:-https://github.com/vaibhavbai1/base-siege.git}"
PROJECT_DIR="${BASE_SIEGE_ROOT:-$HOME/base-siege}"
LFS_MODE="models"
SKIP_CLONE=0
RUN_TESTS=0

# The twenty checkpoints the application registry (ui/v2_checkpoints.json)
# approves: 4,194,304-step milestones for screening seeds 0-2, and the final
# model.zip for replication seeds 3-9.
#
# runs/study-v2/P-0/model.zip is NOT in the registry but IS needed by the test
# suite: the existing_policy fixture in tests/test_v2_application.py takes the
# first candidate path passing Path.is_file(), and an LFS pointer stub passes
# that check. Left unhydrated it shadows the registry checkpoint, and four
# tests error with "wasn't a zip-file".
#
# P-0 only, deliberately: the fixture reads no other seed, and widening this
# to runs/study-v2/*/model.zip pulls 20 files totalling 91.3 MiB to no end.
LFS_MODEL_GLOBS='runs/version2-longer-control/*/milestones/step-000004194304.zip,runs/version2-seed-replication/*/model.zip,runs/study-v2/P-0/model.zip'

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)        PROJECT_DIR="$2"; shift 2 ;;
    --repo)       REPO_URL="$2"; shift 2 ;;
    --lfs)        LFS_MODE="$2"; shift 2 ;;
    --skip-clone) SKIP_CLONE=1; shift ;;
    --tests)      RUN_TESTS=1; shift ;;
    -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

case "$LFS_MODE" in
  models|all|none) ;;
  *) echo "--lfs must be one of: models, all, none" >&2; exit 2 ;;
esac

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[33m  ! %s\033[0m\n' "$1"; }
ok()   { printf '\033[32m  ok %s\033[0m\n' "$1"; }

# --------------------------------------------------------------------------
step "Preflight"

if ! command -v git >/dev/null 2>&1; then
  echo "git not found. Install it before running this script." >&2; exit 1
fi

# SageMaker Studio images do not ship git-lfs. Try conda first (present on
# Studio images, needs no root), then apt, then an unpacked binary.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v git-lfs >/dev/null 2>&1; then
  warn "git-lfs not found. Trying to install it."

  if command -v conda >/dev/null 2>&1; then
    conda install -y -q -c conda-forge git-lfs || true
  elif command -v micromamba >/dev/null 2>&1; then
    micromamba install -y -q -c conda-forge git-lfs || true
  fi

  if ! command -v git-lfs >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
    if [[ $EUID -eq 0 ]]; then
      apt-get update -qq && apt-get install -y -qq git-lfs || true
    elif sudo -n true 2>/dev/null; then
      sudo apt-get update -qq && sudo apt-get install -y -qq git-lfs || true
    fi
  fi

  # Last resort: the official release tarball into ~/.local/bin, no root.
  if ! command -v git-lfs >/dev/null 2>&1; then
    warn "Falling back to the git-lfs release binary."
    # Pinned rather than looked up: the unauthenticated GitHub API is
    # rate-limited and answers 403 often enough to break a bootstrap.
    tmp=$(mktemp -d); mkdir -p "$HOME/.local/bin"
    url="https://github.com/git-lfs/git-lfs/releases/download/v${GIT_LFS_VERSION:-3.7.0}/git-lfs-linux-amd64-v${GIT_LFS_VERSION:-3.7.0}.tar.gz"
    if curl -fsSL "$url" | tar xz -C "$tmp" --strip-components=1; then
      install -m 0755 "$tmp/git-lfs" "$HOME/.local/bin/git-lfs"
    fi
    rm -rf "$tmp"
  fi

  command -v git-lfs >/dev/null 2>&1 || {
    echo "Could not install git-lfs. Install it by hand, then re-run." >&2; exit 1; }
fi
git lfs install --skip-smudge >/dev/null
ok "git $(git --version | awk '{print $3}'), $(git lfs version | awk '{print $1}')"

# Studio's 50 GB space volume is the binding constraint, not RAM.
avail_gib=$(df -BG --output=avail "$(dirname "$PROJECT_DIR")" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
printf '  free space on %s: %s GiB\n' "$(dirname "$PROJECT_DIR")" "${avail_gib:-?}"
if [[ "${avail_gib:-0}" -lt 25 ]]; then
  warn "Under 25 GiB free. The locked CUDA environment alone needs roughly 8-10 GiB."
fi

# --------------------------------------------------------------------------
step "uv"

if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "uv install failed." >&2; exit 1; }
ok "uv $(uv --version | awk '{print $2}')"

# Keep uv's cache on the persistent volume so a restarted space does not
# re-download torch, but remember it counts against the 50 GB.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"

# --------------------------------------------------------------------------
step "Project source"

if [[ "$SKIP_CLONE" -eq 0 && ! -d "$PROJECT_DIR/.git" ]]; then
  if [[ -e "$PROJECT_DIR" ]]; then
    echo "$PROJECT_DIR exists but is not a git clone. Move it aside or pass --skip-clone." >&2
    exit 1
  fi
  # Smudge is skipped so the clone does not drag down 5.3 GiB of LFS objects.
  GIT_LFS_SKIP_SMUDGE=1 git clone "$REPO_URL" "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"
[[ -f env/siege_env.py ]] || { echo "$PROJECT_DIR is not the Base Siege project root." >&2; exit 1; }
ok "project at $PROJECT_DIR"

# --------------------------------------------------------------------------
step "Git LFS objects (mode: $LFS_MODE)"

case "$LFS_MODE" in
  models)
    git lfs pull --include="$LFS_MODEL_GLOBS"
    ok "pulled the 20 approved checkpoints plus the study-v2 endpoints (~183 MiB)"
    ;;
  all)
    warn "Pulling every LFS object: ~5.3 GiB fetched, and LFS keeps a second"
    warn "copy under .git/lfs/objects, so budget ~10.6 GiB."
    git lfs pull
    ok "pulled all LFS objects"
    ;;
  none)
    warn "No LFS objects pulled. Policy replay and the checkpoint check will fail."
    ;;
esac

if [[ "$LFS_MODE" != "none" ]]; then
  # A pointer stub is ~132 bytes; a real checkpoint is ~4.8 MB. Use -1024c,
  # not -1k: find rounds 1K blocks up, so -1k never matches a small file.
  stubs=$(find runs -name '*.zip' -size -1024c 2>/dev/null | wc -l | tr -d ' ')
  [[ "$stubs" -gt 0 ]] && warn "$stubs .zip files under runs/ are still LFS pointer stubs."
fi

# --------------------------------------------------------------------------
step "Python 3.12 environment"

uv python install 3.12
uv sync --frozen --python 3.12
ok "locked environment built in .venv"

# --------------------------------------------------------------------------
step "Jupyter kernel"

uv run --frozen python -m ipykernel install --user \
  --name base-siege --display-name "Python 3 (Base Siege)"
ok 'select "Python 3 (Base Siege)" in the notebooks'

# --------------------------------------------------------------------------
step "Runtime report"

uv run --frozen python - <<'PY'
import platform, shutil, sys
from importlib.metadata import version
import torch

print(f"  python     {platform.python_version()}  ({sys.executable})")
for pkg in ("torch", "numpy", "gymnasium", "stable-baselines3", "sb3-contrib", "streamlit"):
    print(f"  {pkg:<18} {version(pkg)}")
print(f"  cuda available  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  gpu             {torch.cuda.get_device_name(0)}  (cuda {torch.version.cuda})")
else:
    print("  gpu             none -- training with --device cuda will raise, not fall back")
print(f"  free space      {shutil.disk_usage('.').free / 1024**3:.1f} GiB")
PY

# --------------------------------------------------------------------------
step "Checkpoint verification"

if [[ "$LFS_MODE" == "none" ]]; then
  warn "Skipped: no checkpoints on disk."
else
  if uv run --frozen python -m ui.v2_models check; then
    ok 'expect "status": "ready" and "verified_v2_models": 20 above'
  else
    warn "Verification failed. Re-run the LFS pull; do not edit the expected hashes."
  fi
fi

# --------------------------------------------------------------------------
if [[ "$RUN_TESTS" -eq 1 ]]; then
  step "Test suite"
  uv run --frozen pytest -q
fi

step "Done"
cat <<EOF
  Project   $PROJECT_DIR
  Launch    bash sagemaker_app.sh
  Notebooks open notebooks/00_sagemaker_setup.ipynb with the Base Siege kernel

  If uv is not on PATH in a new terminal:
      export PATH="\$HOME/.local/bin:\$PATH"
EOF
