# Source this inside WSL before running anything:  source /mnt/g/dev/repo/xev/sentiment/env.sh
# The venv and the model cache live on the Linux filesystem; reading them through /mnt/g is much slower.
export PATH="$HOME/.local/bin:$PATH"          # uv lives here; a non-login shell does not add it
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/xev-sentiment"
export HF_HOME="$HOME/.cache/huggingface"
export UV_LINK_MODE=copy
export TOKENIZERS_PARALLELISM=false
cd "$(dirname "${BASH_SOURCE[0]}")"
