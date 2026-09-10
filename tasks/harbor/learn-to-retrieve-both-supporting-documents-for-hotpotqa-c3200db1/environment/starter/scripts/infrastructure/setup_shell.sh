. $HOME/.local/bin/env
if [ -f "$HOME/.env" ]; then
    export $(grep -v '^#' "$HOME/.env" | xargs)
fi
export HF_TOKEN=${HF_TOKEN}
export HYDRA_FULL_ERROR=1