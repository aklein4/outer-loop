#!/usr/bin/env bash

(
  set -euo pipefail

  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  cd "$SCRIPT_DIR"

  VENV_PYTHON="$HOME/iTTT/src/.venv/bin/python"
  SCRIPT="scripts/visualize_oloop_lora.py"
  CHECKPOINT="aklein4/horizon-v2_baseline"
  STEP=350
  BASE_LR=3e-4
  AUX_WEIGHT=0.1

  "$VENV_PYTHON" "$SCRIPT" \
    checkpoint_url="$CHECKPOINT" \
    checkpoint_step="$STEP" \
    base_lr="$BASE_LR" \
    adaptation.aux_loss_weight="$AUX_WEIGHT" \
    'sampling.subsets=[code-search-net--code_search_net]' \
    'sampling.latents=[biojava/biojava]' \
    name=code_search_net_trajectory

  "$VENV_PYTHON" "$SCRIPT" \
    checkpoint_url="$CHECKPOINT" \
    checkpoint_step="$STEP" \
    base_lr="$BASE_LR" \
    adaptation.aux_loss_weight="$AUX_WEIGHT" \
    'sampling.subsets=[Lyun0912--LongABC]' \
    'sampling.latents=[rpj_book_0150312]' \
    name=LongABC_book_trajectory

  "$VENV_PYTHON" "$SCRIPT" \
    checkpoint_url="$CHECKPOINT" \
    checkpoint_step="$STEP" \
    base_lr="$BASE_LR" \
    adaptation.aux_loss_weight="$AUX_WEIGHT" \
    'sampling.subsets=[PleIAs--SYNTH]' \
    'sampling.latents=[https://en.wikipedia.org/wiki/Optical_telescope]' \
    name=SYNTH_telescope_trajectory

  "$VENV_PYTHON" "$SCRIPT" \
    checkpoint_url="$CHECKPOINT" \
    checkpoint_step="$STEP" \
    base_lr="$BASE_LR" \
    adaptation.aux_loss_weight="$AUX_WEIGHT" \
    sampling.mode=episodes \
    sampling.n=2 \
    name=two_episodes_per_subset
)
