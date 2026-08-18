#!/usr/bin/env bash

(
  set -euo pipefail

  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  cd "$SCRIPT_DIR"

  # Set sampling.n for random selection; sampling.latents selects exact full trajectories.
  # ~/iTTT/src/.venv/bin/python scripts/visualize_piano_token_gates.py \
  #   'sampling.subsets=[code-search-net--code_search_net]' \
  #   'sampling.latents=[biojava/biojava]' \
  #   name=code_search_net_trajectory

  ~/iTTT/src/.venv/bin/python scripts/visualize_piano_token_gates.py \
      sampling.mode=episodes \
      sampling.n=2 \
      name=three_episodes_per_subset

)
