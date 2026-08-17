#!/usr/bin/env bash

(
  set -euo pipefail

  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  cd "$SCRIPT_DIR"

  # Set sampling.n for random selection; sampling.latents selects exact full trajectories.
  ~/iTTT/src/.venv/bin/python scripts/visualize_piano_token_gates.py \
    'sampling.subsets=[PleIAs--SYNTH]' \
    'sampling.latents=[https://en.wikipedia.org/wiki/Optical_telescope]' \
    name=SYNTH_telescope_trajectory_new
)
