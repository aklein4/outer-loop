
import pandas as pd
import matplotlib.pyplot as plt

import wandb
api = wandb.Api()
run = api.run("/aklein4/iTTT-Cluster/runs/3da68lel") # no muon
# run = api.run("/aklein4/iTTT-Cluster/runs/ymwimx1n") # muon

EPISODE_PREFIX = "lm_loss/episode_"

# (label, color, [start_step, end_step)) -- end is exclusive
STEP_RANGES = [
    ("meta-training steps 0-49", "blue", (0, 49)),
    ("meta-training steps 1400-1599", "red", (1400, 1599)),
]

OUTPUT_PATH = "horizon_curves.png"


def main():
    # find which episodes were logged
    sample_cols = run.history(samples=1).columns
    episodes = sorted(
        int(c[len(EPISODE_PREFIX):]) for c in sample_cols if c.startswith(EPISODE_PREFIX)
    )
    episode_keys = [f"{EPISODE_PREFIX}{e:02d}" for e in episodes]

    # pull full history (with step) for every episode
    rows = list(run.scan_history(keys=["_step"] + episode_keys))
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, color, (start, end) in STEP_RANGES:
        window = df[(df["_step"] >= start) & (df["_step"] <= end)]
        # average each episode's loss across the steps in the window
        mean_loss = [window[k].mean() for k in episode_keys]
        ax.plot(episodes, mean_loss, color=color, label=label, linewidth=1.5)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs episode position for different meta-training step ranges")
    ax.legend()
    plt.grid()
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
