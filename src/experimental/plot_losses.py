
import pandas as pd
import matplotlib.pyplot as plt

import wandb
api = wandb.Api()
run = api.run("/aklein4/iTTT-Cluster/runs/3da68lel")
run_2 = api.run("/aklein4/iTTT-Cluster/runs/ymwimx1n")

EPISODE_KEY = "lm_loss/episode_00"
DECADE_PREFIX = "grouped_lm_loss/decade_"

ROLLING_WINDOW = 200
ROLLING_MIN = 50

OUTPUT_PATH = "losses.png"


def smooth(series):
    return series.rolling(ROLLING_WINDOW, min_periods=ROLLING_MIN).mean()


def main():
    # find which decades were logged
    sample_cols = run.history(samples=1).columns
    decades = sorted(
        int(c[len(DECADE_PREFIX):]) for c in sample_cols if c.startswith(DECADE_PREFIX)
    )
    decade_keys = [f"{DECADE_PREFIX}{d}" for d in decades]

    # pull full history for the keys we care about
    keys = [EPISODE_KEY] + decade_keys
    rows = list(run.scan_history(keys=keys))
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))

    # episode_00 in black
    ax.plot(smooth(df[EPISODE_KEY]), color="black", label="episode 00", linewidth=1.5)

    # decades as a color gradient
    cmap = plt.get_cmap("viridis", len(decades))
    for i, d in enumerate(decades):
        min_episode = max(1, 10 * d)
        max_episode = min(64-1, 10 * (d + 1))
        ax.plot(
            smooth(df[f"{DECADE_PREFIX}{d}"]),
            color=cmap(len(decades)-i-1),
            label=f"episodes {min_episode:02d}-{max_episode:02d}",
            linewidth=1.5,
        )

    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"LM loss (rolling window={ROLLING_WINDOW}, min={ROLLING_MIN})")
    ax.legend()
    plt.grid()
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
