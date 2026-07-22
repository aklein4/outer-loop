
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import wandb


RUN_ID = "/aklein4/iTTT-Cluster/runs/el8wq588"

EPISODE_PREFIX = "lm_loss/episode_"

NUM_RANGES = 7

RANGE_WIDTH = 400
EPISODE_ROLLING_WINDOW = 3

COLORMAP = "viridis_r"

OUTPUT_PATH = "horizon_curves.png"


def get_step_ranges(steps):
    min_step = int(steps.min())
    max_step = int(steps.max())
    latest_start = max_step - RANGE_WIDTH + 1
    if latest_start < min_step:
        raise ValueError(
            f"Need at least {RANGE_WIDTH} training steps, found {min_step}-{max_step}."
        )

    starts = np.linspace(min_step, latest_start, NUM_RANGES)
    starts = np.rint(starts).astype(int)
    return [(start, start + RANGE_WIDTH) for start in starts]


def main():
    run = wandb.Api().run(RUN_ID)

    # find which episodes were logged
    sample_cols = run.history(samples=1).columns
    episodes = sorted(
        int(c[len(EPISODE_PREFIX):]) for c in sample_cols if c.startswith(EPISODE_PREFIX)
    )
    episode_keys = [f"{EPISODE_PREFIX}{e:02d}" for e in episodes]

    # pull full history (with step) for every episode
    rows = list(run.scan_history(keys=["_step"] + episode_keys))
    df = pd.DataFrame(rows)
    df["_step"] = pd.to_numeric(df["_step"], errors="coerce")
    df[episode_keys] = df[episode_keys].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["_step"])

    fig, ax = plt.subplots(figsize=(10, 6))

    step_ranges = get_step_ranges(df["_step"])
    norm = Normalize(
        vmin=min(start for start, _ in step_ranges),
        vmax=max(start for start, _ in step_ranges),
    )
    cmap = plt.get_cmap(COLORMAP)

    for start, end in step_ranges:
        window = df[(df["_step"] >= start) & (df["_step"] < end)]
        # average each episode's loss across the steps in the window
        mean_loss = pd.Series([window[k].mean() for k in episode_keys])
        smoothed_loss = mean_loss.rolling(
            window=EPISODE_ROLLING_WINDOW,
            center=False,
            min_periods=1,
        ).mean()
        label = f"meta-training steps {start}-{end - 1}"
        ax.plot(episodes, smoothed_loss, color=cmap(norm(start)), label=label, linewidth=1.5)

    ax.set_xlabel("Task-Specific Examples Seen")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs task-specific examples for different amounts of meta-training")
    # ax.legend()
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Meta-training steps")
    plt.grid()
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
