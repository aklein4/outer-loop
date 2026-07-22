
import pandas as pd
import matplotlib.pyplot as plt

import wandb
api = wandb.Api()
run = api.run("/aklein4/iTTT-Cluster/runs/0lkhuhp9") # recurrent-rank
run_2 = api.run("/aklein4/iTTT-Cluster/runs/fqigzd96") # recurrent-v2
# run_2 = api.run("/aklein4/iTTT-Cluster/runs/ymwimx1n") # alpha
# run_2 = api.run("/aklein4/iTTT-Cluster/runs/6sdfq3j4") # aux

RUN_NAME = "recurrent-rank"
RUN_2_NAME = "recurrent-v2"

RUNS = [run, run_2]
RUN_LABELS = [RUN_NAME, RUN_2_NAME]
RUN_COLORS = ["blue", "red"]

EPISODE_PREFIX = "lm_loss/episode_"
OVERALL_LOSS_KEY = "overall_lm_loss"
DECADE_PREFIX = "grouped_lm_loss/decade_"

ROLLING_WINDOW = 200
ROLLING_MIN = 100

OUTPUT_PATH = "compare_losses.png"


def smooth(series: pd.Series) -> pd.Series:
    return series.rolling(ROLLING_WINDOW, min_periods=ROLLING_MIN, center=False).mean()


def load_df(r, keys):
    rows = list(r.scan_history(keys=keys))
    df = pd.DataFrame(rows)
    df[keys] = df[keys].apply(pd.to_numeric, errors="coerce")
    return df


def main():
    # find which episodes and decades were logged (from the first run)
    sample_cols = run.history(samples=1).columns
    episodes = sorted(
        int(c[len(EPISODE_PREFIX):]) for c in sample_cols if c.startswith(EPISODE_PREFIX)
    )
    decades = sorted(
        int(c[len(DECADE_PREFIX):]) for c in sample_cols if c.startswith(DECADE_PREFIX)
    )
    episode_keys = [f"{EPISODE_PREFIX}{e:02d}" for e in episodes]
    decade_keys = [f"{DECADE_PREFIX}{d}" for d in decades]
    keys = episode_keys + decade_keys

    # pull full history for both runs
    dfs = [load_df(r, keys) for r in RUNS]
    for df in dfs:
        df[OVERALL_LOSS_KEY] = df[episode_keys].mean(axis=1)

    # one line per subplot: overall loss followed by each decade
    lines = []  # (key, title)
    lines.append((OVERALL_LOSS_KEY, "loss"))
    for d in decades:
        min_episode = max(1, 10 * d)
        max_episode = min(64 - 1, 10 * (d + 1))
        lines.append(
            (f"{DECADE_PREFIX}{d}", f"episodes {min_episode:02d}-{max_episode:02d}")
        )

    ncols = 4
    nrows = (len(lines) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, lines):
        for df, run_label, run_color in zip(dfs, RUN_LABELS, RUN_COLORS):
            ax.plot(
                smooth(df[key]),
                color=run_color,
                linewidth=1.5,
                label=run_label,
            )
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid()
        ax.legend()

    # hide any unused axes
    for ax in axes[len(lines):]:
        ax.axis("off")

    fig.suptitle(f"LM loss (rolling window={ROLLING_WINDOW}, min={ROLLING_MIN})")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
