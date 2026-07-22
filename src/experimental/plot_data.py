
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE = "training_data.csv"

RUNS = [
    "longattn-precond",
    "longattn-oloop",
    "longattn-seperated",
]


def main():
    
    df = pd.read_csv(FILE)

    d = {}
    for run in RUNS:
        
        x = df[f"{run} - grouped_lm_loss/decade_6"]

        x_running = x.rolling(window=100)
        d[run] = x_running.mean()

        plt.plot(x_running.mean()[:1000], label=run)

    plt.legend()
    plt.grid()
    plt.savefig("data.png")


if __name__ == "__main__":
    main()