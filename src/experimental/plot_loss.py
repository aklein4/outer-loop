import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


FILE = "losses.csv"

RUNS = {
    "oloop-alpha": "LoRA",
    "oloop-hyper": "Hyper"
}

KEY = "grouped_lm_loss/decade_03"


def main():
    
    df = pd.read_csv(FILE)

    lines = {}
    for name, label in RUNS.items():

        data = list(df[f"{name} - {KEY}"].dropna())

        roll = pd.DataFrame({"k": data}).rolling(200, min_periods=25).mean()
        lines[label] = roll

        plt.plot(roll, label=label)
    
    plt.xlabel("Training Step")
    plt.ylabel(KEY)

    plt.legend()
    plt.grid()

    plt.title("Training Progress")

    plt.tight_layout()
    plt.savefig("training_loss.png")

    plt.clf()
    plt.plot(
        lines["Hyper"] - lines["LoRA"]
    )

    plt.xlabel("Training Step")
    plt.ylabel("Hyper - LoRA Loss")

    plt.grid()
    
    plt.title("Relative Loss")

    plt.tight_layout()
    plt.savefig("relative_loss.png")
    

if __name__ == "__main__":
    main()
