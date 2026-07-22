import torch

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from omegaconf import DictConfig

import datasets

from models import load_checkpoint_state
from collators.simple import SimpleCollator
import utils.constants as constants
from utils.loss_utils import lm_loss_fn
from utils.import_utils import import_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NAME = "hyper-trained-test" 

MODEL_TYPE = "oloop.OLoopModel"
CONFIG_PATH = "configs/model/oloop-llama3p2-1b.yaml"

CHECKPOINT_URL = 'aklein4/outer-loop_oloop-hyper' # 'aklein4/Llama-3.2-1B-TPU'
CHECKPOINT_STEP = 500

MODEL_KWARGS = {
    "attention_kernel": "gpu_flash_attention",
    "chunk_size": 1024,
    # "momentum_beta": 0.75,
    # "base_lr": 1e-3,
}

DATA_URL = "aklein4/longattn-Llama3-32K"

NUM_EXAMPLES = 128
BS = 1


def main():
    
    print("Starting...")
    print(f"Using device: {DEVICE}")

    data = datasets.load_dataset(DATA_URL, split='train', streaming=True)
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=BS,
        collate_fn=SimpleCollator()
    )
    print("Data loaded.")

    with open(CONFIG_PATH, "r") as f:
        config_dict = yaml.safe_load(f)
    config_dict.update(MODEL_KWARGS)
    config = DictConfig(config_dict)

    model = import_model(MODEL_TYPE)(config).to(DEVICE)
    print("Model initialized.")
    model = load_checkpoint_state(
        model, CHECKPOINT_URL, CHECKPOINT_STEP, strict=False
    )
    print("Checkpoint loaded.")

    print("Running...")
    losses = []
    for i, batch in tqdm(enumerate(loader), total=NUM_EXAMPLES // BS):

        input_ids = batch["input_ids"].to(DEVICE)

        with torch.autocast("cuda", torch.bfloat16):
                
            if hasattr(model, "compute_logits"):
                logits = model.compute_logits(input_ids, verbose=True)
            else:
                with torch.no_grad():
                    logits = model(input_ids, shift_states=True)[0]
            
            with torch.no_grad():
                loss = lm_loss_fn(
                    logits, input_ids,
                    shift_logits=False,
                    shift_labels=True,
                    reduction='none',
                )

        losses.append(loss.detach().cpu())
        del logits

        if len(losses) >= NUM_EXAMPLES // BS:
            break

    losses = torch.cat(losses, dim=0)
    torch.save(
        losses,
        os.path.join(constants.LOCAL_DATA_PATH, f"{NAME}_losses.pt")
    )


def nan_mean(x):
    mask = np.isfinite(x)
    s = mask.sum(0)
    x = np.where(mask, x, np.zeros_like(x))
    return x.sum(0) / (s + 1e-7)


def analyze_results():

    def load_loss(name):
        return nan_mean(torch.load(os.path.join(constants.LOCAL_DATA_PATH, f"{name}_losses.pt")).float().numpy())

    df = pd.DataFrame({
        "baseline": load_loss("sliding-trained"),
        "no-meta": load_loss("sliding-no-meta"),
        "meta": load_loss("hyper-trained"),
        # "no-meta-1e-2": load_loss("sliding-no-meta-1e-2"),
    })

    # df = pd.DataFrame({
    #     "sliding": load_loss("sliding"),
    #     "llama": load_loss("llama"),
    #     "oloop": load_loss("oloop"),
    #     # "simple": load_loss("simple"),
    #     "alpha": load_loss("alpha"),
    #     # "lr=3e-3": load_loss("lr=3e-3"),
    #     # "no-svd": load_loss("no-svd"),
    #     # "lr=1e-2": load_loss("lr=1e-2"),
    #     # "hyper": load_loss("hyper"),
    #     "hyper": load_loss("hyper-init"),
    #     "hyper-trained": load_loss("hyper-trained"),
    #     "sliding-trained": load_loss("sliding-trained"),
    #     "sliding-no-meta": load_loss("sliding-no-meta"),
    # })

    print("\n === Average Losses === ")
    for col in df.columns:
        print(f"    {col}: {df[col].mean():.2f}")
    print("")

    for i, col in enumerate(df.columns):

        x = np.arange(len(df[col]))
        y_running = df[col].rolling(window=2000, min_periods=1500)
        
        plt.plot(x, y_running.mean(), label=col, color=(f"C{i-1}" if i > 0 else "black"))
    
    plt.legend()
    plt.grid() 

    plt.title("Loss by Token Position")
    plt.xlabel("Token Position")
    plt.ylabel("Loss (log perplexity)")

    plt.tight_layout()
    plt.savefig("oloop_comparison.png", dpi=300)

    plt.clf()

    for i, col in enumerate(df.columns):
        if col == "meta":
            continue

        x = np.arange(len(df[col]))
        y_running = df[col].rolling(window=5000, min_periods=1500)
        
        plt.plot(x, y_running.mean(), label=col, color=(f"C{i-1}" if i > 0 else "black"), linewidth=0.25)
    
    plt.legend()
    plt.grid() 

    plt.title("Loss by Token Position (Zoomed)")
    plt.xlabel("Token Position")
    plt.ylabel("Loss (log perplexity)")

    plt.ylim(1.415, 1.425)
    plt.xlim(15000, 20000)

    plt.tight_layout()
    plt.savefig("oloop_comparison_zoom.png", dpi=300)


if __name__ == "__main__":
    # main()
    analyze_results()
