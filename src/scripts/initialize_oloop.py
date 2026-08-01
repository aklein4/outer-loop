"""Activation-initialize an OLoop model and upload it as a checkpoint."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import datasets
import huggingface_hub as hf
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from collators.horizon import HorizonCollator
from models import load_checkpoint_state
from utils.import_utils import import_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--repo", default="aklein4/oloop-init")
    parser.add_argument("--step", type=int, default=0)
    args, overrides = parser.parse_known_args()

    with initialize_config_dir(version_base=None, config_dir=str(SRC / "configs")):
        config = compose(config_name="default", overrides=overrides)

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    attention_kernel = config.model.attention_kernel
    config.model.attention_kernel = None
    model = import_model(config.model.type)(
        config.model
    )
    if config.model.pretrained_url is not None:
        load_checkpoint_state(
            model,
            config.model.pretrained_url,
            config.model.pretrained_step,
            strict=config.model.pretrained_strict,
        )
    model = model.float().to(DEVICE).eval()

    config.data.collator.kwargs["cluster_length"] = 1
    collator = HorizonCollator(**config.data.collator.kwargs)

    data = datasets.load_dataset(config.data.dataset.url, **config.data.dataset.kwargs)
    rows = list(data.take(args.episodes))
    batch = collator(rows)

    input_ids = batch["input_ids"][:, 0].to(DEVICE)
    init_mask = batch["attention_mask"][:, 0].to(DEVICE)

    model.enable_init(init_mask)
    with torch.no_grad():
        model(input_ids)
    model.disable_init()
    config.model.attention_kernel = attention_kernel

    hf.create_repo(args.repo, repo_type="model", exist_ok=True)
    with tempfile.TemporaryDirectory() as save_dir:
        with open(Path(save_dir) / "config.json", "w") as f:
            json.dump(OmegaConf.to_container(config.model, resolve=True), f, indent=4)
        torch.save(
            {name: value.detach().cpu() for name, value in model.state_dict().items()},
            Path(save_dir) / "model.pt",
        )
        hf.HfApi().upload_folder(
            repo_id=args.repo,
            repo_type="model",
            folder_path=save_dir,
            path_in_repo=f"{args.step:012d}",
        )

    print(f"Uploaded {args.episodes} episode initialization to {args.repo} step {args.step}")


if __name__ == "__main__":
    main()
