import torch

import yaml
from omegaconf import DictConfig

from models import load_checkpoint_state
from models.oloop import OLoopModel

CONFIG_PATH = "configs/model/oloop-smollm2-360m.yaml"

URL = 'aklein4/SmolLM2-360M-TPU'


def main():

    with open(CONFIG_PATH, "r") as f:
        config_dict = yaml.safe_load(f)
    config = DictConfig(config_dict)

    model = OLoopModel(config)
    print("Model initialized!")

    load_checkpoint_state(
        model, URL, 0, strict=False
    )
    print("Checkpoint loaded!")

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):_}")


if __name__ == "__main__":
    main()
