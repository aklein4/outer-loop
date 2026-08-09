
import random
import numpy as np

from handlers import HANDLERS
from trajectory import trajectify


HORIZON_LENGTH = 64

DS_NAME = "aklein4/latent-compilation"
LOG_FILE = "compilation_log.txt"

NAMES_TO_DO = None
DEBUG = True

NUM_PROC = 8
BATCH_SIZE = 1024
MAX_COUNT = 10000


def main():
    
    random.seed(42)
    np.random.seed(42)

    with open(LOG_FILE, "w") as f:
        f.write("")

    # select the handlers to run
    handler_list = HANDLERS
    if NAMES_TO_DO is not None:

        names = [h_type().source() for h_type in handler_list]
        for name in NAMES_TO_DO:
            if name not in names:
                raise ValueError(f"Dataset name {name} not found in handlers.")

        handler_list = [
            h_type for h_type in handler_list if h_type().source() in NAMES_TO_DO
        ]

    # iterate over handlers
    total_examples = 0
    for i, h_type in enumerate(handler_list):
        h = h_type()

        print("")
        print(f"[{i+1}/{len(handler_list)}] Processing dataset: {h.source()}")
        print("")

        try:

            ds, counts = h.process(
                max_count=MAX_COUNT,
                num_proc=NUM_PROC,
                batch_size=BATCH_SIZE
            )
            trajectory_ds = trajectify(ds, counts, HORIZON_LENGTH)
            
            trajectory_ds.push_to_hub(
                DS_NAME,
                config_name=h.source().replace("/", "--"),
                private=False,
                split="train",
            )

        except Exception as e:
            if isinstance(e, KeyboardInterrupt) or DEBUG:
                raise e

            with open(LOG_FILE, "a") as f:
                f.write(f"\n[{i+1}/{len(handler_list)}] {h.source()}: FAIL")
            continue

        with open(LOG_FILE, "a") as f:
            f.write(f"\n[{i+1}/{len(handler_list)}] {h.source()}: SUCCESS ({len(trajectory_ds):_} examples)")
        total_examples += len(ds)
    
    with open(LOG_FILE, "a") as f:
        f.write(f"\n\nTotal examples: {total_examples:_}\n")


if __name__ == "__main__":
    main()
