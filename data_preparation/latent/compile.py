
import random
import numpy as np

from handlers import get_handlers
from trajectory import trajectify
from tokens import GigaChat


HORIZON_LENGTH = 64

DS_NAME = "aklein4/latent-compilation"
LOG_FILE = "compilation_log.txt"

NAMES_TO_DO = None
DEBUG = True

NUM_PROC = 8
BATCH_SIZE = 1024
MAX_COUNT = 10000
KEEP_IN_MEMORY = True

TOKENIZER_URL = "meta-llama/Llama-3.2-1B-Instruct"
MAX_SEQUENCE_LENGTH = 1024


def main():

    with open(LOG_FILE, "w") as f:
        f.write("")

    # select the handlers to run
    handler_list = get_handlers(NAMES_TO_DO)

    # load the tokenizer
    tokenizer = GigaChat(
        tokenizer_url=TOKENIZER_URL,
        max_length=MAX_SEQUENCE_LENGTH,
    )

    # iterate over handlers
    total_examples = 0
    for i, h_type in enumerate(handler_list):
        h = h_type()

        print("")
        print(f"[{i+1}/{len(handler_list)}] Processing dataset: {h.source()}")
        print("")

        random.seed(42)
        np.random.seed(42)

        try:

            ds = h.process(
                tokenizer=tokenizer,
                max_count=MAX_COUNT,
                num_proc=NUM_PROC,
                batch_size=BATCH_SIZE
            )

            trajectory_ds = trajectify(
                ds,
                HORIZON_LENGTH,
                num_proc=NUM_PROC,
                keep_in_memory=KEEP_IN_MEMORY,
            )
            trajectory_count = len(trajectory_ds) if trajectory_ds is not None else 0

            if trajectory_ds is not None:

                trajectory_ds = trajectory_ds.shuffle(
                    seed=random.randrange(2**31),
                    load_from_cache_file=False
                )
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
            f.write(f"\n[{i+1}/{len(handler_list)}] {h.source()}: SUCCESS ({trajectory_count:_} examples)")
        total_examples += trajectory_count
    
    with open(LOG_FILE, "a") as f:
        f.write(f"\n\nTotal examples: {total_examples:_}\n")


if __name__ == "__main__":
    main()
