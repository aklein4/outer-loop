
import gc
import os
import random
import tempfile
import traceback
import numpy as np

from handlers import get_handlers
from trajectory import trajectify
from tokens import GigaChat


HORIZON_LENGTH = 64

DS_NAME = "aklein4/latent-compilation"
LOG_FILE = "compilation_log.txt"

NAMES_TO_DO = [
    "Lyun0912/LongABC",
    # "blitt/SPoRC",
    # "PleIAs/YouTube-Commons",
    # "theelderemo/genius-lyrics-cleaned",
    # "sxiong/DHSA_Long-Data-Collections",
    # "bigscience/P3",
    # "clips/mqa",
]
DEBUG = False

NUM_PROC = 48
BATCH_SIZE = 256
MAX_COUNT = None
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

        stage = "processing"
        ds = None
        trajectory_ds = None
        intermediate_cache = tempfile.TemporaryDirectory(
            prefix=f"latent-compile-{h.source().replace('/', '--')}-",
        )
        try:

            print(f"Processing {h.source()}...", flush=True)
            ds = h.process(
                tokenizer=tokenizer,
                max_count=MAX_COUNT,
                num_proc=NUM_PROC,
                batch_size=BATCH_SIZE,
                intermediate_cache_dir=intermediate_cache.name,
            )

            stage = "trajectifying"
            print(f"Trajectifying {h.source()}...", flush=True)
            trajectory_ds = trajectify(
                ds,
                HORIZON_LENGTH,
                num_proc=NUM_PROC,
                keep_in_memory=KEEP_IN_MEMORY,
                shuffle_episodes=h.shuffle_episodes,
                drop_incomplete=h.drop_incomplete_trajectories,
            )
            trajectory_count = len(trajectory_ds) if trajectory_ds is not None else 0

            if trajectory_ds is not None:

                stage = "shuffling"
                print(f"Shuffling {h.source()}...", flush=True)
                trajectory_ds = trajectory_ds.shuffle(
                    seed=random.randrange(2**31),
                    load_from_cache_file=False,
                    indices_cache_file_name=os.path.join(
                        intermediate_cache.name,
                        "trajectory-shuffle-indices.arrow",
                    ),
                )

                stage = "uploading"
                print(f"Uploading {h.source()}...", flush=True)
                trajectory_ds.push_to_hub(
                    DS_NAME,
                    config_name=h.source().replace("/", "--"),
                    private=False,
                    split="train",
                )

        except Exception as error:
            failure = (
                f"[{i+1}/{len(handler_list)}] {h.source()}: "
                f"FAIL during {stage}: {type(error).__name__}: {error}"
            )
            details = traceback.format_exc()
            print(f"{failure}\n{details}", flush=True)

            with open(LOG_FILE, "a") as f:
                f.write(f"\n{failure}\n{details}")

            if DEBUG:
                raise
            continue

        finally:
            # Every Arrow transform is routed here. Close its memory maps
            # before unlinking so disk space is reclaimed after each handler,
            # including when processing or upload raises.
            ds = None
            trajectory_ds = None
            gc.collect()
            intermediate_cache.cleanup()

        with open(LOG_FILE, "a") as f:
            f.write(f"\n[{i+1}/{len(handler_list)}] {h.source()}: SUCCESS ({trajectory_count:_} examples)")
        total_examples += trajectory_count
    
    with open(LOG_FILE, "a") as f:
        f.write(f"\n\nTotal examples: {total_examples:_}\n")


if __name__ == "__main__":
    main()
