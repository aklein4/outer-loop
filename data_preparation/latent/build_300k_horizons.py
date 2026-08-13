"""Build and upload the 300k-row latent-compilation mixture."""

import tempfile

import datasets


SOURCE_DATASET = "aklein4/latent-compilation"
DESTINATION_DATASET = "aklein4/300k-horizons"
SPLIT = "train"
SHUFFLE_SEED = 42

# Subset names use the source dataset's config-name convention: the `/` in
# the original dataset name is encoded as `--`.
SUBSET_COUNTS = {
    "PleIAs--SYNTH": 50_000,
    "sxiong--DHSA_Long-Data-Collections": 50_000,
    "Lyun0912--LongABC": 47_745,
    "HuggingFaceTB--smoltalk2": 20_000,
    "PaDaS-Lab--webfaq-v2": 20_000,
    "code-search-net--code_search_net": 15_000,
    "Agent-Ark--Toucan-1.5M": 10_740,
    "HuggingFaceTB--stackexchange_2025_md": 10_000,
    "LxYxvv--quora_qa_raw": 10_000,
    "recursal--Fanatic-Fandom": 10_000,
    "ray0rf1re--AO3-2020": 10_000,
    "barilan--blog_authorship_corpus": 7_636,
    "arranonymsub--HiCUPID": 7_000,
    "webis--tldr-17": 5_109,
    "natural-instructions": 5_000,
    "theelderemo--genius-lyrics-cleaned": 5_000,
    "nvidia--Nemotron-SFT-Agentic-v2": 4_311,
    "blitt--SPoRC": 4_175,
    "tasksource--tasksource-instruct-v0": 3_000,
    "bigcode--starcoder2data-extras": 2_000,
    "nvidia--Nemotron-Agentic-v1": 1_740,
    "corbt--enron-emails": 1_000,
    "YuanPJ--summ_screen": 408,
    "BEE-spoke-data--medium-articles-en": 136,
    "RyokoAI--Fandom23K": 0,
}


TOOL_CALL = {
    "type": datasets.Value("string"),
    "function": {
        "name": datasets.Value("string"),
        "arguments": datasets.Value("string"),
    },
}
MESSAGE = {
    "role": datasets.Value("string"),
    "content": datasets.Value("string"),
    "tool_calls": datasets.List(TOOL_CALL),
}
FEATURES = datasets.Features(
    {
        "latent": datasets.Value("string"),
        "source": datasets.Value("string"),
        "kind": datasets.Value("string"),
        **{f"episode_{index:02d}": datasets.List(MESSAGE) for index in range(64)},
    }
)


def normalize_row(row):
    """Give every config the union schema without discarding tool calls."""
    for index in range(64):
        for message in row[f"episode_{index:02d}"]:
            message.setdefault("tool_calls", None)
    return row


def selected_rows():
    """Stream only each requested prefix, as BaseHandler does for max_count."""
    for config_name, count in SUBSET_COUNTS.items():
        if count == 0:
            continue

        print(f"Loading {count:_} rows from {config_name}...", flush=True)
        source = datasets.load_dataset(
            SOURCE_DATASET,
            config_name,
            split=SPLIT,
            streaming=True,
        )
        loaded = 0
        for row in source.take(count):
            yield normalize_row(row)
            loaded += 1

        if loaded != count:
            raise RuntimeError(
                f"{config_name} supplied {loaded:_} rows; expected {count:_}"
            )


def main():
    expected_rows = sum(SUBSET_COUNTS.values())
    if expected_rows != 300_000:
        raise ValueError(f"Hard-coded counts total {expected_rows:_}, not 300,000")

    with tempfile.TemporaryDirectory(prefix="build-300k-horizons-") as cache_dir:
        dataset = datasets.Dataset.from_generator(
            selected_rows,
            features=FEATURES,
            cache_dir=cache_dir,
        )
        if len(dataset) != expected_rows:
            raise RuntimeError(
                f"Materialized {len(dataset):_} rows; expected {expected_rows:_}"
            )

        print(f"Globally shuffling {len(dataset):_} rows...", flush=True)
        dataset = dataset.shuffle(
            seed=SHUFFLE_SEED,
            load_from_cache_file=False,
            indices_cache_file_name=f"{cache_dir}/shuffle-indices.arrow",
        )

        print(f"Uploading to {DESTINATION_DATASET}...", flush=True)
        result = dataset.push_to_hub(
            DESTINATION_DATASET,
            split=SPLIT,
            private=False,
        )
        print(result, flush=True)


if __name__ == "__main__":
    main()
