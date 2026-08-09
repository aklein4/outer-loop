
from tqdm import tqdm
import random

import datasets


def list_range(*args):
    return list(range(*args))


def trajectify(
    ds: datasets.Dataset,
    counts: dict,
    horizon_length: int,
    latent_key: str = "latent",
    messages_key: str = "messages",
    episode_key: str = "episode"
) -> datasets.Dataset:
    """
    Convert a dataset of conversations into a dataset of trajectories.
    """
    assert horizon_length % 2 == 0, "Horizon length must be even."

    pbar = tqdm(total=len(counts), desc="Trajectifying", leave=False)

    ds = ds.sort(latent_key)

    trajectories = []

    # iterate over different types of latents
    i = 0
    while i < len(ds):

        # all the examples for the current latent
        latent = ds[latent_key][i]
        latent_num = counts[latent]

        inds = list_range(i, i + latent_num)
        all_latent = ds.select(inds, keep_in_memory=True)
        all_latent = all_latent.shuffle(
            seed=random.randrange(int(2**31)),
            keep_in_memory=True
        )

        # iterate over trajectories
        for start in range(0, latent_num, horizon_length):
            end = min(start + horizon_length, latent_num)
            curr_len = end - start

            if curr_len < horizon_length // 2:
                break

            curr = all_latent.select(
                range(start, end), keep_in_memory=True
            )

            # duplicate examples if the trajectory is too short
            if curr_len < horizon_length:
                curr = datasets.concatenate_datasets([
                    curr, curr.select(range(horizon_length - curr_len), keep_in_memory=True)
                ])
                curr = curr.shuffle(
                    seed=random.randrange(int(2**31)),
                    keep_in_memory=True
                )

            # all non-message keys should be the same
            out = {}
            for key in curr.column_names:
                if key == messages_key:
                    continue
                assert len(set(curr[key])) == 1, f"Expected only one unique value for key {key} in trajectory, but found {len(set(curr[key]))}."
                out[key] = curr[key][0]

            # add the messages to the trajectory
            for i in range(horizon_length):
                out[f"{episode_key}_{i:02d}"] = curr[messages_key][i]

            trajectories.append(out)

        i = max(inds) + 1
        pbar.update(1)

    pbar.close()

    return datasets.Dataset.from_list(trajectories)
