
from tqdm import tqdm
import random

import datasets


def trajectify(
    ds: datasets.Dataset,
    horizon_length: int,
    latent_key: str = "latent",
    messages_key: str = "messages",
    episode_key: str = "episode"
) -> datasets.Dataset | None:
    """
    Convert a dataset of conversations into a dataset of trajectories.
    """
    assert horizon_length % 2 == 0, "Horizon length must be even."

    ds = ds.sort(latent_key)
    latents = ds[latent_key]

    trajectories = []
    pbar = tqdm(desc="Trajectifying", leave=False)

    # iterate over different types of latents
    latent_start = 0
    while latent_start < len(ds):
        latent = latents[latent_start]

        # all the examples for the current latent                                               
        latent_end = latent_start + 1
        while latent_end < len(ds) and latents[latent_end] == latent:
            latent_end += 1
        latent_num = latent_end - latent_start

        latent_indices = list(range(latent_start, latent_end))
        random.shuffle(latent_indices)

        # iterate over trajectories
        for curr_start in range(0, latent_num, horizon_length):
            curr_end = min(curr_start + horizon_length, latent_num)
            curr_num = curr_end - curr_start

            if curr_num < horizon_length // 2:
                break

            curr_indices = latent_indices[curr_start:curr_end]

            # duplicate examples if the trajectory is too short
            if curr_num < horizon_length:
                curr_indices += curr_indices[:horizon_length - curr_num]
                random.shuffle(curr_indices)

            curr = ds[curr_indices]

            # all non-message keys should be the same
            out = {}
            for key, values in curr.items():
                if key == messages_key:
                    continue
                unique_count = len(set(values))
                assert unique_count == 1, f"Expected only one unique value for key {key} in trajectory, but found {unique_count}."
                out[key] = values[0]

            # add the messages to the trajectory
            for episode_i in range(horizon_length):
                out[f"{episode_key}_{episode_i:02d}"] = curr[messages_key][episode_i]

            trajectories.append(out)

        latent_start = latent_end
        pbar.update(1)

    pbar.close()

    if len(trajectories) == 0:
        return None
    return datasets.Dataset.from_list(trajectories)
