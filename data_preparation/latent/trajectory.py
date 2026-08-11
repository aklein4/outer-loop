
from tqdm import tqdm
import random

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from datasets.fingerprint import generate_random_fingerprint


_TAKE_BATCH_SIZE = 1_000_000


def _take_chunked(column, indices, batch_size=_TAKE_BATCH_SIZE):
    """Gather a column without combining all selected values into one array.

    Arrow's ``take`` concatenates the selected chunks internally. Nested
    variable-width columns such as ``messages`` can exceed the 32-bit offset
    limit on full-sized datasets even though each input chunk is valid. Taking
    bounded slices and retaining their chunks avoids that concatenation.
    """
    if len(indices) == 0:
        return pa.chunked_array([], type=column.type)

    chunks = []
    chunk_offsets = np.cumsum(
        [0, *(len(chunk) for chunk in column.chunks)],
        dtype=np.int64,
    )
    start = 0
    while start < len(indices):
        current_batch_size = min(batch_size, len(indices) - start)
        values = indices.slice(start, current_batch_size).to_numpy(
            zero_copy_only=False,
        )
        chunk_ids = np.searchsorted(
            chunk_offsets[1:],
            values,
            side="right",
        )
        selected_arrays = []
        grouped_positions = []
        try:
            for chunk_id in np.unique(chunk_ids):
                positions = np.flatnonzero(chunk_ids == chunk_id)
                local_indices = pa.array(
                    values[positions] - chunk_offsets[chunk_id],
                    type=pa.int64(),
                )
                selected_arrays.append(
                    pc.take(column.chunk(chunk_id), local_indices)
                )
                grouped_positions.append(positions)

            combined = pa.concat_arrays(selected_arrays)
            inverse_order = np.argsort(
                np.concatenate(grouped_positions),
                kind="stable",
            )
            chunks.append(pc.take(combined, pa.array(inverse_order)))
        except pa.ArrowInvalid as error:
            if "offset overflow" not in str(error) or current_batch_size == 1:
                raise
            batch_size = max(1, current_batch_size // 2)
            continue

        start += current_batch_size

    return pa.chunked_array(chunks, type=column.type)


def trajectify(
    ds: datasets.Dataset,
    horizon_length: int,
    latent_key: str = "latent",
    messages_key: str = "messages",
    episode_key: str = "episode",
    num_proc: int | None = None,
    keep_in_memory: bool = False,
) -> datasets.Dataset | None:
    """
    Convert a dataset of conversations into a dataset of trajectories.
    """
    assert horizon_length % 2 == 0, "Horizon length must be even."

    if len(ds) == 0:
        return None

    # Trajectories only require equal latent values to be contiguous. Encode
    # them once and stably sort the integer codes; lexical string sorting is
    # needlessly expensive for tens of millions of rows.
    encoded_latents = pc.dictionary_encode(ds.data.column(latent_key))
    dictionary_size = max(
        (len(chunk.dictionary) for chunk in encoded_latents.chunks),
        default=0,
    )
    latent_codes = np.concatenate([
        pc.fill_null(chunk.indices, dictionary_size).to_numpy(
            zero_copy_only=False,
        )
        for chunk in encoded_latents.chunks
    ]).astype(np.int64, copy=False)
    sorted_to_source = np.argsort(latent_codes, kind="stable")
    sorted_codes = latent_codes[sorted_to_source]
    latent_ends = np.concatenate((
        np.flatnonzero(np.diff(sorted_codes)) + 1,
        np.array([len(ds)]),
    ))

    trajectory_indices = []
    pbar = tqdm(total=len(latent_ends), desc="Trajectifying", leave=False)

    # iterate over different types of latents
    latent_start = 0
    for latent_end in latent_ends:
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

            trajectory_indices.append(curr_indices)

        latent_start = latent_end
        pbar.update(1)

    pbar.close()

    if len(trajectory_indices) == 0:
        return None

    episode_indices = [
        pa.array(
            [
                sorted_to_source[indices[episode_i]]
                for indices in trajectory_indices
            ],
            type=pa.int64(),
        )
        for episode_i in range(horizon_length)
    ]
    first_indices = episode_indices[0]
    arrays = []
    names = []

    # Keep invariant metadata without converting the large message column to
    # Python objects. Validate it using Arrow-native comparisons.
    for key in ds.column_names:
        if key == messages_key:
            continue

        column = ds.data.column(key)
        first_values = pc.take(column, first_indices)
        if not pa.types.is_null(column.type):
            for indices in episode_indices[1:]:
                episode_values = pc.take(column, indices)
                matches = pc.or_(
                    pc.fill_null(pc.equal(episode_values, first_values), False),
                    pc.and_(pc.is_null(episode_values), pc.is_null(first_values)),
                )
                assert pc.all(matches).as_py(), (
                    f"Expected only one unique value for key {key} in trajectory."
                )

        arrays.append(first_values)
        names.append(key)

    # Gather each episode column directly in Arrow. The previous implementation
    # deserialized every trajectory batch into Python before rebuilding Arrow.
    messages = ds.data.column(messages_key)
    for episode_i, indices in enumerate(tqdm(
        episode_indices,
        desc="Materializing episodes",
        leave=False,
    )):
        arrays.append(_take_chunked(messages, indices))
        names.append(f"{episode_key}_{episode_i:02d}")

    return datasets.Dataset(
        pa.Table.from_arrays(arrays, names=names),
        # Hashing this in-memory table makes ``datasets`` serialize every
        # nested message buffer and can consume hundreds of GiB. This output is
        # freshly generated and immediately shuffled with caching disabled, so
        # a unique fingerprint is sufficient.
        fingerprint=generate_random_fingerprint(),
    )
