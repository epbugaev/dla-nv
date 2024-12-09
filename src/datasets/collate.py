import torch
import torch.nn.functional as F


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}
    copy_keys = ["audio_path"]

    for copy_key in copy_keys:
        result_batch[copy_key] = [item[copy_key] for item in dataset_items]

    tensor_keys = [
        "audio",
        "spectrogram",
        "original_audio",
        "original_spectrogram",
    ]

    for k in tensor_keys:
        last_dim_mx = max([item[k].shape[-1] for item in dataset_items])

        # Pad all tensors so last dimension sizes match
        result_batch[k] = torch.concat(
            [
                F.pad(item[k], (0, last_dim_mx - item[k].shape[-1]), "constant", 0)
                for item in dataset_items
            ],
            axis=0,
        )
        result_batch[k + "_length"] = torch.tensor(
            [item[k].shape[-1] for item in dataset_items]
        )

    return result_batch