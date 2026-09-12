"""Memory-conscious sliding-window inference for 3D segmentation models."""

import itertools
import tempfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def extract_logits(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if isinstance(output, dict):
        for key in ("logits", "main_logits", "logits_main", "prediction"):
            if torch.is_tensor(output.get(key)):
                return output[key]
    raise TypeError("Model must return logits shaped [B, C, D, H, W]")


def _scan_starts(image_size, patch_size, overlap):
    starts_per_axis = []
    for image_dim, patch_dim in zip(image_size, patch_size):
        if image_dim == patch_dim:
            starts_per_axis.append([0])
            continue
        step = max(int(round(patch_dim * (1.0 - overlap))), 1)
        starts = list(range(0, image_dim - patch_dim + 1, step))
        final = image_dim - patch_dim
        if starts[-1] != final:
            starts.append(final)
        starts_per_axis.append(starts)
    return list(itertools.product(*starts_per_axis))


def _gaussian(patch_size, sigma_scale):
    squared_axes = []
    for size in patch_size:
        coordinate = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
        squared_axes.append((coordinate / max(size * sigma_scale, 1e-6)) ** 2)
    result = np.exp(
        -0.5
        * (
            squared_axes[0][:, None, None]
            + squared_axes[1][None, :, None]
            + squared_axes[2][None, None, :]
        )
    ).astype(np.float32)
    result /= result.max()
    return np.maximum(result, 1e-4)


def _allocate(score_shape, spatial_shape, max_ram_mb, temp_dir):
    required = (int(np.prod(score_shape)) + int(np.prod(spatial_shape))) * 4
    if required <= int(max_ram_mb * 1024**2):
        return (
            np.zeros(score_shape, dtype=np.float32),
            np.zeros(spatial_shape, dtype=np.float32),
            None,
        )
    temporary = tempfile.TemporaryDirectory(prefix="pce_val_", dir=temp_dir)
    scores = np.memmap(
        Path(temporary.name) / "scores.dat",
        mode="w+",
        dtype=np.float32,
        shape=score_shape,
    )
    counts = np.memmap(
        Path(temporary.name) / "counts.dat",
        mode="w+",
        dtype=np.float32,
        shape=spatial_shape,
    )
    scores[:] = 0
    counts[:] = 0
    return scores, counts, temporary


@torch.inference_mode()
def sliding_window_predict(
    model,
    image,
    num_classes,
    patch_size,
    device,
    overlap=0.5,
    sw_batch_size=1,
    use_amp=False,
    max_accumulator_mb=1024,
    temp_dir=None,
):
    """Predict one ``[1, 1, D, H, W]`` volume and return a DHW NumPy mask."""
    patch_size = tuple(int(value) for value in patch_size)
    if image.ndim != 5 or image.shape[0] != 1:
        raise ValueError("image must have shape [1, C, D, H, W]")
    if len(patch_size) != 3 or any(value <= 0 for value in patch_size):
        raise ValueError("patch_size must contain three positive values")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")
    if sw_batch_size < 1:
        raise ValueError("sw_batch_size must be positive")

    original_shape = tuple(int(value) for value in image.shape[2:])
    axis_padding = []
    for current, target in zip(original_shape, patch_size):
        total = max(target - current, 0)
        axis_padding.append((total // 2, total - total // 2))
    d_pad, h_pad, w_pad = axis_padding
    image = F.pad(
        image.cpu().float(),
        (w_pad[0], w_pad[1], h_pad[0], h_pad[1], d_pad[0], d_pad[1]),
    )
    spatial_shape = tuple(int(value) for value in image.shape[2:])
    starts = _scan_starts(spatial_shape, patch_size, overlap)
    importance = _gaussian(patch_size, sigma_scale=0.125)
    scores, counts, temporary = _allocate(
        (num_classes,) + spatial_shape,
        spatial_shape,
        max_accumulator_mb,
        temp_dir,
    )
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and device.type == "cuda"
        else nullcontext()
    )

    try:
        for offset in range(0, len(starts), sw_batch_size):
            batch_starts = starts[offset : offset + sw_batch_size]
            patches = torch.stack(
                [
                    image[
                        0,
                        :,
                        d : d + patch_size[0],
                        h : h + patch_size[1],
                        w : w + patch_size[2],
                    ]
                    for d, h, w in batch_starts
                ]
            ).to(device, non_blocking=True)
            with amp_context:
                logits = extract_logits(model(patches))
            if tuple(logits.shape) != (
                len(batch_starts),
                num_classes,
                *patch_size,
            ):
                raise ValueError("Unexpected model output shape: {}".format(tuple(logits.shape)))
            logits = logits.float().cpu().numpy()
            for batch_index, (d, h, w) in enumerate(batch_starts):
                region = (
                    slice(d, d + patch_size[0]),
                    slice(h, h + patch_size[1]),
                    slice(w, w + patch_size[2]),
                )
                scores[(slice(None),) + region] += logits[batch_index] * importance[None]
                counts[region] += importance

        prediction = np.empty(spatial_shape, dtype=np.uint8)
        for start in range(0, spatial_shape[0], 16):
            end = min(start + 16, spatial_shape[0])
            prediction[start:end] = np.argmax(
                np.asarray(scores[:, start:end]) / np.asarray(counts[start:end])[None],
                axis=0,
            )
        crop = tuple(
            slice(before, before + size)
            for (before, _), size in zip(axis_padding, original_shape)
        )
        return np.ascontiguousarray(prediction[crop])
    finally:
        del scores, counts
        if temporary is not None:
            temporary.cleanup()
