import random

import numpy as np
import torch
import torch.nn.functional as F

try:
    import pywt
except ImportError as exc:
    raise ImportError(
        "PyWavelets is required for XNetv2 wavelet preprocessing. Install `PyWavelets` first."
    ) from exc


def _safe_minmax_scale(array):
    array = np.asarray(array, dtype=np.float32)
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if max_value - min_value < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value) * 255.0


def _sample_mix_weight(value):
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return float(value[0])
        return random.uniform(float(value[0]), float(value[1]))
    return float(value)


def build_wavelet_views(image, wavelet_type="haar", alpha=(0.0, 0.4), beta=(0.0, 0.4)):
    image = np.asarray(image, dtype=np.float32)
    ll, (lh, hl, hh) = pywt.dwt2(image, wavelet_type, axes=(0, 1))

    ll = _safe_minmax_scale(ll)
    lh = _safe_minmax_scale(lh)
    hl = _safe_minmax_scale(hl)
    hh = _safe_minmax_scale(hh)

    high = _safe_minmax_scale(lh + hl + hh)
    low = _safe_minmax_scale(ll + _sample_mix_weight(alpha) * high)
    high = _safe_minmax_scale(high + _sample_mix_weight(beta) * ll)
    return low.astype(np.float32), high.astype(np.float32)


def build_wavelet_batch_from_tensor(image_tensor, wavelet_type="haar", alpha=(0.2, 0.2), beta=(0.2, 0.2)):
    if image_tensor.dim() != 4 or image_tensor.size(1) != 1:
        raise ValueError("Expected image tensor with shape [B, 1, H, W], got {}".format(tuple(image_tensor.shape)))

    device = image_tensor.device
    dtype = image_tensor.dtype
    target_size = image_tensor.shape[-2:]
    low_batch = []
    high_batch = []
    for image in image_tensor.detach().cpu().numpy():
        low, high = build_wavelet_views(image[0], wavelet_type=wavelet_type, alpha=alpha, beta=beta)
        low_batch.append(low)
        high_batch.append(high)

    low_tensor = torch.from_numpy(np.stack(low_batch, axis=0)).unsqueeze(1).to(device=device, dtype=dtype)
    high_tensor = torch.from_numpy(np.stack(high_batch, axis=0)).unsqueeze(1).to(device=device, dtype=dtype)
    if low_tensor.shape[-2:] != target_size:
        low_tensor = F.interpolate(low_tensor, size=target_size, mode="bilinear", align_corners=False)
    if high_tensor.shape[-2:] != target_size:
        high_tensor = F.interpolate(high_tensor, size=target_size, mode="bilinear", align_corners=False)
    return low_tensor, high_tensor
