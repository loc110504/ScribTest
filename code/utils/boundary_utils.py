import torch
import torch.nn.functional as F


def _normalize_per_image(x, eps=1e-8):
    b = x.shape[0]
    flat = x.flatten(1)
    minv = flat.min(dim=1)[0].view(b, 1, 1, 1)
    maxv = flat.max(dim=1)[0].view(b, 1, 1, 1)
    return ((x - minv) / (maxv - minv + eps)).clamp(0.0, 1.0)


def sobel_magnitude(x, normalize=True, eps=1e-8):
    if x.dim() != 4:
        raise ValueError(f"x must be [B,C,H,W], got {x.shape}")
    if x.size(1) != 1:
        x = x.mean(dim=1, keepdim=True)

    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + eps)
    if normalize:
        mag = _normalize_per_image(mag, eps=eps)
    return mag


def boundary_likelihood(image, probs_s=None, lambda_image=1.0, eps=1e-8):
    b_img = sobel_magnitude(image, normalize=True, eps=eps)
    if probs_s is None or lambda_image >= 1.0:
        return b_img

    conf = probs_s.max(dim=1, keepdim=True)[0]
    b_prob = sobel_magnitude(conf, normalize=True, eps=eps)
    b = lambda_image * b_img + (1.0 - lambda_image) * b_prob
    return b.clamp(0.0, 1.0)
