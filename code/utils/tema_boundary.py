import torch
import torch.nn.functional as F


def sobel_magnitude(image, eps=1e-8):
    """
    image: [B,1,H,W] or [B,C,H,W]
    returns: [B,1,H,W] normalized per image
    """
    if image.dim() != 4:
        raise ValueError(f'Expected image shape [B,C,H,W], got {tuple(image.shape)}')

    if image.shape[1] > 1:
        image = image.mean(dim=1, keepdim=True)

    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=image.device,
        dtype=image.dtype,
    ).unsqueeze(0)
    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=image.device,
        dtype=image.dtype,
    ).unsqueeze(0)

    gx = F.conv2d(image, sobel_x, padding=1)
    gy = F.conv2d(image, sobel_y, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + eps)

    b = mag.shape[0]
    mag_flat = mag.view(b, -1)
    mag_min = mag_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    mag_max = mag_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    mag = (mag - mag_min) / (mag_max - mag_min + eps)
    return mag.clamp(0.0, 1.0)


def boundary_likelihood(image, probs_s=None, lambda_image=1.0, eps=1e-8):
    b_img = sobel_magnitude(image, eps=eps)
    if probs_s is None or lambda_image >= 1.0:
        return b_img
    conf = probs_s.max(dim=1, keepdim=True)[0]
    b_prob = sobel_magnitude(conf, eps=eps)
    return (lambda_image * b_img + (1.0 - lambda_image) * b_prob).clamp(0.0, 1.0)
