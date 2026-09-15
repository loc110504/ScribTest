"""
Scribble-Calibrated Mean Teacher for Scribble-Supervised Medical Image Segmentation.

Core method:
    1. partial cross-entropy on the available scribble labels;
    2. an EMA teacher provides stable weak-view soft targets;
    3. pseudo-label reliability combines teacher confidence, normalized
       student-teacher Jensen-Shannon agreement, and teacher certainty;
    4. a small rotating hold-out of the *existing* scribble pixels is used only
       to calibrate class-wise reliability thresholds online;
    5. only unlabeled pixels whose reliability exceeds the calibrated threshold
       of the teacher-predicted class receive the soft teacher target.

No new scribble style or synthetic scribble label is generated.  The calibration
hold-out is a temporary partition of the original sparse supervision; random
rotation across iterations lets all available scribble pixels participate in
supervised learning over training while reducing same-step calibration leakage.

The optional weak/strong image split from the supplied baseline is retained.
Only intensity-space perturbations are used for the student's strong view, so
student predictions, teacher predictions, and scribble labels remain pixel aligned.
"""
import argparse
import math
import logging
import os
import random
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader.acdc import ACDCDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import ramps
from utils.ema_optim import WeightEMA
from val import test_single_volume


parser = argparse.ArgumentParser()

# =========================
# Basic training arguments
# =========================
parser.add_argument('--root_path', type=str, default='../../data/ACDC', help='dataset root')
parser.add_argument('--exp', type=str, default='ScribbleCalibrated_MT', help='experiment name')
parser.add_argument('--data', type=str, default='ACDC', help='dataset name')
parser.add_argument('--fold', type=str, default='MAAGfold70', help='dataset fold')
parser.add_argument('--sup_type', type=str, default='scribble', help='supervision type')
parser.add_argument('--model', type=str, default='unet_hl', help='network name')
parser.add_argument('--num_classes', type=int, default=4, help='number of segmentation classes')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum training iterations')
parser.add_argument('--batch_size', type=int, default=8, help='batch size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation learning rate')
parser.add_argument('--patch_size', nargs=2, type=int, default=[256, 256], help='network input patch size')
parser.add_argument('--seed', type=int, default=2022, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

# =========================
# EMA teacher (Eq. 15)
# =========================
parser.add_argument('--ema_decay', type=float, default=0.99, help='EMA decay rate alpha for the teacher network')

# =========================
# Pseudo-label loss ramp-up (Eq. 13, 14)
# =========================
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='pseudo-loss ramp-up epoch length')
parser.add_argument('--pseudo_loss_weight', type=float, default=8.0, help='lambda_max: max weight for the pseudo-label loss')
parser.add_argument('--pseudo_mask_mode', type=str, default='unlabeled',
                    choices=['unlabeled'],
                    help='pseudo-label supervision is applied only to originally unlabeled pixels')

# =========================
# Scribble-grounded reliability calibration
# =========================
parser.add_argument('--calibration_holdout', type=float, default=0.20,
                    help='fraction of existing scribble pixels temporarily held out per class/sample for calibration')
parser.add_argument('--calibration_target_precision', type=float, default=0.95,
                    help='minimum empirical precision required for accepted calibration predictions')
parser.add_argument('--calibration_bins', type=int, default=100,
                    help='number of bins for the online class-wise reliability histogram')
parser.add_argument('--calibration_min_samples', type=float, default=32.0,
                    help='minimum effective calibration samples before a class receives an adaptive threshold')
parser.add_argument('--calibration_ema_decay', type=float, default=0.99,
                    help='EMA decay for online calibration histograms; recent teacher behavior is emphasized')
parser.add_argument('--calibration_fallback_threshold', type=float, default=1.01,
                    help='threshold used until a class has enough calibration evidence; >1 disables pseudo labels safely')

# =========================
# Weak(teacher)/strong(student) augmentation split inherited from the baseline.
# Intensity-only perturbations preserve pixel alignment among the teacher view,
# student view, and scribble labels, which the reliability score requires.
# =========================
parser.add_argument('--use_strong_aug', type=int, default=1, choices=[0, 1],
                    help='feed the teacher a clean (weak) view and the student an intensity-perturbed '
                         '(strong) view of the same image, instead of an identical input to both')
parser.add_argument('--strong_brightness', type=float, default=0.2,
                    help='max relative brightness jitter for the student strong view')
parser.add_argument('--strong_brightness_prob', type=float, default=0.5,
                    help='per-sample probability of applying brightness jitter (0 disables it)')
parser.add_argument('--strong_contrast', type=float, default=0.2,
                    help='max relative contrast jitter for the student strong view')
parser.add_argument('--strong_contrast_prob', type=float, default=0.5,
                    help='per-sample probability of applying contrast jitter (0 disables it)')
parser.add_argument('--strong_gamma', type=float, default=0.3,
                    help='max relative gamma jitter for the student strong view')
parser.add_argument('--strong_gamma_prob', type=float, default=0.3,
                    help='per-sample probability of applying gamma jitter (0 disables it)')
parser.add_argument('--strong_noise_std', type=float, default=0.05,
                    help='max Gaussian noise std, as a fraction of each sample\'s own std, for the student strong view')
parser.add_argument('--strong_noise_prob', type=float, default=0.5,
                    help='per-sample probability of applying Gaussian noise (0 disables it)')
parser.add_argument('--strong_blur_prob', type=float, default=0.3,
                    help='probability of applying Gaussian blur, drawn ONCE per iteration and shared '
                         'across the whole batch (not per-sample, to keep a single conv call)')
parser.add_argument('--strong_blur_sigma_min', type=float, default=0.2,
                    help='minimum Gaussian blur sigma, in pixels')
parser.add_argument('--strong_blur_sigma_max', type=float, default=1.0,
                    help='maximum Gaussian blur sigma, in pixels')
parser.add_argument('--strong_cutout_prob', type=float, default=0.0,
                    help='per-sample probability of CutOut on the student strong view. '
                         'Off by default: scribble labels are sparse, so occluding a region '
                         'risks landing on and confusing a labeled scribble pixel.')
parser.add_argument('--strong_cutout_max_area', type=float, default=0.05,
                    help='maximum CutOut hole area, as a fraction of the image area')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def get_current_consistency_weight(epoch, train_args):
    """lambda(k) = lambda_max * RampUp(k), Eq. 14, using a sigmoid ramp-up."""
    return ramps.sigmoid_rampup(epoch, train_args.consistency_rampup)


def unpack_model_output(output):
    """Support models that return either logits or tuple/list where first item is logits."""
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _sample_uniform(low, high, size, device):
    return torch.empty(size, device=device).uniform_(low, high)


def _gaussian_kernel1d(sigma, device):
    radius = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_blur(x, sigma):
    """Separable Gaussian blur, same sigma for the whole batch (single-channel input)."""
    kernel1d = _gaussian_kernel1d(sigma, x.device)
    k = kernel1d.numel()
    pad = k // 2
    kernel_h = kernel1d.view(1, 1, 1, k)
    kernel_v = kernel1d.view(1, 1, k, 1)
    x = F.pad(x, (pad, pad, 0, 0), mode='reflect')
    x = F.conv2d(x, kernel_h)
    x = F.pad(x, (0, 0, pad, pad), mode='reflect')
    x = F.conv2d(x, kernel_v)
    return x


def _bernoulli_gate(prob, size, device, on_value, off_value):
    """Per-sample mask: `on_value` with probability `prob`, else `off_value` (the no-op value)."""
    if prob <= 0:
        return torch.full(size, off_value, device=device)
    if prob >= 1:
        return on_value
    apply = (torch.rand(size, device=device) < prob).to(on_value.dtype)
    return apply * on_value + (1.0 - apply) * off_value


def strong_intensity_augment(image, train_args):
    """
    Build the student's "strong" view from the teacher's clean "weak" view.

    Only intensity-space perturbations are applied (brightness, contrast, gamma,
    Gaussian noise, Gaussian blur): geometry is left untouched, so every pixel
    still lines up with the same pixel in the teacher's weak view and in the
    scribble label. This keeps the reliability comparison and partial CE pixel aligned, and follows the
    FixMatch/UniMatch weak-to-strong recipe restricted to appearance space for
    dense, pixel-aligned supervision, plus nnU-Net-style photometric augmentation
    for medical images.

    Each transform is applied independently per-sample with its own probability
    (`strong_*_prob`, default 0.3-0.5), matching how nnU-Net and RandAugment-style
    pipelines randomly turn transforms on/off rather than always applying every
    one at full strength. Gaussian blur is the exception: it is drawn once per
    iteration and shared across the whole batch, so a single conv call can be used.
    """
    x = image
    b = x.shape[0]
    device = x.device

    # Brightness & contrast: sampled per-sample magnitude, gated per-sample by their own probability.
    brightness = _sample_uniform(1.0 - train_args.strong_brightness, 1.0 + train_args.strong_brightness, (b, 1, 1, 1), device)
    brightness = _bernoulli_gate(train_args.strong_brightness_prob, (b, 1, 1, 1), device, on_value=brightness, off_value=1.0)

    contrast = _sample_uniform(1.0 - train_args.strong_contrast, 1.0 + train_args.strong_contrast, (b, 1, 1, 1), device)
    contrast = _bernoulli_gate(train_args.strong_contrast_prob, (b, 1, 1, 1), device, on_value=contrast, off_value=1.0)

    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * contrast + mean * brightness

    # Gamma correction: sampled per-sample, on a per-sample min-max normalized copy.
    if train_args.strong_gamma > 0 and train_args.strong_gamma_prob > 0:
        gamma = _sample_uniform(1.0 - train_args.strong_gamma, 1.0 + train_args.strong_gamma, (b, 1, 1, 1), device)
        gamma = _bernoulli_gate(train_args.strong_gamma_prob, (b, 1, 1, 1), device, on_value=gamma, off_value=1.0)
        x_min = x.amin(dim=(1, 2, 3), keepdim=True)
        x_max = x.amax(dim=(1, 2, 3), keepdim=True)
        x_range = (x_max - x_min).clamp_min(1e-5)
        x_norm = ((x - x_min) / x_range).clamp(0.0, 1.0).pow(gamma)
        x = x_norm * x_range + x_min

    # Gaussian noise: sampled per-sample, scaled by each sample's own std, gated per-sample.
    if train_args.strong_noise_std > 0 and train_args.strong_noise_prob > 0:
        std = x.std(dim=(1, 2, 3), keepdim=True)
        noise_scale = _sample_uniform(0.0, train_args.strong_noise_std, (b, 1, 1, 1), device)
        noise_scale = _bernoulli_gate(train_args.strong_noise_prob, (b, 1, 1, 1), device, on_value=noise_scale, off_value=0.0)
        x = x + torch.randn_like(x) * std * noise_scale

    # Gaussian blur: one draw per iteration (shared across the batch) for simplicity.
    if train_args.strong_blur_prob > 0 and random.random() < train_args.strong_blur_prob:
        sigma = random.uniform(train_args.strong_blur_sigma_min, train_args.strong_blur_sigma_max)
        x = _gaussian_blur(x, sigma)

    return x


def strong_cutout(image, train_args):
    """
    Optional CutOut on the student strong view (off by default).

    Occlusion does not shift geometry, so pixel alignment with the teacher/label
    is preserved. It is disabled by default because with sparse scribble labels,
    a hole can land on and blank out one of the few labeled pixels, adding noise
    to the already-scarce partial CE supervision.
    """
    if train_args.strong_cutout_prob <= 0:
        return image

    out = image.clone()
    fill_value = image.mean()
    b, _, h, w = out.shape
    for i in range(b):
        if random.random() < train_args.strong_cutout_prob:
            area_frac = random.uniform(0.01, max(train_args.strong_cutout_max_area, 0.01))
            hole_h = max(1, min(h, int(round((area_frac ** 0.5) * h))))
            hole_w = max(1, min(w, int(round((area_frac ** 0.5) * w))))
            cy = random.randint(0, h - hole_h)
            cx = random.randint(0, w - hole_w)
            out[i, :, cy:cy + hole_h, cx:cx + hole_w] = fill_value
    return out


def build_student_strong_view(image, train_args):
    x = strong_intensity_augment(image, train_args)
    x = strong_cutout(x, train_args)
    return x


def masked_soft_ce_loss(logits, target_prob, mask, eps=1e-8):
    """
    Masked soft cross-entropy, Eq. 12.

    L_pseudo = - (1 / (sum_i m_i + eps)) * sum_i m_i * sum_c q_{i,c} * log p^s_{i,c}

    Args:
        logits:      [B, C, H, W] student logits
        target_prob: [B, C, H, W] soft pseudo-label q, detached
        mask:        [B, 1, H, W] reliable mask m, detached
    """
    if mask.sum() < 1:
        # "If no reliable pixels are selected in a mini-batch, the pseudo-label loss is set to zero."
        return logits.new_tensor(0.0)

    log_prob = F.log_softmax(logits, dim=1)
    ce_map = -(target_prob * log_prob).sum(dim=1, keepdim=True)
    return (ce_map * mask).sum() / (mask.sum() + eps)


def make_calibration_holdout(label, ignore_index, holdout_fraction, num_classes):
    """Create a temporary calibration mask from the *existing* scribble pixels.

    The mask is sampled independently for every sample and class.  At least one
    labeled pixel is kept for supervised learning whenever a sample/class has at
    least one scribble pixel.  The original `label` tensor is never modified.

    Args:
        label: [B, ...] integer scribble map; ignore_index marks unlabeled pixels.
    Returns:
        bool tensor with the same shape as label.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError('calibration_holdout must be in (0, 1)')

    b = label.shape[0]
    flat_label = label.reshape(b, -1)
    flat_mask = torch.zeros_like(flat_label, dtype=torch.bool)

    for sample_i in range(b):
        for class_i in range(num_classes):
            indices = torch.nonzero(flat_label[sample_i] == class_i, as_tuple=False).flatten()
            n = indices.numel()
            if n <= 1:
                # With one scribble pixel, keep it for supervision rather than
                # sacrificing the only label from that class in this sample.
                continue

            n_hold = int(round(n * holdout_fraction))
            n_hold = max(1, min(n - 1, n_hold))
            chosen = indices[torch.randperm(n, device=label.device)[:n_hold]]
            flat_mask[sample_i, chosen] = True

    # Defensive: ignore pixels are never allowed into the calibration set.
    flat_mask &= flat_label != ignore_index
    return flat_mask.reshape_as(label)


def compute_reliability_score(student_prob, teacher_prob, eps=1e-8):
    """Compute a bounded reliability score for every pixel.

    Reliability is the product of three terms in [0, 1]:
        teacher confidence
        x (1 - normalized Jensen-Shannon divergence)
        x (1 - normalized teacher entropy)

    The teacher is the pseudo-target source, so confidence/entropy are measured
    from its weak-view prediction.  Student-teacher JS agreement measures whether
    the teacher prediction is stable to the student's strong-view perturbation.
    """
    student_prob = student_prob.detach().clamp_min(eps)
    teacher_prob = teacher_prob.detach().clamp_min(eps)
    student_prob = student_prob / student_prob.sum(dim=1, keepdim=True).clamp_min(eps)
    teacher_prob = teacher_prob / teacher_prob.sum(dim=1, keepdim=True).clamp_min(eps)

    teacher_conf, teacher_pred = torch.max(teacher_prob, dim=1)

    mixture = 0.5 * (student_prob + teacher_prob)
    js_st = 0.5 * (
        (student_prob * (student_prob.log() - mixture.log())).sum(dim=1)
        + (teacher_prob * (teacher_prob.log() - mixture.log())).sum(dim=1)
    )
    js_norm = (js_st / math.log(2.0)).clamp(0.0, 1.0)
    agreement = 1.0 - js_norm

    num_classes = teacher_prob.shape[1]
    if num_classes < 2:
        raise ValueError('reliability calibration requires at least 2 classes')
    entropy = -(teacher_prob * teacher_prob.log()).sum(dim=1)
    entropy_norm = (entropy / math.log(float(num_classes))).clamp(0.0, 1.0)
    certainty = 1.0 - entropy_norm

    reliability = (teacher_conf * agreement * certainty).clamp(0.0, 1.0)

    return {
        'score': reliability.detach(),
        'teacher_pred': teacher_pred.detach(),
        'teacher_conf': teacher_conf.detach(),
        'agreement': agreement.detach(),
        'certainty': certainty.detach(),
        'js_divergence': js_norm.detach(),
        'entropy': entropy_norm.detach(),
    }


class ScribbleReliabilityCalibrator:
    """Online class-wise calibration from held-out scribble pixels.

    Calibration is conditioned on the *teacher-predicted* class because the same
    predicted class determines which threshold is applied to an unlabeled pixel.
    For each class, exponentially decayed histograms track how often teacher
    predictions are correct at each reliability level.  We choose the lowest
    threshold (largest coverage) whose cumulative empirical precision is at least
    `target_precision` and whose effective sample count reaches `min_samples`.
    """

    def __init__(
        self,
        num_classes,
        num_bins=100,
        target_precision=0.95,
        min_samples=32.0,
        ema_decay=0.99,
        fallback_threshold=1.01,
    ):
        if num_classes < 2:
            raise ValueError('num_classes must be >= 2')
        if num_bins < 2:
            raise ValueError('calibration_bins must be >= 2')
        if not 0.0 < target_precision <= 1.0:
            raise ValueError('calibration_target_precision must be in (0, 1]')
        if min_samples <= 0:
            raise ValueError('calibration_min_samples must be > 0')
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError('calibration_ema_decay must be in [0, 1)')

        self.num_classes = num_classes
        self.num_bins = num_bins
        self.target_precision = target_precision
        self.min_samples = float(min_samples)
        self.ema_decay = ema_decay
        self.fallback_threshold = float(fallback_threshold)

        # CPU float64 is deliberate: these are tiny statistics, not model tensors,
        # and double precision avoids drift after many EMA updates.
        self.total_hist = torch.zeros(num_classes, num_bins, dtype=torch.float64)
        self.correct_hist = torch.zeros(num_classes, num_bins, dtype=torch.float64)

    @torch.no_grad()
    def update(self, reliability, teacher_pred, true_label, calibration_mask):
        valid = calibration_mask & (true_label >= 0) & (true_label < self.num_classes)

        # Forget stale teacher behavior even in iterations with no calibration
        # sample for a particular class.
        self.total_hist.mul_(self.ema_decay)
        self.correct_hist.mul_(self.ema_decay)

        if not valid.any():
            return

        scores = reliability[valid].detach().float().cpu().clamp(0.0, 1.0)
        pred = teacher_pred[valid].detach().long().cpu()
        truth = true_label[valid].detach().long().cpu()
        correct = pred.eq(truth)
        bins = torch.clamp((scores * self.num_bins).long(), max=self.num_bins - 1)

        for class_i in range(self.num_classes):
            class_mask = pred == class_i
            if not class_mask.any():
                continue
            class_bins = bins[class_mask]
            total = torch.bincount(class_bins, minlength=self.num_bins).to(torch.float64)
            good = torch.bincount(class_bins[correct[class_mask]], minlength=self.num_bins).to(torch.float64)
            self.total_hist[class_i].add_(total)
            self.correct_hist[class_i].add_(good)

    def _threshold_and_precision_for_class(self, class_i):
        total = self.total_hist[class_i]
        correct = self.correct_hist[class_i]

        # Cumulative statistics for score >= each bin's lower edge.
        cumulative_total = torch.flip(torch.cumsum(torch.flip(total, dims=[0]), dim=0), dims=[0])
        cumulative_correct = torch.flip(torch.cumsum(torch.flip(correct, dims=[0]), dim=0), dims=[0])
        precision = cumulative_correct / cumulative_total.clamp_min(1e-12)

        valid = (cumulative_total >= self.min_samples) & (precision >= self.target_precision)
        if not valid.any():
            return self.fallback_threshold, float('nan'), float(cumulative_total[0].item())

        # Lowest valid score threshold => maximum calibrated coverage.
        bin_i = int(torch.nonzero(valid, as_tuple=False)[0].item())
        threshold = bin_i / float(self.num_bins)
        return threshold, float(precision[bin_i].item()), float(cumulative_total[bin_i].item())

    def get_thresholds(self, device):
        thresholds = [self._threshold_and_precision_for_class(c)[0] for c in range(self.num_classes)]
        return torch.tensor(thresholds, dtype=torch.float32, device=device)

    def get_class_stats(self):
        return [self._threshold_and_precision_for_class(c) for c in range(self.num_classes)]


def calibration_batch_metrics(reliability, teacher_pred, true_label, calibration_mask, class_thresholds):
    """Diagnostics only; does not affect optimization or threshold fitting."""
    valid = calibration_mask & (true_label >= 0) & (true_label < class_thresholds.numel())
    if not valid.any():
        zero = reliability.new_tensor(0.0)
        nan = reliability.new_tensor(float('nan'))
        return {'raw_accuracy': nan, 'accepted_precision': nan, 'coverage': zero, 'count': zero}

    correct = teacher_pred.eq(true_label)
    threshold_map = class_thresholds[teacher_pred]
    accepted = valid & (reliability >= threshold_map)

    raw_accuracy = correct[valid].float().mean()
    coverage = accepted.float().sum() / valid.float().sum().clamp_min(1.0)
    if accepted.any():
        accepted_precision = correct[accepted].float().mean()
    else:
        accepted_precision = reliability.new_tensor(float('nan'))

    return {
        'raw_accuracy': raw_accuracy.detach(),
        'accepted_precision': accepted_precision.detach(),
        'coverage': coverage.detach(),
        'count': valid.float().sum().detach(),
    }


def build_calibrated_pseudo_label(
    teacher_prob,
    label,
    reliability,
    teacher_pred,
    class_thresholds,
    ignore_index,
    pseudo_mask_mode='unlabeled',
):
    """Select reliable unlabeled pixels using class-wise calibrated thresholds."""
    if pseudo_mask_mode != 'unlabeled':
        raise ValueError('Scribble-calibrated training supports pseudo_mask_mode="unlabeled" only')

    candidate_mask = label == ignore_index
    threshold_map = class_thresholds[teacher_pred]
    reliable = candidate_mask & (reliability >= threshold_map)
    reliable_mask = reliable.float().unsqueeze(1)

    # The clean weak-view EMA teacher is always the soft pseudo-target source.
    soft_pseudo_label = teacher_prob.detach()
    soft_pseudo_label = soft_pseudo_label / soft_pseudo_label.sum(dim=1, keepdim=True).clamp_min(1e-8)

    if candidate_mask.any():
        candidate_score = reliability[candidate_mask].mean()
    else:
        candidate_score = reliability.new_tensor(0.0)
    if reliable.any():
        accepted_score = reliability[reliable].mean()
    else:
        accepted_score = reliability.new_tensor(0.0)

    return {
        'soft_pseudo_label': soft_pseudo_label,
        'reliable_mask': reliable_mask.detach(),
        'reliable_ratio': reliable_mask.mean().detach(),
        'candidate_score': candidate_score.detach(),
        'accepted_score': accepted_score.detach(),
    }


def create_model(ema=False, num_classes=4):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes).cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def validate(model, valloader, db_val, num_classes, writer, iter_num):
    model.eval()

    metric_list = 0.0
    for sampled_val in valloader:
        metric_i = test_single_volume(
            sampled_val['image'],
            sampled_val['label'],
            model,
            classes=num_classes,
        )
        metric_list += np.array(metric_i)

    metric_list = metric_list / len(db_val)

    # Background is excluded, so num_classes - 1 foreground classes are logged.
    for class_i in range(num_classes - 1):
        writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
        writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]

    writer.add_scalar('info/val_mean_dice', performance, iter_num)
    writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

    model.train()
    return performance, mean_hd95


def train(train_args, snapshot_path):
    base_lr = train_args.base_lr
    num_classes = train_args.num_classes
    batch_size = train_args.batch_size
    max_iterations = train_args.max_iterations

    # Student f_theta and EMA teacher f_theta_bar (same architecture).
    model = create_model(ema=False, num_classes=num_classes)
    model_ema = create_model(ema=True, num_classes=num_classes)
    # Explicit initialization avoids any dependency on the internal copy direction
    # of utils.ema_optim.WeightEMA and guarantees a valid Mean Teacher start.
    model_ema.load_state_dict(model.state_dict())

    db_train = ACDCDataSets(
        base_dir=train_args.root_path,
        split='train',
        transform=transforms.Compose([RandomGenerator(train_args.patch_size)]),
        fold=train_args.fold,
        sup_type=train_args.sup_type,
    )
    db_val = ACDCDataSets(
        base_dir=train_args.root_path,
        fold=train_args.fold,
        split='val',
    )

    def worker_init_fn(worker_id):
        random.seed(train_args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    valloader = DataLoader(
        db_val,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    model.train()
    model_ema.train()

    # "The student is optimized with stochastic gradient descent using momentum and weight decay."
    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=0.0001,
    )

    # "The teacher is updated with an EMA decay rate of 0.99." (Eq. 15)
    ema_optimizer = WeightEMA(model, model_ema, train_args.ema_decay)

    # Eq. 2: partial cross-entropy over scribble-labeled pixels only (ignore_index excludes Omega_u).
    ce_loss = CrossEntropyLoss(ignore_index=num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info('%d iterations per epoch', len(trainloader))

    calibrator = ScribbleReliabilityCalibrator(
        num_classes=num_classes,
        num_bins=train_args.calibration_bins,
        target_precision=train_args.calibration_target_precision,
        min_samples=train_args.calibration_min_samples,
        ema_decay=train_args.calibration_ema_decay,
        fallback_threshold=train_args.calibration_fallback_threshold,
    )
    logging.info(
        'scribble calibration: holdout=%.3f target_precision=%.3f bins=%d min_samples=%.1f ema_decay=%.3f',
        train_args.calibration_holdout,
        train_args.calibration_target_precision,
        train_args.calibration_bins,
        train_args.calibration_min_samples,
        train_args.calibration_ema_decay,
    )

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for sampled_batch in trainloader:
            volume_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()

            # Weak view (teacher) vs. strong view (student). Geometry is identical for
            # both, so student_prob/teacher_prob/label_batch stay pixel-aligned; see the
            # module docstring for why this is the safe choice for scribble supervision.
            weak_batch = volume_batch
            if train_args.use_strong_aug:
                student_batch = build_student_strong_view(weak_batch, train_args)
            else:
                student_batch = weak_batch

            # -------------------------
            # 1. EMA teacher forward, no gradient (Eq. 1, alg. line 6)
            # -------------------------
            with torch.no_grad():
                ema_output = unpack_model_output(model_ema(weak_batch))
                teacher_prob = torch.softmax(ema_output, dim=1)

            # -------------------------
            # 2. Student forward (Eq. 1, alg. line 5)
            # -------------------------
            outputs = unpack_model_output(model(student_batch))
            student_prob = torch.softmax(outputs, dim=1)

            # -------------------------
            # 3. Rotate a small hold-out of the EXISTING scribble labels.
            #    Held-out pixels calibrate reliability in this step and are excluded
            #    from both partial CE and pseudo-label supervision in this step.
            # -------------------------
            calibration_mask = make_calibration_holdout(
                label=label_batch,
                ignore_index=num_classes,
                holdout_fraction=train_args.calibration_holdout,
                num_classes=num_classes,
            )
            supervised_label = label_batch.clone()
            supervised_label[calibration_mask] = num_classes

            # Partial CE remains the only direct supervision term.
            loss_sup = ce_loss(outputs, supervised_label.long())

            # -------------------------
            # 4. Scribble-calibrated reliability and pseudo-label selection.
            # -------------------------
            reliability_info = compute_reliability_score(
                student_prob=student_prob,
                teacher_prob=teacher_prob,
            )

            calibrator.update(
                reliability=reliability_info['score'],
                teacher_pred=reliability_info['teacher_pred'],
                true_label=label_batch,
                calibration_mask=calibration_mask,
            )
            class_thresholds = calibrator.get_thresholds(device=label_batch.device)

            calibration_info = calibration_batch_metrics(
                reliability=reliability_info['score'],
                teacher_pred=reliability_info['teacher_pred'],
                true_label=label_batch,
                calibration_mask=calibration_mask,
                class_thresholds=class_thresholds,
            )

            pseudo_info = build_calibrated_pseudo_label(
                teacher_prob=teacher_prob,
                label=label_batch,
                reliability=reliability_info['score'],
                teacher_pred=reliability_info['teacher_pred'],
                class_thresholds=class_thresholds,
                ignore_index=num_classes,
                pseudo_mask_mode=train_args.pseudo_mask_mode,
            )

            # -------------------------
            # 5. Masked soft pseudo-label loss (Eq. 12, alg. line 11)
            # -------------------------
            loss_pseudo = masked_soft_ce_loss(
                logits=outputs,
                target_prob=pseudo_info['soft_pseudo_label'],
                mask=pseudo_info['reliable_mask'],
            )

            # -------------------------
            # 6. Sigmoid ramp-up on pseudo-label weight (Eq. 13, 14)
            # -------------------------
            pseudo_weight = (
                get_current_consistency_weight(iter_num // len(trainloader), train_args)
                * train_args.pseudo_loss_weight
            )

            loss = loss_sup + pseudo_weight * loss_pseudo

            # -------------------------
            # 7. Student optimization (alg. line 12)
            # -------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # -------------------------
            # 8. EMA teacher update (Eq. 15, alg. line 13)
            # -------------------------
            ema_optimizer.step()

            # -------------------------
            # 9. Poly LR decay
            # -------------------------
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

            # -------------------------
            # 10. TensorBoard logging
            # -------------------------
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/loss_sup', loss_sup.item(), iter_num)
            writer.add_scalar('info/loss_pseudo', loss_pseudo.item(), iter_num)
            writer.add_scalar('info/pseudo_weight', pseudo_weight, iter_num)

            for class_i, threshold in enumerate(class_thresholds):
                writer.add_scalar('threshold/class_{}'.format(class_i), threshold.item(), iter_num)

            writer.add_scalar('pseudo/reliable_ratio', pseudo_info['reliable_ratio'].item(), iter_num)
            writer.add_scalar('pseudo/candidate_score', pseudo_info['candidate_score'].item(), iter_num)
            writer.add_scalar('pseudo/accepted_score', pseudo_info['accepted_score'].item(), iter_num)
            writer.add_scalar('reliability/mean', reliability_info['score'].mean().item(), iter_num)
            writer.add_scalar('reliability/teacher_conf', reliability_info['teacher_conf'].mean().item(), iter_num)
            writer.add_scalar('reliability/agreement', reliability_info['agreement'].mean().item(), iter_num)
            writer.add_scalar('reliability/certainty', reliability_info['certainty'].mean().item(), iter_num)
            writer.add_scalar('calibration/holdout_count', calibration_info['count'].item(), iter_num)
            writer.add_scalar('calibration/coverage', calibration_info['coverage'].item(), iter_num)
            if torch.isfinite(calibration_info['raw_accuracy']):
                writer.add_scalar('calibration/raw_accuracy', calibration_info['raw_accuracy'].item(), iter_num)
            if torch.isfinite(calibration_info['accepted_precision']):
                writer.add_scalar('calibration/accepted_precision', calibration_info['accepted_precision'].item(), iter_num)

            # -------------------------
            # 11. Console logging
            # -------------------------
            if iter_num % 200 == 0:
                threshold_text = ','.join('{:.3f}'.format(v.item()) for v in class_thresholds)
                logging.info(
                    'iteration %d : loss=%f, loss_sup=%f, loss_pseudo=%f, pseudo_weight=%f, '
                    'reliable=%f, rel_score=%f, cal_coverage=%f, thresholds=[%s]',
                    iter_num,
                    loss.item(),
                    loss_sup.item(),
                    loss_pseudo.item(),
                    pseudo_weight,
                    pseudo_info['reliable_ratio'].item(),
                    reliability_info['score'].mean().item(),
                    calibration_info['coverage'].item(),
                    threshold_text,
                )

            # -------------------------
            # 12. Validation and best checkpoint (selected by mean Dice)
            # -------------------------
            if iter_num > 1 and iter_num % 400 == 0:
                performance, mean_hd95 = validate(
                    model=model,
                    valloader=valloader,
                    db_val=db_val,
                    num_classes=num_classes,
                    writer=writer,
                    iter_num=iter_num,
                )

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(
                        snapshot_path,
                        'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)),
                    )
                    save_best = os.path.join(
                        snapshot_path,
                        '{}_best_model.pth'.format(train_args.model),
                    )
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    logging.info('save best model to %s', save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f',
                    iter_num,
                    performance,
                    mean_hd95,
                )

            # -------------------------
            # 13. Regular checkpoint
            # -------------------------
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info('save model to %s', save_mode_path)

            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    return 'Training Finished!'


if __name__ == '__main__':
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    snapshot_path = '../../checkpoints/{}_{}'.format(args.data, args.exp)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(
        filename=snapshot_path + '/log.txt',
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(str(args))

    result = train(args, snapshot_path)
    logging.info(result)
