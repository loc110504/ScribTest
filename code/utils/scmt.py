"""SC-MT: Scribble-Calibrated Mean Teacher.

Standard Mean Teacher (student + EMA teacher) treats the teacher's raw
softmax confidence as a proxy for correctness when transferring pseudo-labels
to unlabeled pixels -- but confidence and correctness are not the same thing
(a model can be confidently wrong, especially far from any annotated pixel or
on a hard class). SC-MT's contribution is not the network but *how much* to
trust the teacher at every unlabeled pixel: instead of a fixed threshold or a
hand-picked distance-decay heuristic, it measures the teacher's *empirical*
accuracy, conditioned on (confidence, transfer distance, predicted class),
using the scribbles themselves as a rotating internal validation set, and
uses that measured accuracy as a continuous per-pixel reliability weight on
the consistency loss.

Algorithm, in four pieces:

1.  Rotating scribble-block hold-out (K-fold). Every connected scribble
    stroke ("block": one class's connected scribble voxels within one
    annotated slice, matching the granularity a human annotator actually
    draws at) is assigned once, at dataset-construction time, to one of
    ``num_folds`` folds. In training epoch ``t`` (one full pass over the
    ``DataLoader``), the blocks in fold ``t mod num_folds`` are held out of
    the partial-CE loss for that epoch and instead serve as a held-out probe:
    since their true label is known, whether the teacher's prediction there
    is right or wrong is directly observable. Over a full ``num_folds``-epoch
    cycle, every scribble stroke plays both roles (supervision and probe),
    exactly as many times.
2.  Teacher confidence and same-image transfer distance. At every candidate
    pixel the teacher's top-1 probability (confidence) and predicted class
    are recorded; a physical-distance KD-tree query gives the distance to the
    nearest *currently-supervised* (this epoch, not held out) scribble pixel
    of that predicted class, *within the same training patch/slice* -- not
    the whole volume (see :func:`batch_transfer_distance_from_labels`'s
    docstring for why this is a deliberate simplification, not an oversight).
    A predicted class with no such pixel in this image gets a dedicated
    "no-support" distance bin rather than an extrapolated distance.
3.  Class/confidence/distance-conditioned reliability calibration.
    :class:`ReliabilityCalibrationTable` is an EMA-updated empirical-accuracy
    lookup table ``g[class, confidence_bin, distance_bin]``, populated every
    epoch from that epoch's held-out blocks (known correct/incorrect,
    observed for free out of the ordinary training forward pass -- see
    ``scmt_step``). A cell with too few cumulative observations falls back to
    a coarser one: per-class marginal, then a single global value, then (if
    the table has no evidence at all yet) the raw confidence itself -- i.e.
    "trust the teacher's own confidence until proven otherwise".
4.  Reliability-weighted consistency loss. Only genuinely unlabeled pixels
    (never scribbled, in either role) receive a consistency target; the
    per-pixel weight is the calibration table's *measured* reliability at
    that pixel's (predicted class, confidence bin, distance bin), not a
    static threshold, so the student learns more from the teacher exactly
    where the teacher has been empirically shown to be trustworthy.

This module is pure torch/numpy: no dataset/I-O code. See
``train/train_scmt_2d.py`` (ACDC/MSCMR) and ``train/train_scmt_3d.py``
(WORD) for the training loops that wire these pieces to
``dataloader.scribblebench_{2d,3d}``. Kept self-contained (no imports from
``utils.voxtrust3d`` or other per-method utils modules) matching this
repo's convention that every method's algorithm lives in its own file, even
where a low-level helper (KD-tree transfer distance, aligned random
flip/rotate) is structurally similar to another method's.
"""

import hashlib
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Sec. 1: scribble blocks and their rotating K-fold assignment.
# ---------------------------------------------------------------------------


def stable_seed(seed, case, salt=""):
    """Deterministic 32-bit seed from a global seed and a case/slice id.

    Python's built-in ``hash`` is randomized per-process for strings, so it
    cannot be used to get a fold assignment that is reproducible across
    runs/workers; this uses a stable digest instead.
    """
    key = "{}::{}::{}".format(seed, case, salt).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "little")


def assign_scribble_blocks(label, num_classes):
    """Group scribble voxels of every class (background included) into blocks.

    A block is "all scribble voxels of one class, connected (8-connectivity),
    within one annotated slice" -- the granularity a human annotator actually
    draws one stroke at. Axis 0 is the coarse/through-plane axis for a 3D
    ``(D, H, W)`` volume; a 2D ``(H, W)`` ACDC/MSCMR training slice is already
    single-slice, so it is accepted directly with ``(h, w)`` coordinates
    instead of ``(d, h, w)``.

    Background (class 0) is included: in this repo's scribble convention it
    is an explicit, sparse annotation (distinct from ``ignore_index``), so a
    background stroke is real supervision and a real hold-out probe just like
    any foreground class.

    Args:
        label: ``(D, H, W)`` or ``(H, W)`` integer array; ``ignore_index``
            marks unlabeled voxels (any value outside ``[0, num_classes)`` is
            simply never matched by the per-class loop below).
    Returns:
        dict: ``class_id -> list of int64 (N_k, label.ndim) coordinate
        arrays``, one entry per block. Classes with no scribble voxels map to
        ``[]``.
    """
    structure = np.ones((3, 3), dtype=bool)
    blocks = {}
    if label.ndim == 3:
        for class_id in range(num_classes):
            class_blocks = []
            class_mask = label == class_id
            depth_indices = np.flatnonzero(class_mask.any(axis=(1, 2)))
            for d in depth_indices:
                plane = class_mask[d]
                component_labels, n_components = ndimage.label(plane, structure=structure)
                for component_id in range(1, n_components + 1):
                    h_idx, w_idx = np.nonzero(component_labels == component_id)
                    d_idx = np.full_like(h_idx, d)
                    coords = np.stack([d_idx, h_idx, w_idx], axis=1).astype(np.int64)
                    class_blocks.append(coords)
            blocks[class_id] = class_blocks
        return blocks
    if label.ndim == 2:
        for class_id in range(num_classes):
            class_blocks = []
            class_mask = label == class_id
            component_labels, n_components = ndimage.label(class_mask, structure=structure)
            for component_id in range(1, n_components + 1):
                h_idx, w_idx = np.nonzero(component_labels == component_id)
                coords = np.stack([h_idx, w_idx], axis=1).astype(np.int64)
                class_blocks.append(coords)
            blocks[class_id] = class_blocks
        return blocks
    raise ValueError("label must be a (D, H, W) or (H, W) array, got shape {}".format(label.shape))


def assign_block_folds(label, num_classes, num_folds, rng):
    """Assign every scribble block (stroke) a fold id in ``[0, num_folds)``.

    Unlike a fixed once-per-run holdout split, this is a full partition into
    ``num_folds`` rotating groups: every block ends up supervised in
    ``num_folds - 1`` out of every ``num_folds`` epochs and held out as a
    reliability probe in exactly one. There is no "never hold out the last
    block" exception here (unlike a fixed-split design) -- if a class has
    only one block in this image, that block simply loses this class's
    supervision here for one epoch in every ``num_folds``, which is the whole
    point of the rotation (every stroke plays the probe role too).

    Args:
        label: ``(D, H, W)`` or ``(H, W)`` integer scribble array.
        rng: a ``numpy.random.Generator``; caller controls determinism (see
            ``stable_seed``) so the assignment is fixed for an entire run.
    Returns:
        coords_by_class: ``{class_id: (N, label.ndim) int64 array}``.
        fold_by_class: ``{class_id: (N,) int64 array in [0, num_folds)}``,
            aligned 1:1 with ``coords_by_class``.
    """
    if num_folds < 2:
        raise ValueError("num_folds must be >= 2, got {}".format(num_folds))
    blocks_by_class = assign_scribble_blocks(label, num_classes)
    coords_by_class, fold_by_class = {}, {}
    empty_coords = np.zeros((0, label.ndim), dtype=np.int64)
    for class_id, blocks in blocks_by_class.items():
        if not blocks:
            coords_by_class[class_id] = empty_coords
            fold_by_class[class_id] = np.zeros((0,), dtype=np.int64)
            continue
        coords_list, fold_list = [], []
        for coords in blocks:
            fold_id = int(rng.integers(0, num_folds))
            coords_list.append(coords)
            fold_list.append(np.full(len(coords), fold_id, dtype=np.int64))
        coords_by_class[class_id] = np.concatenate(coords_list, axis=0)
        fold_by_class[class_id] = np.concatenate(fold_list, axis=0)
    return coords_by_class, fold_by_class


def partition_for_held_out_fold(coords_by_class, fold_by_class, held_out_fold):
    """Split a fold assignment into this epoch's sup/cal coordinates.

    Returns ``sup_coords`` (fold != held_out_fold, i.e. this epoch's
    ``Omega_s^train``) and ``cal_coords`` (fold == held_out_fold, i.e. this
    epoch's held-out probe set ``F_t``), both ``{class_id: (N, ndim) array}``.
    """
    sup_coords, cal_coords = {}, {}
    for class_id, coords in coords_by_class.items():
        folds = fold_by_class[class_id]
        held_mask = folds == held_out_fold
        sup_coords[class_id] = coords[~held_mask]
        cal_coords[class_id] = coords[held_mask]
    return sup_coords, cal_coords


def build_fold_map(label, num_classes, num_folds, rng):
    """Rasterize :func:`assign_block_folds` into a ``label``-shaped array.

    Every scribble voxel is replaced by its block's fold id; every other
    voxel (``ignore_index``, i.e. not scribbled at all) is ``-1``. This lets
    the fold assignment ride through a standard crop/flip/rotate pipeline as
    just another pixel-aligned integer channel, alongside ``image``/``label``.
    """
    coords_by_class, fold_by_class = assign_block_folds(label, num_classes, num_folds, rng)
    fold_map = np.full(label.shape, -1, dtype=np.int64)
    for class_id, coords in coords_by_class.items():
        if len(coords) == 0:
            continue
        index = tuple(coords[:, axis] for axis in range(label.ndim))
        fold_map[index] = fold_by_class[class_id]
    return fold_map


# ---------------------------------------------------------------------------
# Sec. 2: teacher confidence, predicted class and same-image transfer distance.
# ---------------------------------------------------------------------------


def teacher_confidence(teacher_prob):
    """``(confidence, predicted_class)`` from softmax probabilities.

    Elementwise over the class dimension only, so this is shape-agnostic:
    works for ``[B, C, D, H, W]`` (3D) and ``[B, C, H, W]`` (2D) alike.
    """
    confidence, predicted_class = teacher_prob.max(dim=1)
    return confidence.detach(), predicted_class.detach()


def assign_confidence_bin(confidence, num_confidence_bins):
    """Equal-width confidence bins over ``[0, 1]``; shape-agnostic."""
    if num_confidence_bins < 1:
        raise ValueError("num_confidence_bins must be positive")
    bin_id = (confidence * num_confidence_bins).long()
    return bin_id.clamp(0, num_confidence_bins - 1)


def build_class_trees(coords_by_class, spacing):
    """``{class_id: scipy.spatial.cKDTree}`` over supervised coordinates, in
    physical units. ``spacing`` has 3 entries ``(D, H, W)`` for a 3D patch or
    2 entries ``(H, W)`` for a 2D slice, matching ``coords_by_class``' rank.
    """
    spacing = np.asarray(spacing, dtype=np.float64)
    if spacing.ndim != 1 or spacing.shape[0] not in (2, 3):
        raise ValueError("spacing must have 2 entries (H, W) or 3 entries (D, H, W), got {}".format(spacing.shape))
    trees = {}
    for class_id, coords in coords_by_class.items():
        if len(coords) == 0:
            continue
        physical = coords.astype(np.float64) * spacing[None, :]
        trees[class_id] = cKDTree(physical)
    return trees


def query_transfer_distance(trees, class_ids, points_voxel, spacing):
    """Physical distance from each point to the supervised set of its own class.

    Args:
        trees: ``{class_id: cKDTree}`` (see ``build_class_trees``).
        class_ids: ``(N,)`` int array, the class each point queries against.
        points_voxel: ``(N, 3)`` or ``(N, 2)`` voxel coordinates.
        spacing: ``(3,)`` or ``(2,)`` physical spacing, matching rank.
    Returns:
        ``(N,)`` float64 array; ``NaN`` where ``trees`` has no entry for that
        point's class.
    """
    class_ids = np.asarray(class_ids)
    points_voxel = np.asarray(points_voxel)
    if points_voxel.ndim != 2 or points_voxel.shape[1] not in (2, 3):
        raise ValueError("points_voxel must have shape (N, 2) or (N, 3)")
    if len(class_ids) != len(points_voxel):
        raise ValueError("class_ids and points_voxel must have matching length")
    spacing = np.asarray(spacing, dtype=np.float64)

    distances = np.full(len(points_voxel), np.nan, dtype=np.float64)
    if len(points_voxel) == 0:
        return distances
    physical_points = points_voxel.astype(np.float64) * spacing[None, :]
    for class_id in np.unique(class_ids):
        tree = trees.get(int(class_id))
        if tree is None:
            continue
        mask = class_ids == class_id
        distance, _ = tree.query(physical_points[mask])
        distances[mask] = distance
    return distances


def batch_transfer_distance_from_labels(sup_label, predicted_class, valid, spacing, num_classes):
    """Per-sample Eq.-2 lookup, built directly from a dense ``sup_label`` patch.

    Deliberately patch/slice-local, not whole-volume: this repo's 2D
    pipeline trains on whole native slices (so "local" already equals "the
    whole image"), but the 3D pipeline trains on random sub-volume patches,
    and a scribble stroke's fold assignment is fixed for the whole volume
    (see ``assign_block_folds``) while the *patch* sampled for it changes
    every iteration. Rebuilding the KD-tree straight from whichever
    supervised voxels happen to be visible in the current patch avoids
    tracking original-volume coordinates through padding/cropping entirely
    (contrast with VoxTrust-3D's ``coord_d/h/w`` channels), at the cost of
    the distance being relative to "this patch's evidence" rather than "the
    whole case's evidence" -- a deliberate, documented simplification, not
    an oversight; the calibration table's EMA + hierarchical fallback (Sec.
    3) is not sensitive to this loosening the exact meaning of one distance
    bin edge.

    Args:
        sup_label: ``(B, D, H, W)`` or ``(B, H, W)`` int array, this epoch's
            supervised labels (``ignore_index`` elsewhere).
        predicted_class: same shape, the teacher's argmax.
        valid: same shape, bool -- voxels to actually query.
        spacing: ``(B, 3)`` or ``(B, 2)`` array.
        num_classes: real class count (``sup_label``'s ignore value is
            outside ``[0, num_classes)`` and is simply never matched).
    Returns:
        Float32 array shaped like ``predicted_class``; ``NaN`` where not
        requested or where that predicted class has no supervised voxel in
        this sample's patch.
    """
    batch_size = sup_label.shape[0]
    if spacing.shape[0] != batch_size:
        raise ValueError("spacing must have one entry per batch sample")
    out = np.full(predicted_class.shape, np.nan, dtype=np.float32)
    for b in range(batch_size):
        mask = valid[b]
        if not mask.any():
            continue
        coords_by_class = {c: np.argwhere(sup_label[b] == c) for c in range(num_classes)}
        trees = build_class_trees(coords_by_class, spacing[b])
        classes = predicted_class[b][mask]
        points = np.argwhere(mask)
        distance = query_transfer_distance(trees, classes, points, spacing[b])
        out[b][mask] = distance.astype(np.float32)
    return out


def fit_distance_bin_edges(per_case_partitions, num_classes, num_folds, num_distance_bins):
    """Fit per-class quantile bin edges once, before training starts.

    Simulates the full ``num_folds``-epoch rotation once (every block plays
    the held-out-probe role in exactly one simulated fold) and pools, over
    every case/slice and every fold, the whole-image distance from each
    held-out voxel to the supervised set of its own true class -- the same
    population :func:`batch_transfer_distance_from_labels` measures at
    training time, just computed whole-image instead of patch-local, which
    is representative enough to pick reasonable bin edges (see
    ``batch_transfer_distance_from_labels``'s docstring for why training
    itself does not use the whole image for the 3D pipeline).

    Args:
        per_case_partitions: iterable of dicts with keys ``coords_by_class``,
            ``fold_by_class`` (both ``{class_id: (N,) or (N,ndim) array}``,
            see ``assign_block_folds``) and ``spacing``.
        num_distance_bins: number of finite bins; the caller reserves one
            additional "no-support" bin index (``== num_distance_bins``) on
            top of what this function returns edges for.
    Returns:
        ``(num_classes, max(num_distance_bins - 1, 0))`` float64 array of
        ascending quantile edges; a class with no observations anywhere gets
        an all-``NaN`` row (harmless -- see ``assign_distance_bin``).
    """
    if num_distance_bins < 1:
        raise ValueError("num_distance_bins must be >= 1")
    pooled = {class_id: [] for class_id in range(num_classes)}
    for fold in range(num_folds):
        for case in per_case_partitions:
            sup_coords, cal_coords = partition_for_held_out_fold(
                case["coords_by_class"], case["fold_by_class"], fold
            )
            trees = build_class_trees(sup_coords, case["spacing"])
            for class_id, coords in cal_coords.items():
                if len(coords) == 0:
                    continue
                class_ids = np.full(len(coords), class_id, dtype=np.int64)
                distance = query_transfer_distance(trees, class_ids, coords, case["spacing"])
                valid = np.isfinite(distance)
                if valid.any():
                    pooled[class_id].append(distance[valid])

    n_edges = max(num_distance_bins - 1, 0)
    edges = np.full((num_classes, n_edges), np.nan, dtype=np.float64)
    if n_edges == 0:
        return edges
    for class_id, chunks in pooled.items():
        if not chunks:
            continue
        values = np.concatenate(chunks)
        if values.size == 0:
            continue
        quantiles = np.linspace(0.0, 1.0, num_distance_bins + 1)[1:-1]
        edges[class_id] = np.quantile(values, quantiles)
    return edges


def assign_distance_bin(distance, predicted_class, edges, num_finite_bins):
    """Bucket a batch of distances into ``[0, num_finite_bins]``.

    Index ``num_finite_bins`` is the dedicated "no-support" bin: used both
    when ``distance`` itself is not finite (undefined -- the predicted class
    has no supervised voxel in this image) and, degenerately, when that
    class's quantile edges were never fit (``NaN`` row in ``edges``, handled
    by ``nan_to_num(..., nan=inf)`` so such a class always lands in the first
    finite bin instead).

    Args:
        distance, predicted_class: spatial-rank-matched tensors.
        edges: ``[C, n_edges]`` tensor (see ``fit_distance_bin_edges``);
            ``n_edges == 0`` means no distance conditioning (single finite
            bin, degenerate ``num_finite_bins == 1``).
    """
    valid = torch.isfinite(distance)
    if edges.shape[-1] > 0:
        local_edges = torch.nan_to_num(edges[predicted_class], nan=math.inf)
        stratum = (distance.unsqueeze(-1) >= local_edges).sum(dim=-1)
    else:
        stratum = torch.zeros_like(predicted_class)
    stratum = stratum.clamp(0, num_finite_bins - 1)
    return torch.where(valid, stratum, torch.full_like(stratum, num_finite_bins))


# ---------------------------------------------------------------------------
# Sec. 3: class/confidence/distance-conditioned reliability calibration.
# ---------------------------------------------------------------------------


class ReliabilityCalibrationTable:
    """EMA-updated empirical accuracy ``g[class, confidence_bin, distance_bin]``.

    ``observe`` accumulates one epoch's held-out-fold outcomes (call once per
    training batch, from that batch's ``F_t`` voxels); ``commit_epoch`` folds
    the accumulated tally into the persistent table via EMA and resets the
    accumulator, and must be called exactly once at the end of every training
    epoch (one full pass over the ``DataLoader``). ``query`` looks up the
    table *as of the last commit* (i.e. evidence gathered so far this epoch
    is not used to weight this same epoch's consistency loss, avoiding a
    read-during-write inconsistency).

    A cell needs at least ``min_samples`` cumulative observations (summed
    across every epoch it has ever seen data) before it is trusted; below
    that it falls back, in order: this class's marginal (ignoring
    confidence/distance) -> a single global value -> the caller-supplied
    default (raw teacher confidence -- "trust the teacher's own confidence
    until the calibration has enough evidence to say otherwise").
    """

    def __init__(self, num_classes, num_confidence_bins, num_distance_bins, momentum=0.9, min_samples=10):
        if num_classes < 1 or num_confidence_bins < 1 or num_distance_bins < 1:
            raise ValueError("num_classes/num_confidence_bins/num_distance_bins must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must satisfy 0 <= momentum < 1")
        if min_samples < 1:
            raise ValueError("min_samples must be positive")
        self.num_classes = num_classes
        self.num_confidence_bins = num_confidence_bins
        self.num_distance_bins = num_distance_bins
        self.momentum = momentum
        self.min_samples = min_samples

        cell_shape = (num_classes, num_confidence_bins, num_distance_bins)
        self.cell_value = np.full(cell_shape, np.nan, dtype=np.float64)
        self.cell_count = np.zeros(cell_shape, dtype=np.int64)
        self.class_value = np.full(num_classes, np.nan, dtype=np.float64)
        self.class_count = np.zeros(num_classes, dtype=np.int64)
        self.global_value = math.nan
        self.global_count = 0
        self._reset_epoch_accumulator()

    def _reset_epoch_accumulator(self):
        cell_shape = self.cell_value.shape
        self._epoch_cell_correct = np.zeros(cell_shape, dtype=np.float64)
        self._epoch_cell_total = np.zeros(cell_shape, dtype=np.int64)
        self._epoch_class_correct = np.zeros(self.num_classes, dtype=np.float64)
        self._epoch_class_total = np.zeros(self.num_classes, dtype=np.int64)
        self._epoch_global_correct = 0.0
        self._epoch_global_total = 0

    def observe(self, class_ids, confidence_bins, distance_bins, corrects):
        """Accumulate one batch's held-out-fold outcomes into this epoch's tally."""
        class_ids = np.asarray(class_ids)
        n = len(class_ids)
        if n == 0:
            return
        confidence_bins = np.asarray(confidence_bins)
        distance_bins = np.asarray(distance_bins)
        corrects = np.asarray(corrects, dtype=np.float64)
        if not len({n, len(confidence_bins), len(distance_bins), len(corrects)}) == 1:
            raise ValueError("observe() arrays must all have the same length")

        index = (class_ids, confidence_bins, distance_bins)
        np.add.at(self._epoch_cell_correct, index, corrects)
        np.add.at(self._epoch_cell_total, index, 1)
        np.add.at(self._epoch_class_correct, class_ids, corrects)
        np.add.at(self._epoch_class_total, class_ids, 1)
        self._epoch_global_correct += float(corrects.sum())
        self._epoch_global_total += n

    def commit_epoch(self):
        """EMA-fold this epoch's tally into the persistent table."""
        has_obs = self._epoch_cell_total > 0
        empirical = np.divide(
            self._epoch_cell_correct, self._epoch_cell_total, out=np.zeros_like(self._epoch_cell_correct),
            where=has_obs,
        )
        blended = np.where(np.isnan(self.cell_value), empirical, self.momentum * self.cell_value + (1.0 - self.momentum) * empirical)
        self.cell_value = np.where(has_obs, blended, self.cell_value)
        self.cell_count = self.cell_count + self._epoch_cell_total

        has_class_obs = self._epoch_class_total > 0
        empirical_class = np.divide(
            self._epoch_class_correct, self._epoch_class_total, out=np.zeros_like(self._epoch_class_correct),
            where=has_class_obs,
        )
        blended_class = np.where(
            np.isnan(self.class_value), empirical_class,
            self.momentum * self.class_value + (1.0 - self.momentum) * empirical_class,
        )
        self.class_value = np.where(has_class_obs, blended_class, self.class_value)
        self.class_count = self.class_count + self._epoch_class_total

        if self._epoch_global_total > 0:
            empirical_global = self._epoch_global_correct / self._epoch_global_total
            self.global_value = (
                empirical_global if math.isnan(self.global_value)
                else self.momentum * self.global_value + (1.0 - self.momentum) * empirical_global
            )
            self.global_count += self._epoch_global_total

        self._reset_epoch_accumulator()

    def query(self, class_ids, confidence_bins, distance_bins, default_reliability):
        """Hierarchical lookup: cell -> class -> global -> ``default_reliability``."""
        class_ids = np.asarray(class_ids)
        confidence_bins = np.asarray(confidence_bins)
        distance_bins = np.asarray(distance_bins)
        default_reliability = np.asarray(default_reliability, dtype=np.float64)

        cell_value = self.cell_value[class_ids, confidence_bins, distance_bins]
        cell_ok = self.cell_count[class_ids, confidence_bins, distance_bins] >= self.min_samples
        out = np.where(cell_ok, cell_value, np.nan)

        class_value = self.class_value[class_ids]
        class_ok = self.class_count[class_ids] >= self.min_samples
        fallback_class = np.where(class_ok, class_value, np.nan)
        out = np.where(np.isnan(out), fallback_class, out)

        global_ok = self.global_count >= self.min_samples and not math.isnan(self.global_value)
        fallback_global = self.global_value if global_ok else np.nan
        out = np.where(np.isnan(out), fallback_global, out)

        out = np.where(np.isnan(out), default_reliability, out)
        return np.clip(out, 0.0, 1.0)

    def state_dict(self):
        return {
            "num_classes": self.num_classes,
            "num_confidence_bins": self.num_confidence_bins,
            "num_distance_bins": self.num_distance_bins,
            "momentum": self.momentum,
            "min_samples": self.min_samples,
            "cell_value": self.cell_value,
            "cell_count": self.cell_count,
            "class_value": self.class_value,
            "class_count": self.class_count,
            "global_value": self.global_value,
            "global_count": self.global_count,
        }

    def load_state_dict(self, state):
        if (
            state["num_classes"] != self.num_classes
            or state["num_confidence_bins"] != self.num_confidence_bins
            or state["num_distance_bins"] != self.num_distance_bins
        ):
            raise ValueError("checkpoint calibration table shape does not match this run's configuration")
        self.momentum = state["momentum"]
        self.min_samples = state["min_samples"]
        self.cell_value = np.array(state["cell_value"], dtype=np.float64)
        self.cell_count = np.array(state["cell_count"], dtype=np.int64)
        self.class_value = np.array(state["class_value"], dtype=np.float64)
        self.class_count = np.array(state["class_count"], dtype=np.int64)
        self.global_value = state["global_value"]
        self.global_count = state["global_count"]
        self._reset_epoch_accumulator()


# ---------------------------------------------------------------------------
# Sec. 4: reliability-weighted consistency loss.
# ---------------------------------------------------------------------------


def reliability_weighted_consistency_loss(student_prob, teacher_prob, reliability, omega_u):
    """``(1 / |Omega_u|) * sum_{i in Omega_u} r_i * mean_c (p_student - p_teacher)^2``.

    Shape-agnostic over the spatial rank ([B,C,D,H,W] or [B,C,H,W]).
    ``reliability``/``omega_u`` are spatial-rank-matched (no class dim).
    Normalizing by the pixel *count* (not the sum of weights) matches the
    algorithm's ``(1/|Omega_u|) sum r_i D_i`` formula exactly: a batch with
    uniformly low reliability contributes a correspondingly small loss,
    rather than being renormalized back up to full scale.
    """
    if student_prob.shape != teacher_prob.shape:
        raise ValueError("student_prob/teacher_prob shape mismatch")
    diff2 = (student_prob - teacher_prob.detach()).pow(2).mean(dim=1)
    weight = reliability.detach() * omega_u.float()
    denom = omega_u.float().sum().clamp_min(1.0)
    return (diff2 * weight).sum() / denom


# ---------------------------------------------------------------------------
# Weak(teacher)/strong(student) intensity augmentation. Only intensity-space
# perturbations are used (no spatial blur, unlike some Mean-Teacher
# variants), so this one implementation is shape-agnostic over the spatial
# rank and backs both the 2D and 3D pipelines without a per-rank convolution
# to special-case.
# ---------------------------------------------------------------------------


def _sample_uniform(low, high, size, device):
    return torch.empty(size, device=device).uniform_(low, high)


def _bernoulli_gate(prob, size, device, on_value, off_value):
    if prob <= 0:
        return torch.full(size, off_value, device=device)
    if prob >= 1:
        return on_value
    apply_mask = (torch.rand(size, device=device) < prob).to(on_value.dtype)
    return apply_mask * on_value + (1.0 - apply_mask) * off_value


def strong_intensity_augment(image, args):
    """Per-sample independently Bernoulli-gated brightness/contrast/noise.

    Args: ``image`` is ``[B, 1, H, W]`` or ``[B, 1, D, H, W]``; ``args`` needs
    ``strong_brightness(_prob)``, ``strong_contrast(_prob)``,
    ``strong_noise_std(_prob)``.
    """
    x = image
    batch_size = x.shape[0]
    device = x.device
    dims = tuple(range(1, x.ndim))
    shape = (batch_size,) + (1,) * (x.ndim - 1)

    brightness = _sample_uniform(1.0 - args.strong_brightness, 1.0 + args.strong_brightness, shape, device)
    brightness = _bernoulli_gate(args.strong_brightness_prob, shape, device, brightness, 1.0)
    contrast = _sample_uniform(1.0 - args.strong_contrast, 1.0 + args.strong_contrast, shape, device)
    contrast = _bernoulli_gate(args.strong_contrast_prob, shape, device, contrast, 1.0)

    mean = x.mean(dim=dims, keepdim=True)
    x = (x - mean) * contrast + mean * brightness

    if args.strong_noise_std > 0 and args.strong_noise_prob > 0:
        std = x.std(dim=dims, keepdim=True)
        noise_scale = _sample_uniform(0.0, args.strong_noise_std, shape, device)
        noise_scale = _bernoulli_gate(args.strong_noise_prob, shape, device, noise_scale, 0.0)
        x = x + torch.randn_like(x) * std * noise_scale
    return x


# ---------------------------------------------------------------------------
# Aligned crop/flip/rotate over an arbitrary dict of pixel-aligned channels.
# The standard dataset transforms (dataloader.scribblebench_3d.RandomCrop3D/
# RandomFlipRotate3D, dataloader.scribblebench_2d.RandomGenerator2D) only
# know about fixed "image"/"label"/"gt_label" keys; SC-MT needs one more
# aligned channel (the rasterized fold map, see build_fold_map) carried
# through the exact same crop/flip/rotate/resize policy so it stays
# pixel-aligned with image/label, hence these generic dict-based versions
# reproducing that same policy exactly.
# ---------------------------------------------------------------------------


def _pad_to_shape(array, output_size, value):
    padding = []
    for current, target in zip(array.shape, output_size):
        total = max(target - current, 0)
        padding.append((total // 2, total - total // 2))
    return np.pad(array, padding, mode="constant", constant_values=value), padding


def random_crop_3d(arrays, cval, patch_size, foreground_prob, num_classes, foreground_key="label"):
    """Generic dict-of-arrays counterpart of ``RandomCrop3D`` (identical
    padding/crop/foreground-bias policy: pad symmetrically to at least
    ``patch_size``, foreground-biased crop with probability
    ``foreground_prob`` using ``arrays[foreground_key]`` as the class map,
    else uniform random crop).
    """
    padded = {key: _pad_to_shape(value, patch_size, cval[key])[0] for key, value in arrays.items()}
    label = padded[foreground_key]
    shape = label.shape

    starts = None
    foreground = np.argwhere((label > 0) & (label < num_classes))
    if len(foreground) and random.random() < foreground_prob:
        center = foreground[np.random.randint(len(foreground))]
        starts = []
        for axis, (current, target) in enumerate(zip(shape, patch_size)):
            low = max(int(center[axis]) - target + 1, 0)
            high = min(int(center[axis]), current - target)
            starts.append(np.random.randint(low, high + 1) if high > low else low)
    if starts is None:
        starts = [
            np.random.randint(0, current - target + 1) if current > target else 0
            for current, target in zip(shape, patch_size)
        ]
    slices = tuple(slice(start, start + size) for start, size in zip(starts, patch_size))
    return {key: np.ascontiguousarray(value[slices]) for key, value in padded.items()}


def random_flip_rotate_3d(arrays):
    """Generic dict-of-arrays counterpart of ``RandomFlipRotate3D`` (identical
    per-axis flip probability and in-plane 90-degree rotation policy)."""
    for axis in range(3):
        if random.random() < 0.5:
            arrays = {key: np.flip(value, axis=axis) for key, value in arrays.items()}
    if random.random() < 0.5:
        sample_shape = next(iter(arrays.values())).shape
        square_plane = sample_shape[1] == sample_shape[2]
        choices = (1, 2, 3) if square_plane else (2,)
        k = random.choice(choices)
        arrays = {key: np.rot90(value, k=k, axes=(1, 2)) for key, value in arrays.items()}
    return {key: np.ascontiguousarray(value) for key, value in arrays.items()}


def random_flip_rotate_resize_2d(arrays, cval, output_size):
    """Generic dict-of-arrays counterpart of
    ``dataloader.scribblebench_2d.RandomGenerator2D`` (identical policy: 50%
    rot90+flip, else 25% random rotate +-20 degrees, else identity; always
    resized to ``output_size`` via nearest-neighbor).
    """
    if random.random() > 0.5:
        k = random.randint(0, 3)
        axis = random.randint(0, 1)
        arrays = {key: np.flip(np.rot90(value, k), axis=axis).copy() for key, value in arrays.items()}
    elif random.random() > 0.5:
        angle = random.randint(-20, 19)
        arrays = {
            key: ndimage.rotate(value, angle, order=0, reshape=False, mode="constant", cval=cval[key])
            for key, value in arrays.items()
        }
    height, width = next(iter(arrays.values())).shape
    scale = (output_size[0] / height, output_size[1] / width)
    return {key: np.ascontiguousarray(ndimage.zoom(value, scale, order=0)) for key, value in arrays.items()}
