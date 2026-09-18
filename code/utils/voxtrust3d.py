"""VoxTrust-3D: Scribble-Calibrated Risk-Aware Pseudo-Label Transfer.

Pure, torch/numpy-only building blocks for the method described in
``VoxTrust3D_CVPR2026_Proposed_Method.pdf`` (equation numbers below refer to
that draft). A single 3D U-Net student is paired with an EMA teacher; the
core mechanism is not the network but *which* teacher pseudo-labels the
student is allowed to learn from:

1.  the scribbles already present in a volume are split once, at the
    beginning of a training run, into a supervised part (``Omega_sup``) and a
    held-out calibration part (``Omega_cal``), at *spatial block* granularity
    (Sec. 4.1, Eq. 3);
2.  a bounded reliability score combines the teacher's semantic margin and
    its weak-to-strong prediction stability (Sec. 4.2, Eq. 5-7);
3.  reliability acceptance thresholds are calibrated online, conditioned on
    the teacher-predicted class and the physical 3D transfer distance to the
    nearest supervised voxel of that class, using a Wilson lower-confidence
    bound over a rolling memory of held-out scribble outcomes (Sec. 4.3-4.4,
    Eq. 8-15);
4.  only unlabeled voxels that clear the calibrated threshold receive a soft
    teacher target (Sec. 4.5, Eq. 16-17).

Trust-Advantage EMA (TA-EMA, this project's own extension, not part of the
base paper): points 1-4 above are all *teacher -> student* trust control
(which pseudo-labels the student may learn from). The EMA teacher update
itself (``theta_T <- alpha*theta_T + (1-alpha)*theta_S``) is the opposite
direction, *student -> teacher*, and a fixed ``alpha`` implicitly treats
every student iterate as equally trustworthy. TA-EMA spends the same
``Omega_cal`` evidence a second way -- non-gradient, exactly like point
3 above -- to compare the student's and the teacher's calibrated accuracy on
held-out scribble voxels and throttle the EMA update rate whenever the
student does not show a positive trust advantage. See
``RollingAccuracyBuffer``/``trust_advantage_alpha`` below and
``train/train_voxtrust3d_3d.py``'s ``voxtrust_step`` for how it is wired in.
Consequently ``Omega_cal`` is not used for gradient-based optimization
anywhere in this module, but it does now influence the parameter trajectory
indirectly through this non-gradient EMA gate, not only through the
teacher/pseudo-label machinery of points 1-4.

This module intentionally has no dataset/I-O code; see
``train/train_voxtrust3d_3d.py`` for the training loop that wires these
pieces to ``dataloader.scribblebench_3d.ScribbleBench3DDataset``.
"""

import hashlib
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Sec. 4.1: spatially blocked scribble calibration (Eq. 3)
# ---------------------------------------------------------------------------


def stable_seed(seed, case, salt=""):
    """Deterministic 32-bit seed from a global seed and a case id.

    Python's built-in ``hash`` is randomized per-process for strings, so it
    cannot be used to get a partition that is reproducible across runs/
    workers; this uses a stable digest instead.
    """
    key = "{}::{}::{}".format(seed, case, salt).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "little")


def assign_scribble_blocks(label, num_classes):
    """Group scribble voxels of every class (background included) into blocks.

    A block is the paper's default spatial-calibration unit (Sec. 4.1): "all
    scribble voxels of one class on one annotated slice". This is
    implemented as one connected component (8-connectivity) within one
    ``(class, D-slice)`` plane -- axis 0 is the coarse/through-plane axis for
    ACDC, MSCMR and WORD alike (verified against their NIfTI spacing), and
    using connected components rather than the whole per-slice set also
    isolates distinct strokes that happen to share a slice, matching "for a
    true 3D stroke, a connected stroke component is used".

    ACDC/MSCMR train as independent 2D slices (anisotropic spacing), not full
    3D volumes; since a block is already defined at single-slice granularity,
    the 2D pipeline's "volume" for this partition/calibration machinery is
    simply one slice, and a ``(H, W)`` ``label`` is accepted directly -- one
    plane, no D-slice loop, coordinates are ``(h, w)`` instead of ``(d, h, w)``.

    Background (class 0) is included: in this repo's scribble convention it
    is an explicit, sparse annotation (distinct from ``ignore_index``), so it
    is real supervision just like any foreground class and participates in
    the same partition/calibration machinery.

    Args:
        label: ``(D, H, W)`` (3D volume) or ``(H, W)`` (single 2D slice)
            integer array; ``ignore_index`` marks unlabeled voxels and must
            already be excluded by the caller (any value equal to a real
            class id in ``[0, num_classes)`` is treated as that class's
            scribble).
    Returns:
        dict: ``class_id -> list of int64 (N_k, 3 or 2) voxel-coordinate
        arrays``, one entry per block. Classes with no scribble voxels map
        to ``[]``.
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


def spatially_blocked_partition(label, ignore_index, num_classes, holdout_fraction, rng):
    """Split existing scribble voxels into Omega_sup / Omega_cal once (Eq. 3).

    The split happens at block granularity: a uniformly random ``holdout_fraction``
    of a class's blocks *in this volume* is reserved for calibration, never
    removing a class's last remaining block ("if a class has only one
    scribble block in a volume, that block remains in Omega_sup", Sec. 5).

    Args:
        label: ``(D, H, W)`` (3D volume) or ``(H, W)`` (single 2D slice)
            integer scribble array (``ignore_index`` = unlabeled).
        rng: a ``numpy.random.Generator``; caller controls determinism (see
            ``stable_seed``) so the partition is fixed for an entire run.
    Returns:
        sup_coords, cal_coords: ``{class_id: (N, label.ndim) int64 array}``.
        cal_block_id: ``{class_id: (N,) int64 array}``, the index (within
            that class's block list) each ``cal_coords`` voxel came from --
            used later to cap how many voxels from one block/stroke can
            enter the rolling calibration memory in a single update.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1), got {}".format(holdout_fraction))
    if np.any((label != ignore_index) & ((label < 0) | (label >= num_classes))):
        raise ValueError("label contains a class outside [0, num_classes) and != ignore_index")

    blocks_by_class = assign_scribble_blocks(label, num_classes)
    sup_coords, cal_coords, cal_block_id = {}, {}, {}
    for class_id, blocks in blocks_by_class.items():
        n_blocks = len(blocks)
        empty_coords = np.zeros((0, label.ndim), dtype=np.int64)
        if n_blocks == 0:
            sup_coords[class_id] = empty_coords
            cal_coords[class_id] = empty_coords
            cal_block_id[class_id] = np.zeros((0,), dtype=np.int64)
            continue

        n_hold = max(0, min(n_blocks - 1, int(round(n_blocks * holdout_fraction))))
        held_out = set(rng.choice(n_blocks, size=n_hold, replace=False).tolist()) if n_hold > 0 else set()

        sup_list, cal_list, cal_block_list = [], [], []
        for block_index, coords in enumerate(blocks):
            if block_index in held_out:
                cal_list.append(coords)
                cal_block_list.append(np.full(len(coords), block_index, dtype=np.int64))
            else:
                sup_list.append(coords)

        sup_coords[class_id] = np.concatenate(sup_list, axis=0) if sup_list else empty_coords
        cal_coords[class_id] = np.concatenate(cal_list, axis=0) if cal_list else empty_coords
        cal_block_id[class_id] = (
            np.concatenate(cal_block_list, axis=0) if cal_block_list else np.zeros((0,), dtype=np.int64)
        )
    return sup_coords, cal_coords, cal_block_id


# ---------------------------------------------------------------------------
# Sec. 4.3: physical 3D transfer distance (Eq. 8), via per-class KD-trees
# ---------------------------------------------------------------------------
#
# Ω_sup is sparse (a few hundred to a few thousand voxels even for the
# largest WORD organs), so a KD-tree nearest-neighbor query is an exact,
# memory-negligible stand-in for the paper's distance map: dense per-class
# EDT arrays over a full WORD volume (up to 512x512x241 voxels) would need
# tens of gigabytes across the training set, which a sparse-point KD-tree
# avoids entirely while giving the identical answer.


def build_class_trees(sup_coords, spacing):
    """``{class_id: scipy.spatial.cKDTree}`` over Omega_sup, in physical mm.

    ``spacing`` has 3 entries ``(D, H, W)`` for a 3D volume or 2 entries
    ``(H, W)`` for a single 2D ACDC/MSCMR slice, matching ``sup_coords``'
    coordinate rank.
    """
    spacing = np.asarray(spacing, dtype=np.float64)
    if spacing.ndim != 1 or spacing.shape[0] not in (2, 3):
        raise ValueError("spacing must have 2 entries (H, W) or 3 entries (D, H, W), got {}".format(spacing.shape))
    trees = {}
    for class_id, coords in sup_coords.items():
        if len(coords) == 0:
            continue
        physical = coords.astype(np.float64) * spacing[None, :]
        trees[class_id] = cKDTree(physical)
    return trees


def query_transfer_distance(trees, class_ids, points_voxel, spacing):
    """Eq. 8: physical distance from each point to Omega_sup of its own class.

    Args:
        trees: ``{class_id: cKDTree}`` for one volume (see ``build_class_trees``).
        class_ids: ``(N,)`` int array, the class each point queries against.
        points_voxel: ``(N, 3)`` (3D) or ``(N, 2)`` (2D) int/float array of
            voxel coordinates.
        spacing: ``(3,)`` or ``(2,)`` physical spacing, same convention and
            rank as ``build_class_trees``.
    Returns:
        ``(N,)`` float64 array; ``NaN`` where ``trees`` has no entry for that
        point's class (i.e. that class has no Omega_sup voxel in this volume).
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


def batch_transfer_distance(predicted_class, coord, valid, case_trees, spacing):
    """Batched Eq. 8 lookup for a training patch.

    Shape-agnostic over the spatial rank: works identically for 3D volume
    patches (``D, H, W``) and 2D ACDC/MSCMR slices (``H, W``), since it only
    ever indexes by boolean mask and transposes the leading coordinate-
    channel axis -- whichever rank ``coord``/``spacing`` carry flows straight
    through to :func:`query_transfer_distance`.

    Args:
        predicted_class: ``(B, D, H, W)`` or ``(B, H, W)`` int array, e.g. the
            teacher's argmax.
        coord: ``(B, 3, D, H, W)`` or ``(B, 2, H, W)`` int array of original-
            volume/slice voxel coordinates (see ``build_patch_coordinates``'s
            ``coord_d/h/w`` convention for the 3D case); a voxel is padding/
            invalid wherever any channel < 0.
        valid: ``(B, D, H, W)`` or ``(B, H, W)`` bool array, voxels to
            actually query (e.g. the union of Omega_cal and Omega_u
            candidates in the patch) -- skipping the rest keeps this fast
            even for large patches.
        case_trees: length-``B`` list of ``{class_id: cKDTree}``.
        spacing: ``(B, 3)`` or ``(B, 2)`` array, one spacing per sample.
    Returns:
        Float32 array shaped like ``predicted_class``; ``NaN`` where not
        requested or undefined.
    """
    batch_size = predicted_class.shape[0]
    if len(case_trees) != batch_size or spacing.shape[0] != batch_size:
        raise ValueError("case_trees/spacing must have one entry per batch sample")
    out = np.full(predicted_class.shape, np.nan, dtype=np.float32)
    for b in range(batch_size):
        mask = valid[b]
        if not mask.any():
            continue
        classes = predicted_class[b][mask]
        points = coord[b][:, mask].T
        distance = query_transfer_distance(case_trees[b], classes, points, spacing[b])
        out[b][mask] = distance.astype(np.float32)
    return out


def fit_distance_bins(per_case_partitions, num_classes, num_strata):
    """Sec. 4.3: per-class quantile bin edges and ``d_c^max``, fit once before training.

    Pools ``D_{y_j}(j)`` (distance to Omega_sup of its *own true* scribble
    class) for every calibration voxel ``j``, across all training volumes.
    This is well-defined for every calibration voxel: a class's last block is
    never held out, so Omega_sup of a calibration voxel's own class always
    exists in the same volume.

    Args:
        per_case_partitions: iterable of dicts with keys ``sup_coords``,
            ``cal_coords`` (both ``{class_id: (N,3) array}``) and ``spacing``.
        num_strata: ``B`` in the paper; ``B=1`` degenerates to no distance
            conditioning (every candidate lands in stratum 0).
    Returns:
        edges: ``(num_classes, max(num_strata - 1, 0))`` float64 array of
            quantile edges (ascending); ``NaN`` rows mean "never observed".
        d_max: ``(num_classes,)`` float64 array, the largest pooled distance
            seen for that class; ``NaN`` means "never observed".
    """
    if num_strata < 1:
        raise ValueError("num_strata must be >= 1")
    pooled = {class_id: [] for class_id in range(num_classes)}
    for case in per_case_partitions:
        trees = build_class_trees(case["sup_coords"], case["spacing"])
        for class_id, coords in case["cal_coords"].items():
            if len(coords) == 0:
                continue
            class_ids = np.full(len(coords), class_id, dtype=np.int64)
            distance = query_transfer_distance(trees, class_ids, coords, case["spacing"])
            valid = np.isfinite(distance)
            if valid.any():
                pooled[class_id].append(distance[valid])

    n_edges = max(num_strata - 1, 0)
    edges = np.full((num_classes, n_edges), np.nan, dtype=np.float64)
    d_max = np.full(num_classes, np.nan, dtype=np.float64)
    for class_id, chunks in pooled.items():
        if not chunks:
            continue
        values = np.concatenate(chunks)
        if values.size == 0:
            continue
        d_max[class_id] = float(values.max())
        if n_edges > 0:
            quantiles = np.linspace(0.0, 1.0, num_strata + 1)[1:-1]
            edges[class_id] = np.quantile(values, quantiles)
    return edges, d_max


# ---------------------------------------------------------------------------
# Sec. 4.2: volumetric reliability score (Eq. 5-7)
# ---------------------------------------------------------------------------


def reliability_score(student_prob, teacher_prob, eps=1e-6):
    """Eq. 5-7: bounded reliability from teacher margin and weak-to-strong stability.

    Unlike a 3-cue product (confidence x agreement x certainty), the paper's
    score is the geometric mean of exactly two clipped cues: the teacher's
    semantic margin (top1 vs. top2 class probability) and the student-teacher
    Jensen-Shannon stability. There is no separate confidence/entropy term.

    Elementwise over the class dimension only, so this is shape-agnostic:
    also used as-is for ACDC/MSCMR's ``[B, C, H, W]`` 2D slice pipeline.

    Args:
        student_prob, teacher_prob: ``[B, C, D, H, W]`` (3D) or ``[B, C, H, W]``
            (2D) softmax probabilities, ``C >= 2``, pixel-aligned (same
            spatial transform, only the student's input received extra
            intensity perturbation).
    Returns:
        dict of tensors shaped like ``student_prob`` minus the class dim:
        ``score`` (R_i, Eq. 7), ``teacher_pred`` (ĉ_i, long), ``teacher_conf``,
        ``margin`` (M_i, Eq. 5), ``stability`` (S_i, Eq. 6).
    """
    if student_prob.shape != teacher_prob.shape:
        raise ValueError(
            "student_prob/teacher_prob shape mismatch: {} vs {}".format(
                tuple(student_prob.shape), tuple(teacher_prob.shape)
            )
        )
    if student_prob.shape[1] < 2:
        raise ValueError("reliability_score requires at least 2 classes")

    teacher_prob = teacher_prob.clamp_min(0.0)
    student_prob = student_prob.clamp_min(0.0)

    top2 = teacher_prob.topk(2, dim=1).values
    teacher_conf, teacher_pred = teacher_prob.max(dim=1)
    margin = (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)

    p = student_prob.clamp_min(eps)
    q = teacher_prob.clamp_min(eps)
    p = p / p.sum(dim=1, keepdim=True)
    q = q / q.sum(dim=1, keepdim=True)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=1)
    js = (0.5 * kl_pm + 0.5 * kl_qm).clamp_min(0.0)
    js_norm = (js / math.log(2.0)).clamp(0.0, 1.0)
    stability = (1.0 - js_norm).clamp(0.0, 1.0)

    score = (margin.clamp(eps, 1.0) * stability.clamp(eps, 1.0)).sqrt()

    return {
        "score": score.detach(),
        "teacher_pred": teacher_pred.detach(),
        "teacher_conf": teacher_conf.detach(),
        "margin": margin.detach(),
        "stability": stability.detach(),
    }


# ---------------------------------------------------------------------------
# Sec. 4.4: class- and distance-conditioned risk calibration (Eq. 9-15)
# ---------------------------------------------------------------------------


def wilson_lower_bound(p_hat, n, delta=0.05):
    """Eq. 14: Wilson lower confidence bound. Vectorized; 0 where n <= 0."""
    p_hat = np.asarray(p_hat, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    z = float(norm.ppf(1.0 - delta / 2.0))
    z2 = z * z
    safe_n = np.maximum(n, 1.0)
    center = p_hat + z2 / (2.0 * safe_n)
    spread = z * np.sqrt(np.clip(p_hat * (1.0 - p_hat), 0.0, None) / safe_n + z2 / (4.0 * safe_n**2))
    denom = 1.0 + z2 / safe_n
    lower = (center - spread) / denom
    return np.where(n > 0, lower, 0.0)


def assign_distance_stratum(distance, edges_for_class):
    """Bucket distance(s) into ``[0, len(edges_for_class)]`` (Eq. text, Sec. 4.3)."""
    return np.searchsorted(edges_for_class, distance, side="right")


class RollingCalibrationBuffer:
    """Eq. 9-15: rolling per-(class, bin) and per-class-only (R, e) memory.

    ``update`` caps each spatial calibration block's contribution to
    ``block_cap`` uniformly sampled voxels ("each spatial calibration block
    contributes at most m_max uniformly sampled voxels per update", Sec.
    4.4), and both buffer families evict on a FIFO basis once a
    ``(class, bin)`` or ``class``-only memory reaches ``buffer_size`` (Eq. 10).
    """

    def __init__(self, num_classes, num_strata, buffer_size=4096, block_cap=64, rng=None):
        if num_classes < 1 or num_strata < 1:
            raise ValueError("num_classes and num_strata must be positive")
        if buffer_size < 1 or block_cap < 1:
            raise ValueError("buffer_size and block_cap must be positive")
        self.num_classes = num_classes
        self.num_strata = num_strata
        self.buffer_size = buffer_size
        self.block_cap = block_cap
        self.rng = rng if rng is not None else np.random.default_rng()
        self.per_bin = {}
        self.class_only = {}

    def _bin_buffer(self, class_id, bin_id):
        key = (int(class_id), int(bin_id))
        buffer = self.per_bin.get(key)
        if buffer is None:
            buffer = deque(maxlen=self.buffer_size)
            self.per_bin[key] = buffer
        return buffer

    def _class_buffer(self, class_id):
        key = int(class_id)
        buffer = self.class_only.get(key)
        if buffer is None:
            buffer = deque(maxlen=self.buffer_size)
            self.class_only[key] = buffer
        return buffer

    def update(self, class_ids, bin_ids, block_ids, reliabilities, corrects):
        """Push one batch of calibration observations (Algorithm 1, line 9).

        Args:
            class_ids: teacher-predicted class ĉ_j for each record.
            bin_ids: distance stratum b_j, or a negative sentinel where the
                distance is undefined for that record (per-bin buffer is
                skipped for those, but the class-only buffer still updates).
            block_ids: originating spatial calibration block, used only to
                cap over-representation of one stroke within this update.
            reliabilities, corrects: R_j and e_j = 1[ĉ_j == y_j].
        """
        class_ids = np.asarray(class_ids)
        bin_ids = np.asarray(bin_ids)
        block_ids = np.asarray(block_ids)
        reliabilities = np.asarray(reliabilities, dtype=np.float64)
        corrects = np.asarray(corrects, dtype=np.float64)
        n = len(class_ids)
        lengths = {len(bin_ids), len(block_ids), len(reliabilities), len(corrects)}
        if lengths != {n}:
            raise ValueError("update() arrays must all have the same length")
        if n == 0:
            return

        groups = {}
        for i in range(n):
            groups.setdefault((int(class_ids[i]), int(block_ids[i])), []).append(i)
        keep = []
        for indices in groups.values():
            if len(indices) > self.block_cap:
                indices = self.rng.choice(indices, size=self.block_cap, replace=False).tolist()
            keep.extend(indices)

        for i in keep:
            class_id = int(class_ids[i])
            record = (float(reliabilities[i]), float(corrects[i]))
            self._class_buffer(class_id).append(record)
            bin_id = int(bin_ids[i])
            if 0 <= bin_id < self.num_strata:
                self._bin_buffer(class_id, bin_id).append(record)

    @staticmethod
    def _fit_one(records, grid, n_min, rho, delta):
        if len(records) == 0:
            return math.inf
        arr = np.asarray(records, dtype=np.float64)
        order = np.argsort(arr[:, 0])
        r_sorted = arr[order, 0]
        e_sorted = arr[order, 1]
        n_total = len(r_sorted)
        suffix_k = np.concatenate([np.cumsum(e_sorted[::-1])[::-1], [0.0]])
        idx = np.searchsorted(r_sorted, grid, side="left")
        n_t = (n_total - idx).astype(np.float64)
        k_t = suffix_k[idx]
        p_hat = np.divide(k_t, np.maximum(n_t, 1.0))
        lcb = wilson_lower_bound(p_hat, n_t, delta)
        feasible = (n_t >= n_min) & (lcb >= rho)
        satisfying = np.flatnonzero(feasible)
        if satisfying.size == 0:
            return math.inf
        return float(grid[satisfying[0]])

    def fit_thresholds(self, grid, n_min, rho, delta):
        """Eq. 15: returns ``(thresholds[C, B], class_only_thresholds[C])``; ``inf`` abstains."""
        thresholds = np.full((self.num_classes, self.num_strata), math.inf, dtype=np.float64)
        for (class_id, bin_id), buffer in self.per_bin.items():
            thresholds[class_id, bin_id] = self._fit_one(list(buffer), grid, n_min, rho, delta)
        class_thresholds = np.full(self.num_classes, math.inf, dtype=np.float64)
        for class_id, buffer in self.class_only.items():
            class_thresholds[class_id] = self._fit_one(list(buffer), grid, n_min, rho, delta)
        return thresholds, class_thresholds

    def state_dict(self):
        return {
            "num_classes": self.num_classes,
            "num_strata": self.num_strata,
            "buffer_size": self.buffer_size,
            "block_cap": self.block_cap,
            "per_bin": {key: list(buffer) for key, buffer in self.per_bin.items()},
            "class_only": {key: list(buffer) for key, buffer in self.class_only.items()},
        }

    def load_state_dict(self, state):
        if state["num_classes"] != self.num_classes or state["num_strata"] != self.num_strata:
            raise ValueError("checkpoint calibrator shape does not match this run's configuration")
        self.buffer_size = state["buffer_size"]
        self.block_cap = state["block_cap"]
        self.per_bin = {
            tuple(key): deque(records, maxlen=self.buffer_size) for key, records in state["per_bin"].items()
        }
        self.class_only = {
            int(key): deque(records, maxlen=self.buffer_size) for key, records in state["class_only"].items()
        }


# ---------------------------------------------------------------------------
# Trust-Advantage EMA (TA-EMA): bidirectional trust control's student ->
# teacher half. Uses the same Omega_cal evidence as Sec. 4.3-4.4, but grouped
# by each calibration voxel's *true* scribble class and its distance to
# Omega_sup of that *same true* class -- not the teacher-predicted class/
# distance RollingCalibrationBuffer above uses -- because this is a
# student-vs-teacher comparison and both models must be scored on the
# identical subset; grouping by a prediction (which student and teacher can
# disagree on) would score them on two different subsets. Every calibration
# voxel's true class always has an Omega_sup entry in the same volume/slice
# (spatially_blocked_partition never holds out a class's last block, see its
# and fit_distance_bins' docstrings), so this distance is always defined.
# ---------------------------------------------------------------------------


def bin_distance_by_class(distance, class_ids, edges):
    """Per-row distance-stratum lookup with a per-row class id (vs.
    ``assign_distance_stratum``'s single shared class).

    Unlike the pseudo-label acceptance path (``build_pseudo_targets``), this
    has no ``d_max`` "reject rather than extrapolate" cutoff: it only buckets
    an already-known-correct/incorrect observation for a coarse Wilson-bound
    accuracy estimate, not a novel unlabeled prediction, so extrapolating
    into the outermost stratum for an out-of-range distance is acceptable.

    Args:
        distance: ``(N,)`` float array.
        class_ids: ``(N,)`` int array, values in ``[0, edges.shape[0])``.
        edges: ``(num_classes, num_strata - 1)`` array (see
            ``fit_distance_bins``); ``edges.shape[1] == 0`` means one stratum.
    Returns:
        ``(N,)`` int64 array in ``[0, num_strata - 1]``.
    """
    class_ids = np.asarray(class_ids)
    distance = np.asarray(distance, dtype=np.float64)
    if edges.shape[1] == 0:
        return np.zeros(len(class_ids), dtype=np.int64)
    local_edges = edges[class_ids]
    return (distance[:, None] >= local_edges).sum(axis=1).astype(np.int64)


class RollingAccuracyBuffer:
    """Rolling per-(true class, true-distance-stratum) correctness counts for
    one model (student or teacher), used only to estimate ``Q_m`` below.

    Lighter than ``RollingCalibrationBuffer``: TA-EMA only ever needs a
    Wilson lower-confidence bound on raw accuracy, not a fitted acceptance
    threshold over a reliability grid, so this stores 0/1 correctness only.
    It also does not cap one spatial block's contribution per update the way
    ``RollingCalibrationBuffer.update`` does -- ``Q_m`` is a single coarse
    per-stratum average, not a fine-grained fitted threshold, so occasional
    over-representation of one stroke has a bounded effect; add the same
    ``block_cap`` capping here if that proves too noisy in practice.
    """

    def __init__(self, num_classes, num_strata, buffer_size=4096):
        if num_classes < 1 or num_strata < 1:
            raise ValueError("num_classes and num_strata must be positive")
        if buffer_size < 1:
            raise ValueError("buffer_size must be positive")
        self.num_classes = num_classes
        self.num_strata = num_strata
        self.buffer_size = buffer_size
        self.cells = {}

    def _cell(self, class_id, bin_id):
        key = (int(class_id), int(bin_id))
        buffer = self.cells.get(key)
        if buffer is None:
            buffer = deque(maxlen=self.buffer_size)
            self.cells[key] = buffer
        return buffer

    def update(self, class_ids, bin_ids, corrects):
        class_ids = np.asarray(class_ids)
        bin_ids = np.asarray(bin_ids)
        corrects = np.asarray(corrects, dtype=np.float64)
        n = len(class_ids)
        if len({n, len(bin_ids), len(corrects)}) != 1:
            raise ValueError("update() arrays must all have the same length")
        for i in range(n):
            self._cell(class_ids[i], bin_ids[i]).append(float(corrects[i]))

    def quality(self, n_min, delta):
        """``Q_m``: the mean Wilson LCB over (class, stratum) cells with at
        least ``n_min`` observations; ``None`` if no cell yet qualifies (the
        caller should fall back to a fixed EMA rate in that case)."""
        supported = []
        for buffer in self.cells.values():
            n = len(buffer)
            if n < n_min:
                continue
            p_hat = sum(buffer) / n
            supported.append(wilson_lower_bound(p_hat, n, delta))
        if not supported:
            return None
        return float(np.mean(supported))

    def state_dict(self):
        return {
            "num_classes": self.num_classes,
            "num_strata": self.num_strata,
            "buffer_size": self.buffer_size,
            "cells": {key: list(buffer) for key, buffer in self.cells.items()},
        }

    def load_state_dict(self, state):
        if state["num_classes"] != self.num_classes or state["num_strata"] != self.num_strata:
            raise ValueError("checkpoint accuracy-buffer shape does not match this run's configuration")
        self.buffer_size = state["buffer_size"]
        self.cells = {
            tuple(key): deque(records, maxlen=self.buffer_size) for key, records in state["cells"].items()
        }


def trust_advantage_alpha(ema_decay, q_student, q_teacher, min_scale=0.1):
    """TA-EMA blend rate: the ``alpha_t`` such that
    ``theta_T <- alpha_t*theta_T + (1-alpha_t)*theta_S`` reproduces
    ``eta_t = (1-ema_decay) * Q_S^t * scale``, ``scale=1`` if
    ``Q_S^t > Q_T^t`` else ``min_scale``.

    Falls back to the fixed ``ema_decay`` (standard EMA, ``scale`` effectively
    irrelevant) whenever either quality estimate is unavailable -- warm-up,
    or no (class, stratum) cell has enough calibration evidence yet.

    ``min_scale`` is a deliberate departure from a hard freeze
    (``scale=0``) whenever the student lacks a trust advantage: an EMA
    teacher is a smoothed average of past students, so by construction it
    routinely scores at least as well as any single recent student iterate
    on this same held-out evidence, which would make a strict
    admit/freeze gate stall the teacher for long stretches of training
    rather than only during genuinely bad student iterates. ``min_scale``
    keeps the mechanism able to only *slow*, never *reverse*, the baseline
    EMA rate (``0 < alpha_t <= 1``, i.e. ``eta_t`` is always in
    ``[0, 1 - ema_decay]``), while guaranteeing the teacher keeps absorbing
    some signal from the student even when calibrated evidence favors the
    teacher.
    """
    if q_student is None or q_teacher is None:
        return ema_decay
    if not 0.0 <= min_scale <= 1.0:
        raise ValueError("min_scale must be in [0, 1]")
    scale = 1.0 if q_student > q_teacher else min_scale
    eta = (1.0 - ema_decay) * q_student * scale
    return 1.0 - eta


# ---------------------------------------------------------------------------
# Sec. 4.5: selective soft pseudo-label learning (Eq. 16-18)
# ---------------------------------------------------------------------------


def build_pseudo_targets(
    teacher_prob,
    omega_u,
    distance,
    teacher_pred,
    reliability,
    stratum_edges,
    thresholds_table,
    class_only_thresholds,
    d_max,
):
    """Eq. 16: acceptance mask A_i and the detached soft teacher target.

    Distance-defined candidates use the class-and-stratum threshold and are
    rejected outright when they exceed that class's ``d_max`` ("rejected
    rather than extrapolated"); candidates whose predicted class has no
    Omega_sup in this volume (``distance`` is ``NaN``, or that class's
    ``d_max`` was never observed at all) fall back to the class-only
    threshold; if that is also unavailable (``inf``), the voxel is rejected.

    Shape-agnostic over the spatial rank (indexing/elementwise only), so this
    is used as-is for both the 3D volume patches below and ACDC/MSCMR's 2D
    slice pipeline.

    Args:
        teacher_prob: ``[B, C, D, H, W]`` or ``[B, C, H, W]`` softmax teacher
            probabilities.
        omega_u: real (non-padding) unlabeled candidates, spatial-rank-matched
            bool tensor.
        distance: ``D_{ĉi}(i)`` float tensor, spatial-rank-matched; ``NaN`` if
            undefined.
        teacher_pred, reliability: spatial-rank-matched tensors, from
            ``reliability_score``.
        stratum_edges: ``[C, B-1]`` tensor (see ``fit_distance_bins``).
        thresholds_table: ``[C, B]`` tensor, τ_{c,b} (``inf`` = abstain).
        class_only_thresholds, d_max: ``[C]`` tensors.
    Returns:
        dict with ``target`` (shaped like ``teacher_prob``, detached
        ``sg(q_i)``), ``mask`` (``teacher_prob`` with the class dim replaced
        by a singleton, detached ``A_i``), and diagnostics.
    """
    if stratum_edges.shape[0] != thresholds_table.shape[0]:
        raise ValueError("stratum_edges/thresholds_table class dimension mismatch")

    distance_ok = torch.isfinite(distance)
    dmax_for_pred = d_max[teacher_pred]
    use_distance = distance_ok & torch.isfinite(dmax_for_pred)
    within_support = use_distance & (distance <= dmax_for_pred)

    if stratum_edges.shape[1] > 0:
        local_edges = stratum_edges[teacher_pred]
        stratum_bin = (distance.unsqueeze(-1) >= local_edges).sum(dim=-1)
    else:
        stratum_bin = torch.zeros_like(teacher_pred)
    stratum_bin = stratum_bin.clamp(0, thresholds_table.shape[1] - 1)
    stratum_threshold = thresholds_table[teacher_pred, stratum_bin]
    distance_branch_accept = within_support & (reliability >= stratum_threshold)

    class_only_threshold = class_only_thresholds[teacher_pred]
    fallback_accept = (~use_distance) & (reliability >= class_only_threshold)

    accept = omega_u & (distance_branch_accept | fallback_accept)
    mask = accept.float().unsqueeze(1).detach()

    target = teacher_prob.detach()
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)

    denom = omega_u.float().sum().clamp_min(1.0)
    return {
        "target": target,
        "mask": mask,
        "accepted_ratio": (accept.float().sum() / denom).detach(),
        "distance_branch_ratio": ((distance_branch_accept & omega_u).float().sum() / denom).detach(),
        "fallback_branch_ratio": ((fallback_accept & omega_u).float().sum() / denom).detach(),
        "stratum_bin": stratum_bin.detach(),
    }


def masked_soft_ce_loss(logits, target_prob, mask, eps=1e-8):
    """Eq. 17: masked soft cross-entropy over the accepted unlabeled voxels.

    ``L_pl = -(1 / (sum_i A_i + eps)) * sum_{i in Omega_u} sum_c A_i * sg(q_ic) * log p_ic``
    """
    if mask.sum() < 1:
        return logits.new_tensor(0.0)
    log_prob = F.log_softmax(logits, dim=1)
    ce_map = -(target_prob * log_prob).sum(dim=1, keepdim=True)
    return (ce_map * mask).sum() / (mask.sum() + eps)


# ---------------------------------------------------------------------------
# Sec. 5: augmentation -- weak (teacher) view vs. strong (student) view.
# Only intensity-space perturbations are used, so geometry (and therefore
# pixel alignment with the scribble labels and with the teacher's
# prediction) is never touched. Cutout/copy-paste are excluded from the core
# method (they would erase already-sparse scribble evidence).
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


def _gaussian_blur_3d(x, sigma):
    """Separable 3D Gaussian blur, one shared sigma for the whole batch."""
    for axis in (2, 3, 4):
        size = x.shape[axis]
        radius = min(max(1, int(round(3.0 * sigma))), max(size - 1, 1))
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
        kernel = torch.exp(-(coords**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel_shape = [1, 1, 1, 1, 1]
        kernel_shape[axis] = kernel.numel()
        kernel = kernel.view(kernel_shape)
        pad_pair = 4 - axis  # axis 4(W)->pair 0, 3(H)->pair 1, 2(D)->pair 2
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * pad_pair] = radius
        pad[2 * pad_pair + 1] = radius
        x = F.pad(x, pad, mode="reflect")
        x = F.conv3d(x, kernel)
    return x


def strong_intensity_augment_3d(image, args):
    """3D port of the 2D reference's weak-to-strong appearance perturbation.

    Brightness/contrast/gamma/noise are drawn per-sample and independently
    Bernoulli-gated by their own ``*_prob``; Gaussian blur is drawn once per
    iteration and shared across the batch (one conv3d call). No cutout: the
    paper explicitly excludes occlusion-style augmentation from the core
    method (Sec. 5, "Augmentation").
    """
    x = image
    batch_size = x.shape[0]
    device = x.device
    dims = (1, 2, 3, 4)
    shape = (batch_size, 1, 1, 1, 1)

    brightness = _sample_uniform(1.0 - args.strong_brightness, 1.0 + args.strong_brightness, shape, device)
    brightness = _bernoulli_gate(args.strong_brightness_prob, shape, device, brightness, 1.0)
    contrast = _sample_uniform(1.0 - args.strong_contrast, 1.0 + args.strong_contrast, shape, device)
    contrast = _bernoulli_gate(args.strong_contrast_prob, shape, device, contrast, 1.0)

    mean = x.mean(dim=dims, keepdim=True)
    x = (x - mean) * contrast + mean * brightness

    if args.strong_gamma > 0 and args.strong_gamma_prob > 0:
        gamma = _sample_uniform(1.0 - args.strong_gamma, 1.0 + args.strong_gamma, shape, device)
        gamma = _bernoulli_gate(args.strong_gamma_prob, shape, device, gamma, 1.0)
        x_min = x.amin(dim=dims, keepdim=True)
        x_max = x.amax(dim=dims, keepdim=True)
        x_range = (x_max - x_min).clamp_min(1e-5)
        x_norm = ((x - x_min) / x_range).clamp(0.0, 1.0).pow(gamma)
        x = x_norm * x_range + x_min

    if args.strong_noise_std > 0 and args.strong_noise_prob > 0:
        std = x.std(dim=dims, keepdim=True)
        noise_scale = _sample_uniform(0.0, args.strong_noise_std, shape, device)
        noise_scale = _bernoulli_gate(args.strong_noise_prob, shape, device, noise_scale, 0.0)
        x = x + torch.randn_like(x) * std * noise_scale

    if args.strong_blur_prob > 0 and random.random() < args.strong_blur_prob:
        sigma = random.uniform(args.strong_blur_sigma_min, args.strong_blur_sigma_max)
        x = _gaussian_blur_3d(x, sigma)

    return x


def _gaussian_blur_2d(x, sigma):
    """Separable 2D Gaussian blur, one shared sigma for the whole batch."""
    for axis in (2, 3):
        size = x.shape[axis]
        radius = min(max(1, int(round(3.0 * sigma))), max(size - 1, 1))
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
        kernel = torch.exp(-(coords**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel_shape = [1, 1, 1, 1]
        kernel_shape[axis] = kernel.numel()
        kernel = kernel.view(kernel_shape)
        pad_pair = 3 - axis  # axis 3(W)->pair 0, 2(H)->pair 1
        pad = [0, 0, 0, 0]
        pad[2 * pad_pair] = radius
        pad[2 * pad_pair + 1] = radius
        x = F.pad(x, pad, mode="reflect")
        x = F.conv2d(x, kernel)
    return x


def strong_intensity_augment_2d(image, args):
    """2D counterpart of :func:`strong_intensity_augment_3d`, for the
    ACDC/MSCMR slice pipeline. Identical perturbations and per-sample
    independent Bernoulli gating; only the tensor rank and the shared-blur
    convolution differ (``conv2d`` over ``[B, C, H, W]`` instead of
    ``conv3d`` over ``[B, C, D, H, W]``).
    """
    x = image
    batch_size = x.shape[0]
    device = x.device
    dims = (1, 2, 3)
    shape = (batch_size, 1, 1, 1)

    brightness = _sample_uniform(1.0 - args.strong_brightness, 1.0 + args.strong_brightness, shape, device)
    brightness = _bernoulli_gate(args.strong_brightness_prob, shape, device, brightness, 1.0)
    contrast = _sample_uniform(1.0 - args.strong_contrast, 1.0 + args.strong_contrast, shape, device)
    contrast = _bernoulli_gate(args.strong_contrast_prob, shape, device, contrast, 1.0)

    mean = x.mean(dim=dims, keepdim=True)
    x = (x - mean) * contrast + mean * brightness

    if args.strong_gamma > 0 and args.strong_gamma_prob > 0:
        gamma = _sample_uniform(1.0 - args.strong_gamma, 1.0 + args.strong_gamma, shape, device)
        gamma = _bernoulli_gate(args.strong_gamma_prob, shape, device, gamma, 1.0)
        x_min = x.amin(dim=dims, keepdim=True)
        x_max = x.amax(dim=dims, keepdim=True)
        x_range = (x_max - x_min).clamp_min(1e-5)
        x_norm = ((x - x_min) / x_range).clamp(0.0, 1.0).pow(gamma)
        x = x_norm * x_range + x_min

    if args.strong_noise_std > 0 and args.strong_noise_prob > 0:
        std = x.std(dim=dims, keepdim=True)
        noise_scale = _sample_uniform(0.0, args.strong_noise_std, shape, device)
        noise_scale = _bernoulli_gate(args.strong_noise_prob, shape, device, noise_scale, 0.0)
        x = x + torch.randn_like(x) * std * noise_scale

    if args.strong_blur_prob > 0 and random.random() < args.strong_blur_prob:
        sigma = random.uniform(args.strong_blur_sigma_min, args.strong_blur_sigma_max)
        x = _gaussian_blur_2d(x, sigma)

    return x


# ---------------------------------------------------------------------------
# Padding-safe patch sampling that never materializes a full-volume-sized
# temporary array. WORD volumes are up to 512x512x241 voxels; padding a copy
# of image/label/coordinate channels to at least the patch size (as
# ``dataloader.scribblebench_3d``'s ``RandomCrop3D`` does for its fixed
# "image"/"label"/"gt_label" keys) costs ~500MB-1.5GB *per extra channel per
# sample*, and with ``num_workers`` > 0 that cost is paid independently by
# every worker process -- this is exactly what triggered an OOM kill in
# testing. Every function below instead gathers only the ``patch_size``
# window directly out of the native-resolution array via fancy indexing
# (``gather_patch``) or a bounding-box point filter (``scatter_points_into_patch``),
# so peak per-sample memory is O(patch size), not O(volume size), regardless
# of how many extra pixel-aligned channels (role labels, block ids,
# coordinate channels) are carried alongside the image.
# ---------------------------------------------------------------------------


def choose_patch_origin(shape, patch_size, foreground_coords=None, foreground_prob=0.0):
    """Pick a patch window's position without allocating a padded array.

    Reproduces the exact statistics of "symmetrically pad every axis to at
    least ``patch_size``, then take one uniform (optionally
    foreground-biased) crop" -- matching
    ``dataloader.scribblebench_3d.RandomCrop3D`` -- but returns only the
    window's ``origin`` in the *original* (unpadded) volume's coordinates.

    Local patch index ``j`` along an axis maps to original coordinate
    ``origin[axis] + j``; this is a real, in-bounds voxel iff
    ``0 <= origin[axis] + j < shape[axis]`` (see ``build_patch_coordinates``).
    ``origin[axis]`` may be negative, or ``origin[axis] + patch_size[axis]``
    may exceed ``shape[axis]``, exactly where padding would have been used.

    Args:
        foreground_coords: ``(N, 3)`` array of candidate center voxels in
            the *original* volume (e.g. scribble foreground), or ``None``/
            empty to disable foreground biasing regardless of ``foreground_prob``.
    """
    if len(shape) != 3 or len(patch_size) != 3:
        raise ValueError("shape and patch_size must both be (D, H, W)")
    padding = []
    for current, target in zip(shape, patch_size):
        total = max(target - current, 0)
        padding.append((total // 2, total - total // 2))
    padded_shape = tuple(current + before + after for current, (before, after) in zip(shape, padding))

    starts = None
    if foreground_coords is not None and len(foreground_coords) and random.random() < foreground_prob:
        pick = foreground_coords[np.random.randint(len(foreground_coords))]
        starts = []
        for axis, (current, target) in enumerate(zip(padded_shape, patch_size)):
            center = int(pick[axis]) + padding[axis][0]
            low = max(center - target + 1, 0)
            high = min(center, current - target)
            starts.append(np.random.randint(low, high + 1) if high > low else low)
    if starts is None:
        starts = [
            np.random.randint(0, current - target + 1) if current > target else 0
            for current, target in zip(padded_shape, patch_size)
        ]
    return [start - pad[0] for start, pad in zip(starts, padding)]


def build_patch_coordinates(shape, origin, patch_size):
    """Per-voxel original-volume coordinates for a patch window at ``origin``.

    Returns ``coord_d, coord_h, coord_w, valid``, each ``patch_size``-shaped:
    the three coordinate channels carry the sentinel ``-1`` wherever the
    voxel falls outside ``shape`` (i.e. is virtual padding), so
    ``coord_d >= 0`` alone is a valid/real-voxel test downstream. No array
    of size ``shape`` is ever allocated.
    """
    axis_ranges = [np.arange(o, o + size) for o, size in zip(origin, patch_size)]
    coord_d, coord_h, coord_w = np.meshgrid(*axis_ranges, indexing="ij")
    valid = (
        (coord_d >= 0) & (coord_d < shape[0])
        & (coord_h >= 0) & (coord_h < shape[1])
        & (coord_w >= 0) & (coord_w < shape[2])
    )
    coord_d = np.where(valid, coord_d, -1)
    coord_h = np.where(valid, coord_h, -1)
    coord_w = np.where(valid, coord_w, -1)
    return coord_d, coord_h, coord_w, valid


def gather_patch(array, origin, patch_size, valid, fill_value):
    """Fancy-index a ``patch_size`` window out of ``array`` at ``origin``.

    Equivalent to padding ``array`` to at least ``patch_size`` with
    ``fill_value`` and then slicing out that window, but without ever
    allocating the padded copy: out-of-range positions are gathered from a
    clamped (in-bounds, discarded) index and then overwritten by
    ``fill_value`` via ``valid`` (see ``build_patch_coordinates``).
    """
    safe_axes = [
        np.clip(np.arange(o, o + size), 0, dim - 1) for o, size, dim in zip(origin, patch_size, array.shape)
    ]
    grids = np.meshgrid(*safe_axes, indexing="ij")
    gathered = array[tuple(grids)]
    return np.where(valid, gathered, fill_value)


def scatter_points_into_patch(coords, values, origin, patch_size, out):
    """Write sparse ``(N, 3)`` original-volume ``coords``/``values`` into ``out``.

    Only points that fall inside the ``patch_size`` window at ``origin`` are
    written (bounding-box filter + index shift); the rest are dropped. No
    full-volume array is built -- this is how ``VoxTrustPatch3DDataset``
    rasterizes each case's (typically thousands of voxels, not millions)
    calibration points into a patch-sized mask/block-id channel.
    """
    if len(coords) == 0:
        return
    local = coords - np.asarray(origin, dtype=coords.dtype)[None, :]
    patch_size_arr = np.asarray(patch_size)
    inside = np.all((local >= 0) & (local < patch_size_arr[None, :]), axis=1)
    if not inside.any():
        return
    local = local[inside]
    out[local[:, 0], local[:, 1], local[:, 2]] = values[inside]


def random_flip_rotate(arrays):
    """Aligned random flips (each axis, p=0.5) + one 90-degree in-plane
    rotation (p=0.5), applied identically to every ``(D, H, W)`` array in
    ``arrays`` so image/label/coordinate channels stay pixel-aligned.
    Mirrors ``dataloader.scribblebench_3d.RandomFlipRotate3D``'s policy
    exactly (in-plane-only rotation, since axis 0 has different physical
    spacing from axes 1-2 in every ScribbleBench dataset).
    """
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
    """2D counterpart of :func:`random_flip_rotate` for the ACDC/MSCMR slice
    pipeline.

    3D training crops a fixed-size patch out of a much larger volume (see
    ``choose_patch_origin``/``build_patch_coordinates``/``gather_patch``),
    so out-of-bounds padding is a real concern there. 2D ACDC/MSCMR slices
    are already small, so training uses the *whole* native slice resized to
    ``output_size`` -- there is no crop, no padding, and every array in
    ``arrays`` (including coordinate channels) starts fully valid. This
    function therefore also performs the resize step (nearest-neighbor,
    matching ``dataloader.scribblebench_2d.RandomGenerator2D``, whose exact
    augmentation policy it mirrors: 50% rot90+flip, else 25% random rotate
    +-20 degrees, else identity), and takes an explicit per-key fill value
    for the corners the rotate branch introduces (label-like channels want
    their own ignore/background sentinel; the image wants 0; coordinate
    channels want -1, matching ``build_patch_coordinates``'s padding
    sentinel convention, even though nothing here is actually padding).

    Args:
        arrays: ``{key: (H, W) array}``, all the same native shape.
        cval: ``{key: fill value}`` for every key in ``arrays``, used only by
            the rotate branch.
        output_size: ``(H, W)`` resize target.
    Returns:
        ``{key: (H, W) array}`` at ``output_size`` resolution.
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
