"""NeSy-Scrib: rule-conditioned symbolic anatomical projection for ACDC/MSCMR.

Pure numpy/scipy/torch building blocks for a Mean-Teacher method whose core
mechanism is not a generic post-processing filter but a small, explicit
anatomical rule bank that (a) *repairs* the teacher's hard prediction with a
closed-form, class-specific, scribble-anchored projection operator, and
(b) turns the resulting edit map into a spatial pseudo-label-reliability
signal, alongside the teacher's own softmax-entropy confidence.

Design choices that intentionally depart from a literal "argmin over Q"
symbolic-repair formulation (KL-to-teacher + a logic energy + a scribble
term, optimized every batch):

- **Closed-form, not iterative.** The anatomical predicates this rule bank
  cares about (connected-component count, hole count) are discrete
  properties of a hard mask; optimizing them directly needs a differentiable
  surrogate (e.g. persistent-homology objectives), which adds real
  optimization/complexity cost for a per-batch pseudo-label step. This
  module instead applies one deterministic morphological projection per
  class and reports how much it had to change (see ``symbolic_repair``) --
  cheap, reproducible, and easy to unit-test, at the cost of not being
  provably the *minimal* edit (so this is documented and measured as
  "sparse anatomical repair", not "minimal repair").
- **Rule-dispatched, not one generic filter.** LV and RV are expected
  simply-connected (hole-count 0), so "keep the right component, then fill
  every hole" is safe for them. MYO is *not*: on slices where it actually
  surrounds the LV cavity, MYO genuinely has one hole, and blindly filling
  every hole (as generic "largest-component + fill-holes" post-processing
  would) destroys that anatomy. ``repair_myo`` instead identifies which of
  MYO's holes is the genuine LV cavity (by overlap with the already-repaired
  LV mask) and fills only the others.
- **Scribble-anchored, not size-anchored.** A naive "keep only the largest
  component" can drop a smaller component that a human scribble stroke
  actually falls inside. ``scribble_anchored_keep`` keeps every component
  touched by a scribble pixel of its own class when one exists, falling
  back to the largest component only when the class has no scribble
  evidence in this slice; the final repaired mask is *always* clamped back
  to the scribble labels afterwards, so symbolic repair can never override
  human annotation.
- **Repair vs. diagnosis.** Rules with a safe, unambiguous closed-form fix
  (connectivity, hole identity) are *repaired*. Relational rules whose
  violation has no single safe pixel-level fix (does LV's cavity boundary
  properly touch MYO? does MYO plausibly enclose LV at all?) are
  *diagnostic-only*: ``enclosure_violation``/``adjacency_violation`` never
  edit a pixel, they only discount reliability (Sec. 14's
  ``A_i = prod_k(1 - V_i^(k))``) -- symbolic knowledge can lower trust in a
  prediction it isn't safe to directly rewrite.

ACDC and MSCMR share the same 4-class convention
(``dataloader.scribblebench_3d.DATASET_CONFIGS``): 0=background, 1=RV,
2=MYO, 3=LV. This module hardcodes that convention (``RV``/``MYO``/``LV``
below) rather than taking class ids as arguments, since every dataset this
method supports uses it.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

RV, MYO, LV = 1, 2, 3
STRUCTURE_8CONN = np.ones((3, 3), dtype=bool)


def _labeled_components(mask_bool):
    return ndimage.label(mask_bool, structure=STRUCTURE_8CONN)


def scribble_anchored_keep(mask_bool, class_scribble_bool):
    """Repairable connectivity rule (per class).

    Keeps every connected component of ``mask_bool`` that contains at least
    one pixel of ``class_scribble_bool`` (never drops annotated evidence).
    If the class has no scribble pixels in this slice (or none of its
    scribble pixels fall inside any component of ``mask_bool`` -- the
    teacher disagreed with the scribble exactly there), falls back to
    keeping only the largest component. Anything not kept reverts to
    background.
    """
    mask_bool = np.asarray(mask_bool, dtype=bool)
    if not mask_bool.any():
        return np.zeros_like(mask_bool)
    labeled, n_components = _labeled_components(mask_bool)
    if n_components == 0:
        return np.zeros_like(mask_bool)

    if class_scribble_bool is not None and np.any(class_scribble_bool):
        touched = labeled[np.asarray(class_scribble_bool, dtype=bool)]
        anchored_labels = np.unique(touched[touched > 0])
        if anchored_labels.size > 0:
            return np.isin(labeled, anchored_labels)

    sizes = ndimage.sum(mask_bool, labeled, index=range(1, n_components + 1))
    largest_label = 1 + int(np.argmax(sizes))
    return labeled == largest_label


def repair_simply_connected(mask_bool, class_scribble_bool):
    """LV/RV repair operator: scribble-anchored connectivity, then fill every
    hole (both classes are expected simply-connected, hole-count 0)."""
    kept = scribble_anchored_keep(mask_bool, class_scribble_bool)
    return ndimage.binary_fill_holes(kept)


def repair_myo(myo_bool, lv_bool_repaired, class_scribble_bool):
    """MYO repair operator: scribble-anchored connectivity, then fill every
    hole *except* the one that overlaps the (already-repaired) LV mask the
    most -- the genuine ring cavity. If MYO has no holes, or no hole
    overlaps LV at all (including LV being entirely absent from this
    slice), there is no positive evidence of a real cavity, so every hole is
    filled -- this never *fabricates* a hole, only ever preserves one already
    present and anatomically justified.

    Known limitation: hole components are found with the same 8-connectivity
    as everything else in this module, so a spurious hole that happens to
    touch the genuine LV cavity edge- or corner-adjacent is labeled as one
    *merged* hole component and, since it does overlap LV, preserved whole
    -- the spurious part leaks through unrepaired in that adjacency case.
    Accepted here rather than adding 4-connectivity-only hole splitting,
    which would need its own separate justification and testing.
    """
    kept = scribble_anchored_keep(myo_bool, class_scribble_bool)
    filled = ndimage.binary_fill_holes(kept)
    holes = filled & ~kept
    if not holes.any():
        return kept

    labeled_holes, n_holes = _labeled_components(holes)
    if n_holes == 0 or not np.any(lv_bool_repaired):
        return filled

    overlaps = ndimage.sum(lv_bool_repaired, labeled_holes, index=range(1, n_holes + 1))
    best_index = int(np.argmax(overlaps))
    if overlaps[best_index] <= 0:
        return filled

    genuine_cavity = labeled_holes == (best_index + 1)
    result = filled.copy()
    result[genuine_cavity] = False
    return result


def symbolic_repair(pred_hard, scribble_label, ignore_index):
    """Rule-dispatched anatomical projection (see module docstring).

    Args:
        pred_hard: ``(H, W)`` int array, the teacher's hard argmax.
        scribble_label: ``(H, W)`` int array, this slice's scribble
            (``ignore_index`` = unlabeled).
        ignore_index: the dataset's unlabeled sentinel.
    Returns:
        ``repaired`` (``(H, W)`` int array) and ``changed_mask``
        (``(H, W)`` bool, ``repaired != pred_hard``).
    """
    pred_hard = np.asarray(pred_hard)
    scribble_label = np.asarray(scribble_label)
    if pred_hard.shape != scribble_label.shape:
        raise ValueError("pred_hard/scribble_label shape mismatch")

    rv_bool = repair_simply_connected(pred_hard == RV, scribble_label == RV)
    lv_bool = repair_simply_connected(pred_hard == LV, scribble_label == LV)
    myo_bool = repair_myo(pred_hard == MYO, lv_bool, scribble_label == MYO)

    # The three repaired masks are near-disjoint (each started from a
    # partition of one argmax map; components were only dropped and holes
    # only filled/reopened, never grown into new territory) except for MYO's
    # reopened ring cavity, which by construction is exactly where LV
    # belongs -- so LV is assigned last to win that overlap deterministically.
    repaired = np.zeros_like(pred_hard)
    repaired[rv_bool] = RV
    repaired[myo_bool] = MYO
    repaired[lv_bool] = LV

    annotated = scribble_label != ignore_index
    repaired[annotated] = scribble_label[annotated]

    changed_mask = repaired != pred_hard
    return repaired, changed_mask


def enclosure_violation(myo_bool, lv_bool, iterations=1):
    """Diagnostic-only relational rule: "MYO should enclose/border LV".

    ``1 - |dilate(MYO) ∩ LV| / |LV|`` (0 = every LV pixel borders MYO within
    ``iterations`` dilations, 1 = none does). ``0`` when LV is absent from
    this slice -- nothing to violate.
    """
    myo_bool = np.asarray(myo_bool, dtype=bool)
    lv_bool = np.asarray(lv_bool, dtype=bool)
    if not lv_bool.any():
        return 0.0
    dilated_myo = ndimage.binary_dilation(myo_bool, structure=STRUCTURE_8CONN, iterations=iterations)
    covered = np.logical_and(dilated_myo, lv_bool).sum()
    return float(1.0 - covered / max(int(lv_bool.sum()), 1))


def adjacency_violation(a_bool, b_bool, iterations=1):
    """Diagnostic-only relational rule: two present classes should be in
    contact (e.g. RV-MYO). ``1.0`` if both classes are present in-slice but
    never touch within ``iterations`` dilations, else ``0.0`` (including
    when either class is absent -- nothing to violate)."""
    a_bool = np.asarray(a_bool, dtype=bool)
    b_bool = np.asarray(b_bool, dtype=bool)
    if not (a_bool.any() and b_bool.any()):
        return 0.0
    dilated_a = ndimage.binary_dilation(a_bool, structure=STRUCTURE_8CONN, iterations=iterations)
    return 0.0 if np.logical_and(dilated_a, b_bool).any() else 1.0


def repair_reliability_map(changed_mask, diagnostic_violation, sigma):
    """Sec. 12's soft repair map: pixels near a symbolic edit are also
    distrusted, not only the edited pixels themselves, via a Gaussian decay
    of distance-to-nearest-edit; the diagnostic-only scalar for this slice
    then further discounts everything (``A_i = spatial_trust * (1 - V)``).

    Args:
        changed_mask: ``(H, W)`` bool, from :func:`symbolic_repair`.
        diagnostic_violation: scalar in ``[0, 1]`` for this slice.
        sigma: Gaussian decay radius in pixels.
    Returns:
        ``(H, W)`` float64 array in ``[0, 1]``; ``1`` where repair changed
        nothing nearby and no diagnostic rule was violated.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    changed_mask = np.asarray(changed_mask, dtype=bool)
    if not changed_mask.any():
        distrust = np.zeros(changed_mask.shape, dtype=np.float64)
    else:
        distance = ndimage.distance_transform_edt(~changed_mask)
        distrust = np.exp(-(distance**2) / (2.0 * sigma**2))
    spatial_trust = 1.0 - distrust
    return spatial_trust * (1.0 - float(diagnostic_violation))


def teacher_confidence(teacher_prob, eps=1e-8):
    """The proposal's own literal confidence term:
    ``R_conf = 1 - H(P_t) / log(C)``, elementwise over the class dim.

    Args:
        teacher_prob: ``[B, C, H, W]`` softmax probabilities, ``C >= 2``.
    Returns:
        ``[B, H, W]`` tensor in ``[0, 1]``.
    """
    num_classes = teacher_prob.shape[1]
    if num_classes < 2:
        raise ValueError("teacher_confidence requires at least 2 classes")
    probs = teacher_prob.clamp_min(eps)
    entropy = -(probs * probs.log()).sum(dim=1)
    return (1.0 - entropy / math.log(num_classes)).clamp(0.0, 1.0)


def weighted_pixel_ce_loss(logits, hard_target, weight, eps=1e-8):
    """Per-pixel cross-entropy against a hard label map, weighted
    continuously by ``weight`` and normalized by its sum -- the continuous-
    weight counterpart of ``voxtrust3d.masked_soft_ce_loss``'s binary-mask
    convention.

    Args:
        logits: ``[B, C, H, W]``.
        hard_target: ``[B, H, W]`` long tensor, values in ``[0, C)``
            (never ``ignore_index`` -- callers must resolve unlabeled/
            not-applicable pixels entirely through ``weight``, not through
            an out-of-range target).
        weight: ``[B, H, W]`` non-negative float tensor.
    """
    if weight.sum() < eps:
        return logits.new_tensor(0.0)
    per_pixel_ce = F.cross_entropy(logits, hard_target, reduction="none")
    return (per_pixel_ce * weight).sum() / (weight.sum() + eps)
