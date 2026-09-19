"""DMPLS (Luo et al., MICCAI 2022) -- ``Scribble-Supervised Medical Image
Segmentation via Dual-Branch Network and Dynamically Mixed Pseudo Labels
Supervision``, the single-stage predecessor of DMSPS (Han et al., MedIA
2024, ``code/utils/dmsps.py``).

DMPLS is exactly DMSPS without its stage-2 uncertainty-guided label
expansion: a shared-encoder dual-decoder network (``UNetCCT2D``,
``networks/unet_2d.py``) is trained end-to-end with (paper Eq. 4)::

    L_total = 0.5 * (L_pCE(y1, s) + L_pCE(y2, s)) + lambda * L_PLS(PL, y1, y2)

where the dynamically mixed pseudo label ``PL`` (Eq. 2, hard argmax of a
randomly-weighted mix of the two decoders' softmax outputs) and the soft
pseudo-label supervision ``L_PLS`` (Eq. 3) are the same
``dynamic_mixed_pseudo_label``/``soft_pseudo_supervision_loss`` primitives
already verified against ``HiLab-git/WSL4MIS`` (the official DMPLS/DMSPS
repository) in ``utils/dmsps.py`` -- re-exported here rather than
duplicated, since the equations are identical. Per the paper's own
Implementation Details (Sec. 3.1), ``lambda = 0.5``. Only the ACDC/MSCMR 2D
pipeline is implemented (see ``train/train_dmpls_2d.py``); this project's
benchmark does not run DMPLS on WORD.
"""

import numpy as np
import torch.nn.functional as F

from utils.dmsps import dynamic_mixed_pseudo_label, soft_pseudo_supervision_loss
from train.common_3d import partial_cross_entropy


def dmpls_step(model, image, target, ignore_index, args):
    """One DMPLS forward pass; returns the total loss and its components."""
    main_logits, aux_logits = model(image, return_auxiliary=True)
    loss_pce_main, labeled_voxels = partial_cross_entropy(main_logits, target, ignore_index)
    loss_pce_aux, _ = partial_cross_entropy(aux_logits, target, ignore_index)
    loss_pce = 0.5 * (loss_pce_main + loss_pce_aux)

    probs_main = F.softmax(main_logits, dim=1)
    probs_aux = F.softmax(aux_logits, dim=1)
    alpha = float(np.random.uniform(0.0, 1.0))
    pseudo_target = dynamic_mixed_pseudo_label(probs_main, probs_aux, alpha)
    loss_pls = soft_pseudo_supervision_loss(probs_main, probs_aux, pseudo_target)

    total = loss_pce + args.lambda_pls * loss_pls
    components = {
        "pce": loss_pce.item(),
        "pls": loss_pls.item(),
        "alpha": alpha,
        "labeled_voxels": labeled_voxels.item(),
    }
    return total, components


__all__ = [
    "dmpls_step",
    "dynamic_mixed_pseudo_label",
    "soft_pseudo_supervision_loss",
]
