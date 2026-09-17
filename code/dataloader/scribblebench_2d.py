"""Shared 2D slice data loading and augmentation for ACDC/MSCMR.

ACDC and MSCMR are cardiac MRI with strongly anisotropic voxel spacing (thin
in-plane, thick through-plane); the standard protocol in the scribble-
supervision literature (WSL4MIS, DMSPS, CycleMix, ScribFormer) trains these
two datasets as independent 2D slices and stitches predictions back into a 3D
volume only at evaluation time. This module builds a flat per-(case, slice)
view on top of :class:`ScribbleBench3DDataset`'s already-verified nii.gz
loading, label pairing and normalization -- it does not duplicate that logic.
WORD (near-isotropic CT) stays 3D-only in this benchmark and is not
supported here.
"""

import random

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from .scribblebench_3d import ScribbleBench3DDataset


SUPPORTED_2D_DATASETS = ("ACDC", "MSCMR")


class ScribbleBench2DDataset(Dataset):
    """Flat per-slice view over ACDC/MSCMR ``ScribbleBench3DDataset`` volumes.

    Every volume is loaded once at construction time (ACDC/MSCMR are small
    enough -- a few hundred MB total -- that eager loading is simpler and
    faster than re-reading a nii.gz on every ``__getitem__``, and it plays
    well with multi-worker ``DataLoader``s via Linux's fork-based worker
    start, which shares the already-populated cache instead of reloading it).

    Args mirror :class:`ScribbleBench3DDataset`, with ``dataset_name``
    restricted to ACDC/MSCMR.
    """

    def __init__(
        self,
        dataset_name,
        base_dir=None,
        split="train",
        sup_type="scribble",
        return_full_label=False,
        transform=None,
        normalization=None,
        verify_pairs=True,
    ):
        super().__init__()
        dataset_name = dataset_name.upper()
        if dataset_name not in SUPPORTED_2D_DATASETS:
            raise ValueError(
                "ScribbleBench2DDataset only supports {}; WORD is 3D-only in this "
                "benchmark".format(SUPPORTED_2D_DATASETS)
            )
        self.dataset_name = dataset_name
        self.transform = transform

        volumes = ScribbleBench3DDataset(
            dataset_name=dataset_name,
            base_dir=base_dir,
            split=split,
            sup_type=sup_type,
            return_full_label=return_full_label,
            transform=None,
            normalization=normalization,
            verify_pairs=verify_pairs,
        )
        self.class_names = volumes.class_names
        self.num_classes = volumes.num_classes
        self.ignore_index = volumes.ignore_index

        # Public per-volume caches (native resolution, no augmentation
        # applied): every 2D train script's own dataset variant (DMSPS's
        # stage-2 expanded-label swap, VoxTrust-3D's per-slice calibration
        # partition) is built directly on top of these, the same way
        # ``ExpandedLabelDataset``/``VoxTrustPatch3DDataset`` build on
        # ``ScribbleBench3DDataset``'s per-case samples.
        self.images = []
        self.labels = []
        self.gt_labels = []
        self.cases = []
        self.spacings = []
        self.is_scribble = []
        self.slice_index = []
        for volume_index in range(len(volumes)):
            sample = volumes[volume_index]
            image = sample["image"]
            label = sample["label"]
            self.images.append(image)
            self.labels.append(label)
            self.gt_labels.append(sample.get("gt_label"))
            self.cases.append(sample["case"])
            self.spacings.append(sample["spacing"])
            self.is_scribble.append(sample["is_scribble"])
            for slice_index in range(image.shape[0]):
                self.slice_index.append((volume_index, slice_index))

    def __len__(self):
        return len(self.slice_index)

    def slice_positions_for_volumes(self, volume_indices):
        """Flat ``slice_index`` positions belonging to the given case-level
        (volume) indices -- the bridge between ``train.common_3d.
        make_published_split``'s per-case train/val split (computed against
        ``self.cases``, in the same order every ``ScribbleBench3DDataset``
        built with identical args produces) and this dataset's per-slice
        training samples. Every 2D train script needs this to build its
        training ``Subset``.
        """
        wanted = set(volume_indices)
        return [
            position
            for position, (volume_index, _) in enumerate(self.slice_index)
            if volume_index in wanted
        ]

    def __getitem__(self, idx):
        volume_index, slice_index = self.slice_index[idx]
        image = self.images[volume_index][slice_index]
        label = self.labels[volume_index][slice_index]
        gt_label = self.gt_labels[volume_index]
        sample = {
            "image": image,
            "label": label,
            "idx": idx,
            "case": self.cases[volume_index],
            "slice_index": slice_index,
            "spacing": self.spacings[volume_index],
            "num_classes": self.num_classes,
            "ignore_index": self.ignore_index,
            "is_scribble": self.is_scribble[volume_index],
        }
        if gt_label is not None:
            sample["gt_label"] = gt_label[slice_index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def _random_rot_flip_2d(arrays):
    k = random.randint(0, 3)
    axis = random.randint(0, 1)
    return {key: np.flip(np.rot90(value, k), axis=axis).copy() for key, value in arrays.items()}


def _random_rotate_2d(arrays, ignore_index):
    angle = random.randint(-20, 19)
    rotated = {}
    for key, value in arrays.items():
        if key == "image":
            rotated[key] = ndimage.rotate(value, angle, order=0, reshape=False)
        elif key == "label":
            cval = ignore_index if ignore_index in np.unique(value) else 0
            rotated[key] = ndimage.rotate(value, angle, order=0, reshape=False, mode="constant", cval=cval)
        else:
            rotated[key] = ndimage.rotate(value, angle, order=0, reshape=False, mode="constant", cval=0)
    return rotated


class RandomGenerator2D(object):
    """ACDC/MSCMR 2D training augmentation.

    Verified against ``HiLab-git/WSL4MIS``'s ``dataloaders/dataset.py``
    ``RandomGenerator`` and ``HiLab-git/DMSPS``'s ``dataloader/transform_2D.py``
    ``RandomGenerator``: 50% rot90+flip, else 25% random rotate +-20 degrees
    (scribble labels padded with ``ignore_index`` rather than background
    wherever it already appears in the slice, matching the source's
    class-conditional ``cval``), else identity; always resized to
    ``output_size`` via a nearest-neighbor (``order=0``) zoom for the image
    and label alike. Using ``order=0`` for the image too is intentional and
    verified against the source, not a paraphrase bug -- it keeps the
    resampled image and label exactly pixel-aligned.
    """

    def __init__(self, output_size):
        self.output_size = tuple(int(value) for value in output_size)

    def __call__(self, sample):
        keys = [key for key in ("image", "label", "gt_label") if key in sample]
        arrays = {key: sample[key] for key in keys}
        ignore_index = sample["ignore_index"]

        if random.random() > 0.5:
            arrays = _random_rot_flip_2d(arrays)
        elif random.random() > 0.5:
            arrays = _random_rotate_2d(arrays, ignore_index)

        x, y = arrays["image"].shape
        scale = (self.output_size[0] / x, self.output_size[1] / y)
        resized = {key: zoom(value, scale, order=0) for key, value in arrays.items()}

        sample = dict(sample)
        sample["image"] = torch.from_numpy(resized["image"].astype(np.float32)).unsqueeze(0)
        sample["label"] = torch.from_numpy(resized["label"].astype(np.int64)).long()
        if "gt_label" in resized:
            sample["gt_label"] = torch.from_numpy(resized["gt_label"].astype(np.int64)).long()
        return sample


class ToTensor2D(object):
    """No-op-augmentation counterpart to :class:`RandomGenerator2D`, used for
    validation/test slices that must keep their native resolution (resizing
    happens per-slice at inference time instead, see ``train/common_2d.py``).
    """

    def __call__(self, sample):
        sample = dict(sample)
        image = np.ascontiguousarray(sample["image"], dtype=np.float32)
        label = np.ascontiguousarray(sample["label"], dtype=np.int64)
        sample["image"] = torch.from_numpy(image).unsqueeze(0)
        sample["label"] = torch.from_numpy(label).long()
        if "gt_label" in sample:
            gt_label = np.ascontiguousarray(sample["gt_label"], dtype=np.int64)
            sample["gt_label"] = torch.from_numpy(gt_label).long()
        return sample
