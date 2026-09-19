"""ACDC/MSCMR "expert scribble" 2D dataset -- the original WSL4MIS/CycleMix
h5 archive (``<repo_root>/data/ACDC``, ``<repo_root>/data/MSCMR``), kept
alongside (not instead of) the NIfTI-based ``dataset/ScribbleBench``
pipeline (``scribblebench_2d.py``/``scribblebench_3d.py``).

This is a *different on-disk source* for the same two anatomies, not a
different benchmark: the label convention is identical (4 classes --
background/RV/MYO/LV, ``ignore_index=4``, see ``scribblebench_3d.DATASET_CONFIGS``,
reused here directly) and the published train/val/test patient split
(``train/legacy_splits.py``) is the same protocol ScribbleBench already uses
(ScribFormer's ACDC MAAGfold70; CycleMix's MSCMR 23/5/15) -- reused here too,
for direct cross-source comparability. What differs is the literal annotation
file and preprocessing: this archive stores already length-1902/382-slice,
min-max-normalized-to-``[0, 1]`` 2D h5 slices (``image``/``label``/``scribble``
keys) instead of z-score-normalized NIfTI volumes, and its scribbles are the
original expert-drawn strokes from the WSL4MIS/CycleMix papers rather than
ScribbleBench's own scribble source.

Two dataset classes, mirroring ``scribblebench_2d.py``/``scribblebench_3d.py``'s
split into a flat per-slice train view and a per-case dense-labeled volume
view:

- :class:`ExpertScribble2DDataset` -- flat per-slice train split, built
  directly from ``<DATASET>_training_slices/*.h5`` (already 2D, so unlike
  ``ScribbleBench2DDataset`` there is no 3D volume to unroll -- slices for one
  case are simply re-stacked in slice order). Deliberately implements the
  *exact same public attributes* as ``ScribbleBench2DDataset``
  (``images``/``labels``/``cases``/``spacings``/``is_scribble``/``slice_index``/
  ``num_classes``/``ignore_index``, plus ``slice_positions_for_volumes``) so
  every existing per-method 2D dataset wrapper (``ExpandedLabel2DDataset`` in
  ``train_dmsps_2d.py``, ``VoxTrustSlice2DDataset`` in
  ``train_voxtrust3d_2d.py``) can be reused completely unchanged, just fed
  this class instead.
- :class:`ExpertScribbleVolumeDataset` -- per-case ``(D, H, W)`` volumes with
  dense ``gt_label``, for validation (``ACDC_training_volumes`` filtered to
  the published val patients; MSCMR's own dedicated
  ``MSCMR_validation_volumes`` folder) and test (``ACDC_training_volumes``
  filtered to the published test patients; MSCMR's own dedicated
  ``MSCMR_testing_volumes`` folder) -- the 2D-pipeline counterpart of
  ``ScribbleBench3DDataset(..., return_full_label=True)``/``split="test"``.

No voxel spacing is stored anywhere in this archive (unlike the NIfTI
pipeline's header-derived spacing): every sample reports a placeholder
isotropic ``(1.0, 1.0, 1.0)`` spacing. This is harmless for training (only
``utils.voxtrust3d``'s physical transfer distance actually consumes spacing,
and degrades gracefully to a pixel distance), but means HD95/ASSD reported by
``test/test_pce_2d_expert.py``/``test/test_dmsps_2d_expert.py`` are in pixel
units, not calibrated millimeters -- read them as such.
"""

import re
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset

from .scribblebench_3d import DATASET_CONFIGS

SUPPORTED_EXPERT_DATASETS = ("ACDC", "MSCMR")

# case: everything before "_slice_<N>"; ACDC keeps its "_frameNN" suffix,
# MSCMR keeps its "_DE" suffix -- both are exactly what
# train.common_3d.patient_id() already knows how to strip back to a bare
# patient/subject id, so no dataset-specific parsing is needed downstream.
_SLICE_NAME_PATTERN = re.compile(r"^(?P<case>.+)_slice_(?P<slice>\d+)$")

_PLACEHOLDER_SPACING = np.array([1.0, 1.0, 1.0], dtype=np.float32)


def _default_base_dir(dataset_name):
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / dataset_name


def _resolve_base_dir(dataset_name, base_dir):
    path = Path(base_dir).expanduser().resolve() if base_dir else _default_base_dir(dataset_name)
    if not path.is_dir():
        raise FileNotFoundError("Could not find expert-scribble {} archive at {}".format(dataset_name, path))
    return path


class ExpertScribble2DDataset(Dataset):
    """Flat per-slice train split over the expert-scribble h5 archive.

    Args mirror :class:`dataloader.scribblebench_2d.ScribbleBench2DDataset`
    (minus ``return_full_label``, ``normalization`` and ``verify_pairs``,
    which that class only ever uses for its ``split="train"`` flat view in
    ways no training script actually exercises -- see this module's tests).
    """

    def __init__(self, dataset_name, base_dir=None, split="train", sup_type="scribble", transform=None):
        super().__init__()
        dataset_name = dataset_name.upper()
        if dataset_name not in SUPPORTED_EXPERT_DATASETS:
            raise ValueError("ExpertScribble2DDataset only supports {}".format(SUPPORTED_EXPERT_DATASETS))
        if split != "train":
            raise ValueError(
                "ExpertScribble2DDataset only serves the flat per-slice train split; "
                "use ExpertScribbleVolumeDataset for val/test"
            )
        if sup_type not in ("scribble", "label", "dense"):
            raise ValueError("sup_type must be one of: scribble, label, dense")

        self.dataset_name = dataset_name
        self.sup_type = sup_type
        self.transform = transform
        config = DATASET_CONFIGS[dataset_name]
        self.class_names = config["class_names"]
        self.num_classes = len(self.class_names)
        self.ignore_index = config["ignore_index"]

        base_dir = _resolve_base_dir(dataset_name, base_dir)
        slices_folder = base_dir / "{}_training_slices".format(dataset_name)
        if not slices_folder.is_dir():
            raise FileNotFoundError("Missing {}".format(slices_folder))

        by_case = {}
        for path in sorted(slices_folder.glob("*.h5")):
            match = _SLICE_NAME_PATTERN.match(path.stem)
            if not match:
                raise ValueError("Unexpected {} slice filename: {}".format(dataset_name, path.name))
            by_case.setdefault(match.group("case"), []).append((int(match.group("slice")), path))

        self.images = []
        self.labels = []
        self.cases = []
        self.spacings = []
        self.is_scribble = []
        self.slice_index = []
        label_key = "scribble" if sup_type == "scribble" else "label"
        for case in sorted(by_case):
            entries = sorted(by_case[case], key=lambda item: item[0])
            slice_ids = [slice_id for slice_id, _ in entries]
            if slice_ids != list(range(len(entries))):
                raise ValueError(
                    "Case {} has non-contiguous slice indices {}; expected 0..{}".format(
                        case, slice_ids, len(entries) - 1
                    )
                )
            images, labels = [], []
            for _, path in entries:
                with h5py.File(path, "r") as handle:
                    images.append(np.asarray(handle["image"], dtype=np.float32))
                    labels.append(np.asarray(handle[label_key], dtype=np.int64))
            volume_index = len(self.images)
            self.images.append(np.stack(images, axis=0))
            self.labels.append(np.stack(labels, axis=0))
            self.cases.append(case)
            self.spacings.append(_PLACEHOLDER_SPACING.copy())
            self.is_scribble.append(sup_type == "scribble")
            for slice_position in range(len(entries)):
                self.slice_index.append((volume_index, slice_position))

    def __len__(self):
        return len(self.slice_index)

    def slice_positions_for_volumes(self, volume_indices):
        """Identical contract to ``ScribbleBench2DDataset``'s method of the
        same name -- see its docstring."""
        wanted = set(volume_indices)
        return [
            position
            for position, (volume_index, _) in enumerate(self.slice_index)
            if volume_index in wanted
        ]

    def __getitem__(self, idx):
        volume_index, slice_index = self.slice_index[idx]
        sample = {
            "image": self.images[volume_index][slice_index],
            "label": self.labels[volume_index][slice_index],
            "idx": idx,
            "case": self.cases[volume_index],
            "slice_index": slice_index,
            "spacing": self.spacings[volume_index],
            "num_classes": self.num_classes,
            "ignore_index": self.ignore_index,
            "is_scribble": self.is_scribble[volume_index],
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class ExpertScribbleVolumeDataset(Dataset):
    """Per-case dense-labeled ``(D, H, W)`` volumes, for validation/test.

    ACDC has one archive of volumes (``ACDC_training_volumes``, all 100
    patients, all with dense ``label``) shared by both ``split="val"`` and
    ``split="test"`` -- exactly like ``build_val_dataset()`` in
    ``train_pce_2d.py`` returns every ScribbleBench case and lets the caller
    filter by the split it actually wants; MSCMR instead ships one dedicated
    folder per split (``MSCMR_validation_volumes``, ``MSCMR_testing_volumes``),
    each already containing exactly its published patient set. Either way,
    the caller is expected to filter ``.cases`` against
    ``train.legacy_splits.published_groups``/``published_test_groups`` (see
    ``train/train_pce_2d_expert.py``'s ``resolve_case_split``/
    ``resolve_test_split``), matching this repo's usual "the loader serves
    everything on disk; the training/eval script owns the split" convention.
    """

    def __init__(self, dataset_name, base_dir=None, split="val"):
        super().__init__()
        dataset_name = dataset_name.upper()
        if dataset_name not in SUPPORTED_EXPERT_DATASETS:
            raise ValueError("ExpertScribbleVolumeDataset only supports {}".format(SUPPORTED_EXPERT_DATASETS))
        if split not in ("val", "test"):
            raise ValueError("split must be 'val' or 'test'")

        self.dataset_name = dataset_name
        self.split = split
        config = DATASET_CONFIGS[dataset_name]
        self.class_names = config["class_names"]
        self.num_classes = len(self.class_names)
        self.ignore_index = config["ignore_index"]

        base_dir = _resolve_base_dir(dataset_name, base_dir)
        if dataset_name == "ACDC":
            folder = base_dir / "ACDC_training_volumes"
        else:
            folder = base_dir / ("MSCMR_validation_volumes" if split == "val" else "MSCMR_testing_volumes")
        if not folder.is_dir():
            raise FileNotFoundError("Missing {}".format(folder))

        self.paths = sorted(folder.glob("*.h5"))
        if not self.paths:
            raise FileNotFoundError("No .h5 volumes found in {}".format(folder))
        self.cases = [path.stem for path in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with h5py.File(self.paths[index], "r") as handle:
            image = np.asarray(handle["image"], dtype=np.float32)
            label = np.asarray(handle["label"], dtype=np.int64)
        return {
            "image": image,
            "gt_label": label,
            "label": label,
            "case": self.cases[index],
            "spacing": _PLACEHOLDER_SPACING.copy(),
        }
