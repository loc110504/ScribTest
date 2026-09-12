import random

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from utils.xnetv2_wavelet import build_wavelet_views


class WaveletTrainingWrapper(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        sample = dict(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample


class WaveletRandomGenerator(object):
    def __init__(self, output_size, wavelet_type="haar", alpha=(0.0, 0.4), beta=(0.0, 0.4), ignore_index=4):
        self.output_size = output_size
        self.wavelet_type = wavelet_type
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index

    @staticmethod
    def _random_rot_flip(*arrays):
        k = np.random.randint(0, 4)
        arrays = [np.rot90(array, k) for array in arrays]
        axis = np.random.randint(0, 2)
        arrays = [np.flip(array, axis=axis).copy() for array in arrays]
        return arrays

    def _random_rotate(self, image, label, low, high, gt_label=None):
        angle = np.random.randint(-20, 20)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        low = ndimage.rotate(low, angle, order=0, reshape=False)
        high = ndimage.rotate(high, angle, order=0, reshape=False)
        cval = self.ignore_index if self.ignore_index in np.unique(label) else 0
        label = ndimage.rotate(label, angle, order=0, reshape=False, mode="constant", cval=cval)
        if gt_label is not None:
            gt_label = ndimage.rotate(gt_label, angle, order=0, reshape=False, mode="constant", cval=0)
        return image, label, low, high, gt_label

    def _resize_image(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)

    def __call__(self, sample):
        image = sample["image"]
        label = sample["label"]
        gt_label = sample.get("gt_label")
        low, high = build_wavelet_views(
            image,
            wavelet_type=self.wavelet_type,
            alpha=self.alpha,
            beta=self.beta,
        )

        if random.random() > 0.5:
            arrays = [image, label, low, high]
            if gt_label is not None:
                arrays.append(gt_label)
            arrays = self._random_rot_flip(*arrays)
            image, label, low, high = arrays[:4]
            if gt_label is not None:
                gt_label = arrays[4]
        elif random.random() > 0.5:
            image, label, low, high, gt_label = self._random_rotate(image, label, low, high, gt_label)

        image = self._resize_image(image)
        label = self._resize_image(label)
        low = self._resize_image(low)
        high = self._resize_image(high)

        sample_out = {
            "image": torch.from_numpy(image.astype(np.float32)).unsqueeze(0),
            "label": torch.from_numpy(label.astype(np.uint8)),
            "wavelet_l": torch.from_numpy(low.astype(np.float32)).unsqueeze(0),
            "wavelet_h": torch.from_numpy(high.astype(np.float32)).unsqueeze(0),
        }
        if gt_label is not None:
            gt_label = self._resize_image(gt_label)
            sample_out["gt_label"] = torch.from_numpy(gt_label.astype(np.uint8))
        return sample_out
