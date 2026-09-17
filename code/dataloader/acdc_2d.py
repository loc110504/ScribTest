"""2D per-slice ScribbleBench ACDC dataset."""

from .scribblebench_2d import ScribbleBench2DDataset


class ACDC2DDataSets(ScribbleBench2DDataset):
    def __init__(self, base_dir=None, **kwargs):
        super().__init__(dataset_name="ACDC", base_dir=base_dir, **kwargs)


ACDC2DDataset = ACDC2DDataSets
