"""2D per-slice ScribbleBench MSCMR dataset."""

from .scribblebench_2d import ScribbleBench2DDataset


class MSCMR2DDataSets(ScribbleBench2DDataset):
    def __init__(self, base_dir=None, **kwargs):
        super().__init__(dataset_name="MSCMR", base_dir=base_dir, **kwargs)


MSCMR2DDataset = MSCMR2DDataSets
