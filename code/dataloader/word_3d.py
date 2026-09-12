"""3D ScribbleBench WORD dataset."""

from .scribblebench_3d import ScribbleBench3DDataset


class WORD3DDataSets(ScribbleBench3DDataset):
    def __init__(self, base_dir=None, **kwargs):
        super().__init__(dataset_name="WORD", base_dir=base_dir, **kwargs)


WORD3DDataset = WORD3DDataSets
