"""3D ScribbleBench MSCMR dataset."""

from .scribblebench_3d import ScribbleBench3DDataset


class MSCMR3DDataSets(ScribbleBench3DDataset):
    def __init__(self, base_dir=None, **kwargs):
        super().__init__(dataset_name="MSCMR", base_dir=base_dir, **kwargs)


MSCMR3DDataset = MSCMR3DDataSets
