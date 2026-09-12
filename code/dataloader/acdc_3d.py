"""3D ScribbleBench ACDC dataset."""

from .scribblebench_3d import ScribbleBench3DDataset


class ACDC3DDataSets(ScribbleBench3DDataset):
    def __init__(self, base_dir=None, **kwargs):
        super().__init__(dataset_name="ACDC", base_dir=base_dir, **kwargs)


ACDC3DDataset = ACDC3DDataSets
