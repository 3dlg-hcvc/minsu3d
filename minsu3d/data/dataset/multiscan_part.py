from minsu3d.data.dataset import GeneralDataset


class MultiScanPart(GeneralDataset):
    def __int__(self, cfg, split):
        super().__init__(cfg, split)
