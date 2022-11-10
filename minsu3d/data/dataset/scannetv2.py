from minsu3d.data.dataset import GeneralDataset


class ScanNetv2(GeneralDataset):
    def __int__(self, cfg, split):
        super().__init__(cfg, split)
