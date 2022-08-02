from minpg.lib.data.dataset import GeneralDataset


class ScanNet(GeneralDataset):
    def __int__(self, cfg, split):
        super().__init__(cfg, split)
