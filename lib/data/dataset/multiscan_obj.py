from lib.data.dataset import GeneralDataset


class MultiScanObj(GeneralDataset):
    def __int__(self, cfg, split):
        super().__init__(cfg, split)
