from minsu3d.data.dataset import GeneralDataset


class PartNetSim(GeneralDataset):
    def __int__(self, cfg, split):
        super().__init__(cfg, split)