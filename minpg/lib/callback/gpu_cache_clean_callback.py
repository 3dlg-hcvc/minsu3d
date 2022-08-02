from pytorch_lightning.callbacks import Callback
import torch


class GPUCacheCleanCallback(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, unused=0):
        torch.cuda.empty_cache()

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        torch.cuda.empty_cache()

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        torch.cuda.empty_cache()

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        torch.cuda.empty_cache()
