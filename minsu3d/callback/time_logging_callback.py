from pytorch_lightning.callbacks import Callback
import logging
import time


class TimeLoggingCallback(Callback):

    def setup(self, trainer, pl_module, stage=None):
        self.loger = logging.getLogger('lightning')
        self.start_time_memory = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time_memory = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_duration = time.time() - self.start_time_memory
        self.loger.info(f"Training time for epoch {trainer.current_epoch}: {epoch_duration}")

    def on_validation_epoch_start(self, trainer, pl_module):
        self.start_time_memory = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch_duration = time.time() - self.start_time_memory
        self.loger.info(f"Validation time: {epoch_duration}")
