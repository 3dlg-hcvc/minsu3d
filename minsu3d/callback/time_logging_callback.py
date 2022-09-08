from pytorch_lightning.callbacks import Callback
import logging
import time


class TimeLoggingCallback(Callback):

    def setup(self, trainer, pl_module, stage=None):
        self.custom_logger = logging.getLogger("pytorch_lightning")
        self.training_start_time_memory = None
        self.validation_start_time_memory = None
        self.validation_epoch_duration = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.training_start_time_memory = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_duration = time.time() - self.training_start_time_memory - self.validation_epoch_duration
        self.custom_logger.info(f" Epoch training time: {epoch_duration:.2f}s")
        self.training_start_time_memory = None
        self.validation_epoch_duration = 0

    def on_validation_epoch_start(self, trainer, pl_module):
        self.validation_start_time_memory = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        self.validation_epoch_duration = time.time() - self.validation_start_time_memory
        self.custom_logger.info(f" Epoch validation time: {self.validation_epoch_duration:.2f}s")
        self.validation_start_time_memory = None
