from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, start_epoch, **kw):
        self.start_epoch = start_epoch
        super().__init__(**kw)

    def _should_skip_saving_checkpoint(self, trainer: "pl.Trainer") -> bool:
        from pytorch_lightning.trainer.states import TrainerFn

        return (
            bool(trainer.fast_dev_run)  # disable checkpointing with fast_dev_run
            or trainer.state.fn != TrainerFn.FITTING  # don't save anything during non-fit
            or trainer.sanity_checking  # don't save anything during sanity check
            or self._last_global_step_saved == trainer.global_step  # already saved at the last step
            or trainer.current_epoch + 1 <= self.start_epoch  # don't save before the start_epoch
        )
