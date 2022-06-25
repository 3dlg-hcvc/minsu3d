from pytorch_lightning.callbacks import ModelCheckpoint


def init_checkpoint_monitor(cfg):
    if cfg.train.save_checkpoint_every_n_epochs < 0:
        # save the best k checkpoints
        save_top_k = cfg.train.save_checkpoint_every_n_epochs * -1
        every_n_epochs = None
        monitor = cfg.train.monitor
    else:
        # save checkpoints every n epochs
        save_top_k = -1
        every_n_epochs = cfg.train.save_checkpoint_every_n_epochs
        monitor = None

    ckpt_base_name = f"{cfg.general.model}-{cfg.general.dataset}"

    monitor = ModelCheckpoint(
        monitor=monitor,
        mode="max",
        dirpath=cfg.general.root,
        filename=ckpt_base_name + "-{epoch}",
        save_top_k=save_top_k,
        every_n_epochs=every_n_epochs,
        save_last=True
    )
    return monitor
