from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class MyEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        self._run_early_stopping_check(trainer)