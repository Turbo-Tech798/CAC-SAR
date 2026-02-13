from share import *
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from ASC.z_mstar_ratio_main import asc_extract, asc_extract_batch

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    class SetStartingEpoch(Callback):
        def __init__(self, starting_epoch):
            self.starting_epoch = starting_epoch

        def on_train_start(self, trainer, pl_module):
            trainer.fit_loop.epoch_progress.current.completed = self.starting_epoch


    def test_asc_extract():
        dummy_abs = np.random.rand(10, 128, 128).astype(np.float32)
        dummy_phase = np.random.rand(10, 128, 128).astype(np.float32)

        results = asc_extract_batch(dummy_abs, dummy_phase)
        assert len(results) == 10
        print("go on")
 
    # Configs
    # resume_path = 'D:/zx/lora_sd21/control_sd21_lora_ini.ckpt'
    resume_path = 'E:/zx/enhance_datasets_asc/model_epoch_210.ckpt'
    batch_size = 64
    logger_freq = 4170
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v21.yaml').cpu() 
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    # print(model)

    # Misc
    dataset = MyDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq, start_epoch=210, save_model_frequency=1)


    set_starting_epoch_callback = SetStartingEpoch(starting_epoch=210)

    trainer = Trainer(
        gpus=1,
        precision=32,
        callbacks=[logger, set_starting_epoch_callback],
        devices='cuda:0',
        check_val_every_n_epoch=5,
        max_epochs=250,
    )
    trainer.fit(model, dataloader)
