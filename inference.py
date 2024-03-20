import os
import torch

import numpy as np
import lightning.pytorch as pl

from src.globals import *
from src.evaluations import evaluate
from src.model import GenomicDataset, scGrapHiC
from src.utils import initialize_parameters_from_args

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint



PARAMETERS = initialize_parameters_from_args()
pl.seed_everything(PARAMETERS['seed'])

print(PARAMETERS)

test_dataset = GenomicDataset(
    os.path.join(MOUSE_PROCESSED_DATA_HIRES, 'test.npz'),
    PARAMETERS
)

ood_dataset = GenomicDataset(
    os.path.join(MOUSE_PROCESSED_DATA_HIRES, 'ood.npz'),
    PARAMETERS
)

test_data_loader = torch.utils.data.DataLoader(test_dataset, PARAMETERS['batch_size'], shuffle=False)
ood_data_loader = torch.utils.data.DataLoader(ood_dataset, PARAMETERS['batch_size'], shuffle=False)

tb_logger = TensorBoardLogger("logs", name=PARAMETERS['experiment'])
checkpoint_callback = ModelCheckpoint(monitor="valid/SCC",  save_top_k=3, mode='max')
scgraphic = scGrapHiC(PARAMETERS)

trainer = pl.Trainer(
    accelerator='cpu',
    max_epochs=PARAMETERS['epochs'], 
    check_val_every_n_epoch=50, 
    logger=tb_logger,
    deterministic=True,
    callbacks=[checkpoint_callback],
    gradient_clip_val=PARAMETERS['gradient_clip_value'],
    profiler="simple"
)

ckpt_path = os.path.join(MODEL_WEIGHTS, 'scgraphic')

checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
scgraphic.load_state_dict(checkpoint['state_dict'])

trainer.test(scgraphic, test_data_loader) 
trainer.test(scgraphic, ood_data_loader)

evaluate(os.path.join(RESULTS, PARAMETERS['experiment']), PARAMETERS)
