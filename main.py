import os
import torch

import numpy as np
import lightning.pytorch as pl

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint



from src.globals import *
from src.evaluations import *
from src.visualizations import *
from src.normalizations import *
from src.dataset_creator import *
from src.download_datasets import *
from src.preprocess_datasets import *

from src.model import GenomicDataset, scGrapHiC
from src.pseudobulk import create_pseudobulk_files, parse_metadata
from src.utils import initialize_parameters_from_args, read_npy_file

PARAMETERS = initialize_parameters_from_args()
pl.seed_everything(PARAMETERS['seed'])

print(PARAMETERS)


# These functions download all the required datasets and setup the directory structure to contain these datasets
create_directory_structure()
download_hires_schic_datasets()
download_hires_scrnaseq_datasets()


# This function parsses the HIRES dataset and prepares it for pseudo-bulking
preprocess_hires_datasets(PARAMETERS)

# These functions then create the pseudo-bulked datasets from the HiRES datasets
create_pseudobulk_files(HIRES_BRAIN_METADATA_FILE)
create_pseudobulk_files(HIRES_EMBRYO_METADATA_FILE)


exclusion_set = ['EX15', 'brain']
create_schic_pseudobulk_dataset(exclusion_set, PARAMETERS, 'train', PARAMETERS['experiment'])
create_schic_pseudobulk_dataset(exclusion_set, PARAMETERS, 'valid', PARAMETERS['experiment'])
create_schic_pseudobulk_dataset(exclusion_set, PARAMETERS, 'test', PARAMETERS['experiment'])

exclusion_set = ['E70', 'E75', 'E80', 'E85', 'E95', 'EX05']
create_schic_pseudobulk_dataset(exclusion_set, PARAMETERS, 'ood', PARAMETERS['experiment'])

train_dataset = GenomicDataset(
    os.path.join(MOUSE_PROCESSED_DATA_HIRES, '{}_train.npz'.format(PARAMETERS['experiment'])),
    PARAMETERS
)
valid_dataset = GenomicDataset(
    os.path.join(MOUSE_PROCESSED_DATA_HIRES, '{}_valid.npz'.format(PARAMETERS['experiment'])),
    PARAMETERS
)
test_dataset = GenomicDataset(
    os.path.join(MOUSE_PROCESSED_DATA_HIRES, '{}_test.npz'.format(PARAMETERS['experiment'])),
    PARAMETERS
)

ood_dataset = GenomicDataset(
    os.path.join(MOUSE_PROCESSED_DATA_HIRES, '{}_ood.npz'.format(PARAMETERS['experiment'])),
    PARAMETERS
)

train_data_loader =  torch.utils.data.DataLoader(train_dataset, PARAMETERS['batch_size'], shuffle=True)
validation_data_loader =  torch.utils.data.DataLoader(valid_dataset, PARAMETERS['batch_size'], shuffle=False)
test_data_loader = torch.utils.data.DataLoader(test_dataset, PARAMETERS['batch_size'], shuffle=False)
ood_data_loader = torch.utils.data.DataLoader(ood_dataset, PARAMETERS['batch_size'], shuffle=False)

tb_logger = TensorBoardLogger("logs", name=PARAMETERS['experiment'])
checkpoint_callback = ModelCheckpoint(monitor="valid/SCC",  save_top_k=3, mode='max')
scgraphic = scGrapHiC(PARAMETERS)

trainer = pl.Trainer(
    max_epochs=PARAMETERS['epochs'], 
    check_val_every_n_epoch=50, 
    logger=tb_logger,
    deterministic=True,
    callbacks=[checkpoint_callback],
    gradient_clip_val=PARAMETERS['gradient_clip_value'],
)

trainer.fit(scgraphic, train_data_loader, validation_data_loader)

trainer.test(scgraphic, test_data_loader) 
trainer.test(scgraphic, ood_data_loader)


# evaluate(os.path.join(RESULTS, PARAMETERS['experiment']), PARAMETERS)
