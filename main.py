import os
import torch

import lightning.pytorch as pl

from src.globals import *
from src.model import GenomicDataset, scGrapHiC, scGrapHiC_AB, scGrapHiC_TAD
from src.dataset_creator import create_schires_dataset
from src.download_datasets import create_directory_structure
from src.preprocess_datasets import preprocess_hires_datasets
from src.pseudobulk import create_pseudobulk_files

create_directory_structure()
create_pseudobulk_files(HIRES_BRAIN_METADATA_FILE)
# read_metadata_file(HIRES_EMBRYO_METADATA_FILE)


# PARAMETERS = {
#     # Dataset parameters
#     'resolution': 1000000,
#     'library_size': 100000,
#     'normalize_umi': False, 
#     'normalize_track': True,
    
#     # Shared Parameters
#     'rna_seq_tracks' : 2,
#     'pos_encodings_dim': 2,
#     'max_number_of_nodes': 256,
#     'batch_size': 32,
#     'bulk_hic': 'pn5_zygote',
    
#     # Model Parameters
#     'conv1d_kernel_size': 32, 
#     'encoder_hidden_embedding_size': 32,
#     'num_encoder_attn_blocks': 8,
#     'num_heads_encoder_attn_blocks': 1,
#     'num_decoder_residual_blocks': 8,
#     'width': 7
# }




# # preprocess_hires_datasets(PARAMETERS)
# # create_schires_dataset(PARAMETERS)



# train_dataset = GenomicDataset(
#     os.path.join(MOUSE_PROCESSED_DATA_HIRES, 'train.npz')
# )

# test_dataset = GenomicDataset(
#     os.path.join(MOUSE_PROCESSED_DATA_HIRES, 'test.npz')
# )

# train_data_loader =  torch.utils.data.DataLoader(train_dataset, PARAMETERS['batch_size'], shuffle=True)
# validation_data_loader =  torch.utils.data.DataLoader(test_dataset, PARAMETERS['batch_size'], shuffle=False)

# scgraphic = scGrapHiC_TAD(PARAMETERS)

# trainer = pl.Trainer(max_epochs=300, check_val_every_n_epoch=30)
# trainer.fit(scgraphic, train_data_loader, validation_data_loader)

















# cell_by_gene_matrix = read_cell_by_gene_matrix('data/mm10/HiRES/scRNA-seq/GSE223917_HiRES_brain.rna.umicount.tsv.gz')
# # cell_by_gene_matrix = normalize_cell_by_gene_matrix(cell_by_gene_matrix)
# coordinate_matrix = convert_cell_by_gene_to_coordinate_matrix(cell_by_gene_matrix, 'data/mm10/gene_coordinates.csv')



# generate_scRNAseq_coordiante_tracks(coordinate_matrix, 'data/mm10/HiRES/scRNA-seq/', 'data/mm10/chrom.sizes')

# cooler_files_from_pairix_file(
#     'data/mm10/HiRES/scHi-C/GasaE751001/GSM6998595_GasaE751001.pairs.gz',
#     'data/mm10/HiRES/scHi-C/GasaE751001/',
#     'data/mm10/chrom.sizes'
# )

# matrix =  extract_dense_matrix_from_cooler_file('data/mm10/HiRES/scHi-C/GasaE751001/chr1.cool')
# #matrix = matImpute(matrix, 1, 0.5)
# matrix = matrix[:100, :100]

# visualize_hic_contact_matrix(
# 	matrix, 
# 	'matrix.png',
# 	False
# )
