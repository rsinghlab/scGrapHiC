import os
import torch
import json
import argparse

import numpy as np
import pandas as pd

from src.globals import DATASET_LABELS_JSON


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_directory(path):
	if not os.path.exists(path):
		try:
			# Create the directory
			os.makedirs(path)
		except OSError as e:
			print(f"Error: {e}")



def read_chromsizes_file(path):
    data = open(path).read().split('\n')
    data = list(map(lambda x: x.split(' '), data))
    data = list(map(lambda x: [x[0], int(x[1])], data))

    return dict(data)




def chrom_bins(chr, chr_size, resolution):
    size = chr_size // resolution + 1

    chr_names = np.array([chr]*size)
    starts = (np.arange(0, size, 1, dtype=int))
    ends = (np.arange(1, size+1, 1, dtype=int))
    
    bins = {
        'chrom': chr_names,
        'start': starts,
        'end': ends
    }

    bins = pd.DataFrame(data=bins) 
    
    return bins


def get_gene_name(attributes):
    attributes = attributes.split(';')
    attribute = list(filter(lambda x: 'gene_name' in x, attributes))[0]
    attribute = attribute.split('=')[-1]
    return attribute









def get_file_name_parameters(path):
    parameters = path.split('/')[-1].split('_')    
    idx = 0
    stage = None
    if len(parameters) > 4:
        # We have embryo files
        stage = parameters[idx]
        idx=1
    
    tissue = parameters[idx]
    cell_type = parameters[idx+1]
    num_cells = int(parameters[idx+2].split('n')[-1])

    return stage, tissue, cell_type, num_cells





def divide_signal(signal, chr, cropping_params):
    result = []
    index = []

    stride = cropping_params['stride']
    chunk_size = cropping_params['num_nodes']
    padding = cropping_params['padding']

    if (stride < chunk_size and padding):
        pad_len = (chunk_size - stride) // 2
        signal = np.pad(signal, ((pad_len,pad_len), (0, 0)), 'constant')
    
    size = signal.shape[0]

    # mat's shape changed, update!
    for i in range(0, size, stride):
        if (i+chunk_size)<size:
            subImage = signal[i:i+chunk_size, :]
            result.append([subImage])
            index.append((int(chr), int(size), int(i)))
    
    result = np.array(result, dtype=float)
    index = np.array(index, dtype=int)

    return result, index




def divide_matrix(mat, chr_num, cropping_params):
    """
        @params: mat <np.array> HiC matrix that needs to be chunked up
        @params: chr_num <string> Chromosome number of the input HiC matrix
        @params: cropping_params <dict> contains the required parameters to crop the matrix
        @returns: list<np.array>, list first return is chunked up matrices in a list and second 
                return contains the positions 
    """
    result = []
    index = []
    size = mat.shape[0]
    
    stride = cropping_params['stride']
    chunk_size = cropping_params['num_nodes']
    bound = cropping_params['bounds']
    padding = cropping_params['padding']

    if (stride < chunk_size and padding):
        pad_len = (chunk_size - stride) // 2
        mat = np.pad(mat, ((pad_len,pad_len), (pad_len,pad_len)), 'constant')
    
    # mat's shape changed, update!
    height, width = mat.shape
        
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            if abs(i-j)<=bound and (i+chunk_size<height and j+chunk_size<width):
                subImage = mat[i:i+chunk_size, j:j+chunk_size]
                result.append([subImage])
                index.append((int(chr_num), int(size), int(i), int(j)))
    result = np.array(result)
    index = np.array(index)
    
    return result, index




def compactM(matrix, compact_idx, verbose=False):
    """
        Compacting matrix according to the index list.
        @params: matrix <np.array> Full sized matrix, that needs to be compressed
        @params: compact_idx <list> Indexes of rows that contain data
        @params: verbose <boolean> Debugging print statements
        @returns: <np.array> Condesed matrix with zero arrays pruned 
    """

    compact_size = len(compact_idx)
    
    result = np.zeros((compact_size, compact_size)).astype(matrix.dtype)
    
    if verbose: print('Compacting a', matrix.shape, 'shaped matrix to', result.shape, 'shaped!')
    
    for i, idx in enumerate(compact_idx):
        result[i, :] = matrix[idx][compact_idx]
    
    return result


def get_node_features(encoding):
    return encoding.to(torch.float)

def get_edge_attrs(matrix):
    edge_indices = torch.nonzero(matrix)
    edge_features = matrix[edge_indices]
    edge_features = edge_features.reshape(-1, 1)
    
    return edge_features.to(torch.float)

def get_edge_indexes(matrix):
    edge_indices = torch.nonzero(matrix)
    edge_indices = edge_indices.t().to(torch.long).view(2, -1)
    
    return edge_indices



def add_dataset(stage, tissue, cell_type):
    if not stage:
        stage = 'brain'
    
    cell_type = cell_type.replace(' ', '_')
    
    with open(DATASET_LABELS_JSON, 'r') as openfile:
        # Reading from json file
        dataset_object = json.load(openfile)
        
        if stage not in dataset_object['stage'].keys():
            values = dataset_object['stage'].values()
            
            if len(values) == 0:
                dataset_object['stage'][stage] = 0
            else:
                dataset_object['stage'][stage] = max(values) + 1

        if tissue not in dataset_object['tissue'].keys():
            values = dataset_object['tissue'].values()
            if len(values) == 0:
                dataset_object['tissue'][tissue] = 0
            else:
                dataset_object['tissue'][tissue] = max(values) + 1

        if cell_type not in dataset_object['cell_type'].keys():
            values = dataset_object['cell_type'].values()
            if len(values) == 0:
                dataset_object['cell_type'][cell_type] = 0
            else:
                dataset_object['cell_type'][cell_type] = max(values) + 1


    with open(DATASET_LABELS_JSON, "w") as outfile: 
        json.dump(dataset_object, outfile, indent=4)





def read_npy_file(filepath):
    return np.load(filepath)




def initialize_parameters_from_args():
    parser = argparse.ArgumentParser(description='Initialize parameters from command line arguments')
    
    # Experiment parameters
    parser.add_argument('--experiment', type=str, default='scGrapHiC', help='Experiment name')
    
    # Dataset parameters
    parser.add_argument('--resolution', type=int, default=50000, help='Resolution')
    parser.add_argument('--library_size', type=float, default=25000, help='Library size')
    parser.add_argument('--normalize_umi', type=str2bool, default=False, help='Normalize the cell-by-gene matrix')
    parser.add_argument('--normalize_track', type=str2bool, default=True, help='Normalize the genome track')
    parser.add_argument('--num_cells_cutoff', type=int, default=190, help='Cutoff on number of cell required to be considered for testing/evaluation purposes')
    
    # Hi-C normalization parameters
    parser.add_argument('--normalization_algorithm', type=str, default='library_size_normalization', 
            help='Which Hi-C normalization algorithm to use', 
            choices=['log2_norm', 'log10_norm', 'zscore_norm', 'sqrt_norm', 'library_size_normalization', 'none'])
    parser.add_argument('--hic_smoothing', type=str2bool, default=True, help='Use Hi-C soft-threshold eigenvalue smoothing')
    parser.add_argument('--smoothing_threshold', type=float, default=0.25, help='Soft threshold value')
    
    # Dataset creation parameters
    parser.add_argument('--bounds', type=int, default=10, help='Bounds')
    parser.add_argument('--stride', type=int, default=32, help='Stride')
    parser.add_argument('--padding', type=str2bool, default=True, help='Padding')
    parser.add_argument('--num_nodes', type=int, default=128, help='Number of nodes')
    parser.add_argument('--remove_borders', type=int, default=30000000, help='Borders around a chromosome contact map to remove')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    
    # ablation parameters
    parser.add_argument('--rna_seq', type=str2bool, default=False, help='Use RNA-seq in node features')
    parser.add_argument('--use_bulk', type=str2bool, default=False, help='Use bulk Hi-C as a prior on structure')
    parser.add_argument('--positional_encodings', type=str2bool, default=False, help='Use positional encodings acquired through bulk Hi-C')
    parser.add_argument('--ctcf_motif', type=str2bool, default=False, help='Keep CTCF motifs in node features')
    parser.add_argument('--cpg_motif', type=str2bool, default=False, help='Keep CPG motifs in node features')
    parser.add_argument('--node_features', type=int, default=2, help='Node features')
    parser.add_argument('--pos_encodings_dim', type=int, default=16, help='Number of positional encodings')
    parser.add_argument('--bulk_hic', type=str, default='mesc', help='Which Bulk Hi-C contact map to use',
                        choices=['cerebral_cortex', 'mesc', 'inner_cell_mass', 'eight_cells', 'late_two_cell', 'early_two_cell', 'pn5_zygote'])
    
    # Model Encoder Parameters
    parser.add_argument('--conv1d_kernel_size', type=int, default=16, help='Conv1D kernel size')
    parser.add_argument('--encoder_hidden_embedding_size', type=int, default=32, help='Encoder hidden embedding size')
    parser.add_argument('--num_encoder_attn_blocks', type=int, default=4, help='Number of encoder attention blocks')
    parser.add_argument('--num_heads_encoder_attn_blocks', type=int, default=1, help='Number of heads in encoder attention blocks')
    parser.add_argument('--num_graph_conv_blocks', type=int, default=1, help='Number of graph convolution blocks')
    parser.add_argument('--num_graph_encoder_blocks', type=int, default=4, help='Number of graph encoder blocks')
    parser.add_argument('--edge_dims', type=int, default=1, help='Edge dimensions')
    
    # Model Decoder parameters
    parser.add_argument('--num_decoder_residual_blocks', type=int, default=7, help='Number of decoder residual blocks')
    parser.add_argument('--width', type=int, default=7, help='Width')
    parser.add_argument('--num_channels', type=int, default=1, help='Number of channels')
    
    # Loss function parameters
    parser.add_argument('--loss_scale', type=float, default=1, help='Scale the values in generated and target space to scale the contribution of non-zero values')
    
    # Model training Parameters
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--gradient_clip_value', type=float, default=0.1, help='Gradient clipping')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    
    args = parser.parse_args()
    
    parameters = vars(args)  # Convert Namespace to dictionary
    
    return parameters