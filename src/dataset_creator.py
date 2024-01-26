'''
    Lets just create a data loader for a single chromosome to make the logic simpler and the task easier perhaps
'''

import os
import json
import multiprocessing

import numpy as np

from src.globals import *
from itertools import repeat
from multiprocessing import Pool
from src.graph_pe import graph_pe
from sklearn.model_selection import train_test_split
from src.normalizations import normalizations, smooth_adjacency_matrix
from src.normalizations import smooth_adjacency_matrix, generate_expected_contact_matrix
from src.utils import get_file_name_parameters, divide_signal, divide_matrix, add_dataset, compactM




MULTIPROCESSING = True


def create_schires_dataset(PARAMETERS):
    scrnaseq_files = os.listdir(MOUSE_PREPROCESSED_DATA_SCRNASEQ)
    schic_files = os.listdir(MOUSE_PREPROCESSED_DATA_SCHIC)
    
    
    # find the union of these files and only work with them 
    cells = list(set(scrnaseq_files) & set(schic_files))
    chromosme = 'chr1'
    # For each cell create a graphish and then store it in a list that will eventually form our dataloader
    bulk_hic_file = os.path.join(MOUSE_PREPROCESSED_DATA_BULK, PARAMETERS['bulk_hic'], '{}_{}.npz'.format(chromosme, PARAMETERS['resolution']))
    bulk_hic_object = np.load(bulk_hic_file, allow_pickle=True)
    bulk_hic_data = bulk_hic_object['hic']
        
    node_features = []
    bulks = []
    targets = []
    ab_datas = []
    tad_datas = []
    
    for cell in cells:
        try:
            scrnaseq_file = os.path.join(MOUSE_PREPROCESSED_DATA_SCRNASEQ, cell, '{}_{}.npy'.format(chromosme, PARAMETERS['resolution']))
            scrnaseq_data = np.load(scrnaseq_file).transpose(1, 0)
            
            schic_file = os.path.join(MOUSE_PREPROCESSED_DATA_SCHIC, cell, '{}_{}.npy'.format(chromosme, PARAMETERS['resolution']))
            schic_data = np.load(schic_file)            
            # schic_data = matImpute(schic_data, 1, -1)
            
            ab_file = os.path.join(MOUSE_PREPROCESSED_DATA_SCHIC, cell, 'compartments_{}_{}.npy'.format(chromosme, PARAMETERS['resolution']))
            ab_data = np.load(ab_file)
            
            tad_file = os.path.join(MOUSE_PREPROCESSED_DATA_SCHIC, cell, 'TADs_{}_{}.npy'.format(chromosme, PARAMETERS['resolution']))
            tad_dat = np.load(tad_file)
            
        except:
            print('Cell is missing for one of the two modalities.')
            continue
            
        
        pe = graph_pe(bulk_hic_data, encoding_dim=PARAMETERS['pos_encodings_dim'])
        
        if PARAMETERS['pos_encodings_dim'] != 0:
            node_feature = np.concatenate((scrnaseq_data.transpose(1, 0), pe.transpose(1, 0)))
        else:
            node_feature = scrnaseq_data.transpose(1, 0)
            
        
        
        
        
        node_features.append(node_feature.reshape(1, node_feature.shape[0], node_feature.shape[1]))
        # bulks.append(bulk_hic_data.reshape(1, bulk_hic_data.shape[0], bulk_hic_data.shape[1]))
        targets.append(schic_data.reshape(1, schic_data.shape[0], schic_data.shape[1]))
        ab_datas.append(ab_data.reshape(1, ab_data.shape[0], ab_data.shape[1]))
        tad_datas.append(tad_dat.reshape(1, tad_dat.shape[0], tad_dat.shape[1]))
        
    node_features = np.concatenate(node_features)
    # bulks = np.concatenate(bulks)
    targets = np.concatenate(targets)
    ab_datas = np.concatenate(ab_datas)
    tad_datas = np.concatenate(tad_datas)
    
    # node_features_train, node_features_test, bulks_train, bulks_test, targets_train, targets_test = train_test_split(node_features, bulks, targets, test_size=0.15)
    node_features_train, node_features_test, ab_train, ab_test, tad_train, tad_test, targets_train, targets_test = train_test_split(node_features, ab_datas, tad_datas, targets, test_size=0.15)
    
    train_output_file = os.path.join(MOUSE_PROCESSED_DATA_HIRES, 'train.npz')
    
    np.savez_compressed(train_output_file, 
        node_features=node_features_train, 
        hic_targets=targets_train,
        ab_targets=ab_train,
        tad_targets=tad_train
    )
    
    test_output_file = os.path.join(MOUSE_PROCESSED_DATA_HIRES, 'test.npz')
    
    np.savez_compressed(test_output_file, 
        node_features=node_features_test, 
        hic_targets=targets_test,
        ab_targets=ab_test,
        tad_targets=tad_test
    )
    
    

dataset_partitions = {
    'train': [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19],
    'valid': [7, 11],
    'test':  [7, 11],
    'ood':   [7, 11],
    'debug': [1]
}



def create_chromosome_dataset(rna_seq_dataset_path, schic_dataset_path, chromosome, PARAMETERS):
    stage, tissue, cell_type, num_cells = get_file_name_parameters(schic_dataset_path)
    if not stage:
        stage = 'brain'
    
    cell_type = cell_type.replace(' ', '_')
    
    rna_seq_file = os.path.join(rna_seq_dataset_path, 'chr{}_{}.npy'.format(chromosome, PARAMETERS['resolution']))
    schic_file =  os.path.join(schic_dataset_path, 'chr{}_{}.npy'.format(chromosome, PARAMETERS['resolution']))
    
    if PARAMETERS['bulk_hic'] == 'basic_prior':
        bulk_hic_file = os.path.join(MOUSE_PREPROCESSED_DATA_BULK, 'pn5_zygote', 'chr{}_{}.npz'.format(chromosome, PARAMETERS['resolution']))
    else:
        bulk_hic_file = os.path.join(MOUSE_PREPROCESSED_DATA_BULK, PARAMETERS['bulk_hic'], 'chr{}_{}.npz'.format(chromosome, PARAMETERS['resolution']))

    
    informative_indexes_bulk_hic = os.path.join(MOUSE_PREPROCESSED_DATA_BULK, 'mesc', 'chr{}_{}.npz'.format(chromosome, PARAMETERS['resolution']))
    
    
    ctcf_motif_file = os.path.join(MOUSE_PREPROCESSED_MOTIFS_DATA, 'ctcf', 'chr{}_{}.npy'.format(chromosome, PARAMETERS['resolution']))
    cpg_motif_file = os.path.join(MOUSE_PREPROCESSED_MOTIFS_DATA, 'cpg', 'chr{}_{}.npy'.format(chromosome, PARAMETERS['resolution']))
    dataset_labels = json.load(open(DATASET_LABELS_JSON, 'r'))
    
    border_size = PARAMETERS['remove_borders'] // PARAMETERS['resolution']
    
    # Bulk Hi-C (prior)
    bulk_hic_object = np.load(bulk_hic_file, allow_pickle=True)
    # Always load the mESC bulk sample for informative indexes otherwise we can produce misaligned condensed matrices
    informative_indexes = np.load(informative_indexes_bulk_hic, allow_pickle=True)['compact']
    
    bulk_hic_data = compactM(bulk_hic_object['hic'], informative_indexes)
    bulk_hic_data = bulk_hic_data[border_size:, border_size:]
    
    # Divide
    bulk_hic_data, _ = divide_matrix(bulk_hic_data, chromosome, PARAMETERS)
    
    if PARAMETERS['bulk_hic'] == 'basic_prior':
        replace_with_prior = lambda adj: generate_expected_contact_matrix(adj)
        bulk_hic_data = bulk_hic_data.reshape([bulk_hic_data.shape[0], -1])
        bulk_hic_data = np.apply_along_axis(replace_with_prior, 1, bulk_hic_data)
    
    # Features
    rna_seq_data = np.load(rna_seq_file)
    ctcf_motif_data = np.load(ctcf_motif_file)
    cpg_motif_data = np.load(cpg_motif_file)
    
    # Take informative indices only
    rna_seq_data = rna_seq_data.take(informative_indexes, axis=1)
    ctcf_motif_data = ctcf_motif_data.take(informative_indexes, axis=1)
    cpg_motif_data = cpg_motif_data.take(informative_indexes, axis=1)
    
    # Clip Borders
    rna_seq_data = rna_seq_data[:, border_size:]
    ctcf_motif_data = ctcf_motif_data[:, border_size:]
    cpg_motif_data = cpg_motif_data[:, border_size:]
    
    node_features = rna_seq_data
    if PARAMETERS['ctcf_motif'] == True:
        node_features = np.concatenate((node_features, ctcf_motif_data))
    
    if PARAMETERS['cpg_motif'] == True:
        node_features = np.concatenate((node_features, cpg_motif_data))

    node_features, _ = divide_signal(node_features.T, chromosome, PARAMETERS)    
    node_features = node_features[:, 0, :, :]

    # scHi-C data
    schic_data = np.load(schic_file)
    schic_data = compactM(schic_data, informative_indexes)
    schic_data = schic_data[border_size: , border_size:]
    
    # Divide
    schic_data, indexes = divide_matrix(schic_data, chromosome, PARAMETERS)
    
    
    
    schic_data = schic_data.reshape([schic_data.shape[0], -1])
    bulk_hic_data = bulk_hic_data.reshape([bulk_hic_data.shape[0], -1])
    
    if PARAMETERS['normalization_algorithm'] == 'library_size_normalization':
        normalization_function = lambda adj: normalizations[PARAMETERS['normalization_algorithm']](adj, PARAMETERS['library_size'])
        schic_data = np.apply_along_axis(normalization_function, 1, schic_data)
        bulk_hic_data = np.apply_along_axis(normalization_function, 1, bulk_hic_data)
    else:
        normalization_function = lambda adj: normalizations[PARAMETERS['normalization_algorithm']](adj)
        schic_data = np.apply_along_axis(normalization_function, 1, schic_data)
        bulk_hic_data = np.apply_along_axis(normalization_function, 1, bulk_hic_data)
    
    schic_data = schic_data.reshape(schic_data.shape[0], 1, schic_data.shape[1], schic_data.shape[2]) 
    bulk_hic_data = bulk_hic_data.reshape(bulk_hic_data.shape[0], 1, bulk_hic_data.shape[1], bulk_hic_data.shape[2]) 
    
    parameterized_graph_pe = lambda adj: graph_pe(adj, encoding_dim=PARAMETERS['pos_encodings_dim'])
    pe = bulk_hic_data.reshape([bulk_hic_data.shape[0], -1])
    pe = np.apply_along_axis(parameterized_graph_pe, 1, pe)
    
    if PARAMETERS['hic_smoothing']:
        smooth_parameterized = lambda adj: smooth_adjacency_matrix(adj, PARAMETERS['smoothing_threshold'])
        schic_data = schic_data.reshape([schic_data.shape[0], -1])
        schic_data = np.apply_along_axis(smooth_parameterized, 1, schic_data)
    
    metadata = np.array([
        [dataset_labels['stage'][stage]]*indexes.shape[0],
        [dataset_labels['tissue'][tissue]]*indexes.shape[0],
        [dataset_labels['cell_type'][cell_type]]*indexes.shape[0],
        [int(num_cells)]*indexes.shape[0]
    ]).T
    
    return node_features[4:-4, :, :], schic_data[4:-4, :, :, :], pe[4:-4, :, :], bulk_hic_data[4:-4, :, :, :], indexes[4:-4, :], metadata[4:-4, :]



def _create_chromosome_datasets(args):
    '''
        Stupid arg passing trick for multiprocessing
    '''
    return create_chromosome_dataset(*args)



def create_cell_type_dataset(rnaseq_folder, schic_folder, PARAMETERS, set='debug'):
    chromosomes = dataset_partitions[set]

    # Creating arg lists for the multiprocessor
    chromosomes = list(map(lambda x: str(x), chromosomes))
    args = zip(
        repeat(rnaseq_folder),
        repeat(schic_folder),
        chromosomes,
        repeat(PARAMETERS)
    )
    
    num_cpus = multiprocessing.cpu_count() if MULTIPROCESSING else 1
    if num_cpus >= len(chromosomes):
        num_cpus = len(chromosomes)
    
    
    with Pool(num_cpus) as pool:
        results = pool.map(_create_chromosome_datasets, args)
    

    results = list(filter(lambda x: len(x) != 0, results))

    node_features = np.concatenate([r[0] for r in results])
    targets = np.concatenate([r[1] for r in results])
    pes = np.concatenate([r[2] for r in results])
    bulk_hic = np.concatenate([r[3] for r in results])
    indexes = np.concatenate([r[4] for r in results])
    metadatas = np.concatenate([r[5] for r in results])
    
    return node_features, targets, pes, bulk_hic, indexes, metadatas



def create_schic_pseudobulk_dataset(exclusion_set, PARAMETERS, set='debug', descriptor='pb'):
    scrnaseq_dataset_files = list(map(
        lambda x: os.path.join(MOUSE_PREPROCESSED_DATA_PSEUDO_BULK_SCRNASEQ, x), 
        os.listdir(MOUSE_PREPROCESSED_DATA_PSEUDO_BULK_SCRNASEQ))
    )
    scrnaseq_dataset_files = list(filter(
        lambda x: '.csv' not in x,
        scrnaseq_dataset_files
    ))
    
    schic_dataset_paths = []
    scrnaseq_dataset_paths = []
    for scrnaseq_dataset_file in scrnaseq_dataset_files:
        # Check if the dataset has enough cells? 
        stage, tissue, cell_type, num_cells = get_file_name_parameters(scrnaseq_dataset_file)
        
        # Update the json dictionary
        # add_dataset(stage, tissue, cell_type)
        
        
        # Exclusion criterion
        if num_cells < PARAMETERS['num_cells_cutoff']:
            continue
        
        if tissue in exclusion_set or stage in exclusion_set or cell_type in exclusion_set:
            continue
        
        folder = '_'.join([stage, tissue, cell_type, 'n{}'.format(num_cells), 'schic']) if stage else  '_'.join([tissue, cell_type, 'n{}'.format(num_cells), 'schic'])
        
        schic_folder_path = os.path.join(MOUSE_PREPROCESSED_DATA_PSEUDO_BULK_SCHIC, folder)
                
        if os.path.exists(schic_folder_path):
            schic_dataset_paths.append(schic_folder_path)
            scrnaseq_dataset_paths.append(scrnaseq_dataset_file)
    
    nfs = []
    tars = []
    pes = []
    bhs = []
    idxes = []
    metadatas = []
    
    output_file = os.path.join(
        MOUSE_PROCESSED_DATA_HIRES,
        '{}_{}.npz'.format(descriptor, set)
    )
    
    for rnaseq_folder, schic_folder in zip(scrnaseq_dataset_paths, schic_dataset_paths):
        print('Working with: ', rnaseq_folder, ' and ', schic_folder)
        
        _nf, _tar, _pes, _bh, _idx, _metadata = create_cell_type_dataset(
            rnaseq_folder,
            schic_folder,
            PARAMETERS, 
            set,
        )
        nfs.append(_nf)
        tars.append(_tar)
        pes.append(_pes)
        bhs.append(_bh)
        idxes.append(_idx)
        metadatas.append(_metadata)
    
    
    
    nfs = np.concatenate(nfs)
    tars = np.concatenate(tars)
    pes = np.concatenate(pes)
    bhs = np.concatenate(bhs)
    idxes = np.concatenate(idxes) 
    metadatas = np.concatenate(metadatas)
    
    print(nfs.shape, tars.shape, pes.shape, bhs.shape, idxes.shape, metadatas.shape)
    
    print('Saving file:', output_file)
    
    
    np.savez_compressed(output_file, 
        node_features=nfs, 
        targets=tars,
        pes=pes,
        bulk_hics=bhs,
        indexes=idxes,
        metadatas=metadatas
    )    
            
        
    