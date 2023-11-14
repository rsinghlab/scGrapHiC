'''
    Lets just create a data loader for a single chromosome to make the logic simpler and the task easier perhaps
'''

import os

import numpy as np

from src.globals import *
from src.graph_pe import graph_pe
from src.matrix_imputations import matImpute
from sklearn.model_selection import train_test_split
from src.preprocess_datasets import normalize_schic_matrix
from src.visualizations import visualize_hic_contact_matrix, visualize_scnrna_seq_tracks



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
    bulk_hic_data = normalize_schic_matrix(bulk_hic_data, PARAMETERS)
    
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