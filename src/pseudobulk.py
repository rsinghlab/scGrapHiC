import enum
import os


import numpy as np
import pandas as pd



from src.globals import *
from src.utils import create_directory
from src.preprocess_datasets import read_cell_by_gene_matrix, read_pairix_file

def pseudobulk_rnaseq(tissue, cell_names, cell_type, stage=None):
    umi_file = os.path.join(
        MOUSE_RAW_DATA_SCRNASEQ, 
        'GSE223917_HiRES_{}.rna.umicount.tsv.gz'.format(tissue)
    )
    cell_names_with_gene = ['gene'] + cell_names
    
    umi_data = read_cell_by_gene_matrix(umi_file)
    umi_data = umi_data.filter(cell_names_with_gene)
    umi_data[cell_type] = umi_data[cell_names].sum(axis=1)
    umi_data = umi_data.drop(cell_names, axis=1)
    
    umi_data[cell_type] = umi_data[cell_type]/len(cell_names)

    if stage:
        output_path = os.path.join(
            MOUSE_RAW_DATA_PSEUDO_BULK_SCRNASEQ,
            '{}_{}'.format(stage, tissue)
        )
        
    else:
        output_path = os.path.join(
            MOUSE_RAW_DATA_PSEUDO_BULK_SCRNASEQ,
            tissue,
        )
    
    
    create_directory(output_path)
    output_file = os.path.join(
        output_path,
        '{}_umi.tsv'.format(cell_type) 
    )
    
    print('Saving pseudobulk UMI file at {}'.format(output_file))
    umi_data.to_csv(output_file)
    
    
    
    

def pseudobulk_schic(tissue, cell_names, cell_type, stage=None):
    schic_files = list(map(
        lambda x: os.path.join(MOUSE_RAW_DATA_SCHIC, '{}.pairs.gz'.format(x)),
        cell_names
    ))
    
    pseudobulk_dataframe = read_pairix_file(schic_files[0])    
    
    for schic_file in schic_files[1:]:
        schic_data = read_pairix_file(schic_file)        
        pseudobulk_dataframe = pd.concat([pseudobulk_dataframe, schic_data])
        

    if stage:
        output_path = os.path.join(
            MOUSE_RAW_DATA_PSEUDO_BULK_SCHIC,
            '{}_{}_n{}'.format(stage, tissue, len(cell_names))
        )
        
    else:
        output_path = os.path.join(
            MOUSE_RAW_DATA_PSEUDO_BULK_SCRNASEQ,
            '{}_n{}'.format(tissue, len(cell_names))
        )
    create_directory(output_path)
    
    output_file = os.path.join(output_path, '{}_schic.tsv'.format(cell_type))
    
    print('Saving pseudobulk scHi-C file at {}'.format(output_file))
    pseudobulk_dataframe.to_csv(output_file)
    



def parse_metadata(metadata):
    try:
        cell_types = metadata['Celltype'].unique().tolist()
    except KeyError:
        cell_types = metadata['Cell type'].unique().tolist()
    cell_names = []
    
    for cell_type in cell_types:
        try:
            cell_name = metadata[(metadata['Celltype'] == cell_type)]['Cellname'].tolist()
        except KeyError:
             cell_name = metadata[(metadata['Cell type'] == cell_type)]['Cellname'].tolist()
        
        cell_names.append(cell_name)
    
    return cell_types, cell_names
    
    
    

def create_pseudobulk_files(path):
    metadata = pd.read_excel(path)
    tissue = path.split('/')[-1].split('_')[0]
    
    if tissue == 'embryo':
        # we repeat the process for all stages
        stages = metadata['Stage'].unique().tolist()
        for stage in stages: 
            cell_types, cell_names = parse_metadata(metadata[(metadata['Stage'] == stage)])
            for i, cell_type in enumerate(cell_types):
                pseudobulk_rnaseq(tissue, cell_names[i], cell_type, stage)
                
    
    elif tissue == 'brain':
        # there is only one type of cells
        cell_types, cell_names = parse_metadata(metadata)
        for i, cell_type in enumerate(cell_types):
            pseudobulk_rnaseq(tissue, cell_names[i], cell_type)
            pseudobulk_schic(tissue, cell_names[i], cell_type)
            

    else:
        print('Invalid metadata file path, exiting program...')
        exit(1)   
            
            
    
















