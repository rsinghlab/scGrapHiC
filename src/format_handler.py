import os
import cooler

import scanpy as sc
import pandas as pd

from src.utils import *
from src.globals import *
from anndata import AnnData



def read_pairix_file(path):
    '''
        This function reads a pairix file format (.pairs) and returns 
        a dictionary of numpy arrays
        @params: <string> - path, path to the file 
        @returns: <dict> - dictionary of np.arrays 
    '''
    if os.path.exists(path):
        data = pd.read_csv(
            path, header = None,
            comment ='#', sep ='\t', 
            names=[
                'readID', 
                'chr1', 'pos1',
                'chr2', 'pos2', 
                'strand1', 'strand2', 
                'phase0', 'phase1'
            ]

        )
        data['pos1'] = pd.to_numeric(data["pos1"])
        data['pos2'] = pd.to_numeric(data["pos2"])
        data = data.drop(['readID', 'strand1', 'strand2', 'phase0', 'phase1'], axis=1)

        return data
    else:
        print('Invalid pairix file path {}, exiting program.'.format(
            path
        ))


def convert_pairs_to_pixels(dataframe, resolution=RESOLUTION):
    dataframe.loc[:, 'pos1'] = dataframe['pos1'].copy().floordiv(resolution)
    dataframe.loc[:, 'pos2'] = dataframe['pos2'].copy().floordiv(resolution)
    pixels = dataframe.groupby(['pos1', 'pos2']).size().reset_index(name='counts')
    pixels = pixels.rename(columns={'pos1': 'bin1_id', 'pos2': 'bin2_id', 'counts': 'count'}) 
    return pixels



def cooler_files_from_pairix_file(input_file, output_folder, chrom_sizes_file):
    chrom_sizes = read_chromsizes_file(chrom_sizes_file)
    pairix_data = read_pairix_file(input_file)
    
    for chrom in chrom_sizes.keys():
        output_file = os.path.join(output_folder, '{}.cool'.format(chrom))
        chrom_data = pairix_data.loc[(pairix_data['chr1'] == chrom) & (pairix_data['chr2'] == chrom)]
        
        chrom_pixels = convert_pairs_to_pixels(chrom_data)
        bins = chrom_bins(chrom, chrom_sizes[chrom])

        cooler.create_cooler(output_file, bins, chrom_pixels,
                        dtypes={"count":"int"}, 
                        assembly="mm10")
        



def extract_dense_matrix_from_cooler_file(cooler_path, log=True):
    chr = cooler_path.split('/')[-1].split('.')[0]
    clr = cooler.Cooler(cooler_path)
    matrix = clr.matrix(balance=False).fetch(chr)
    if log:
        return np.log2(matrix + 1)
    else: 
        return matrix


def read_cell_by_gene_matrix(path):
    cell_by_gene_data = pd.read_csv(
        path, sep='\t',
        comment='#'
    )
    return cell_by_gene_data


def normalize_cell_by_gene_matrix(cell_by_gene_matrix):
    X = cell_by_gene_matrix.iloc[1:, 1:].to_numpy()
    genes = cell_by_gene_matrix.iloc[1:, 0:1].to_numpy()
    cells = cell_by_gene_matrix.columns[1:].to_numpy()

    adata = AnnData(X.T)
    X_norm = sc.pp.normalize_total(adata, target_sum=1, inplace=False)['X']
    X_norm = X_norm.T

    cell_by_gene_matrix = pd.DataFrame(data=X_norm, index=genes, columns=cells)


    print(cell_by_gene_matrix)




def convert_cell_by_gene_to_coordinate_matrix(cell_by_gene_matrix, gene_coordinate_file):
    gene_coordinates = pd.read_csv(gene_coordinate_file)
    # Inner join on both tables and we drop the gene_names because thats a more comprehensive list (to remove rows with NaNs). 
    merged_tables = pd.merge(gene_coordinates, cell_by_gene_matrix, left_on='gene_name', right_on='gene', how='left').drop('gene_name', axis=1)
    print(merged_tables)
    print(merged_tables[merged_tables.isna().any(axis=1)])
    
    return 


