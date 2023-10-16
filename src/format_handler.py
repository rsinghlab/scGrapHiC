import os
import cooler

import pandas as pd
from src.utils import *
from src.globals import *

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
    bin1 = dataframe['pos1'].tolist()
    bin1 = [x//resolution for x in bin1]

    bin2 = dataframe['pos2'].tolist()
    bin2 =  [x//resolution for x in bin2]
    
    bins = list(map(lambda x: str(x[0])+':'+str(x[1]), zip(bin1, bin2)))
    
    counts = {i:bins.count(i) for i in bins}
    
    
    
    print(counts)


    # counts = [1]*len(bin_1)

    # return {
    #     'bin1_id': bin_1,
    #     'bin2_id': bin_2,
    #     'count': counts
    # }



def cooler_files_from_pairix_file(input_file, output_folder, chrom_sizes_file):
    chrom_sizes = read_chromsizes_file(chrom_sizes_file)
    pairix_data = read_pairix_file(input_file)
    
    for chrom in chrom_sizes.keys():
        output_file = os.path.join(output_folder, '{}.cool'.format(chrom))
        chrom_data = pairix_data.loc[(pairix_data['chr1'] == chrom) & (pairix_data['chr2'] == chrom)]
        
        chrom_pixels = convert_pairs_to_pixels(chrom_data)
        exit(1)
        # bins = chrom_bins(chrom, chrom_sizes[chrom])

        # cooler.create_cooler(output_file, bins, chrom_pixels,
        #                 dtypes={"count":"int"}, 
        #                 assembly="hg19")






    pass