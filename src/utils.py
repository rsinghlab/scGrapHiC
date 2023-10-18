import os

import numpy as np
import pandas as pd

from src.globals import RESOLUTION


def read_chromsizes_file(path):
    data = open(path).read().split('\n')
    data = list(map(lambda x: x.split(' '), data))
    data = list(map(lambda x: [x[0], int(x[1])], data))

    return dict(data)




def chrom_bins(chr, chr_size, resolution=RESOLUTION):
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



def process_GTF3_file(path):
    output_path = os.path.join('/'.join(path.split('/')[:-1]), 'gene_coordinates.csv')

    data = pd.read_csv(
            path, header = None,
            comment ='#', sep ='\t', 
            names=[
                'seqid', 
                'source', 'type',
                'start', 'end', 
                'score', 'strand', 'phase', 
                'attributes'
            ]

        )
    
    # We only need the genes annotations
    data = data.loc[data['type'] == 'gene']
    # We only need the gene name
    data['attributes'] = data['attributes'].map(lambda x: get_gene_name(x))

    data = data.drop(columns=['source', 'type', 'score', 'phase'])
    data = data.rename(columns={'seqid': 'chr', 'attributes': 'gene_name'})
    
    data.to_csv(output_path, index=False)




































