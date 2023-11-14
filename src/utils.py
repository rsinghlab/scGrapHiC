import os

import numpy as np
import pandas as pd

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








































