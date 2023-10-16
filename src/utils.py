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
    starts = (np.arange(0, size, 1, dtype=int))*resolution
    ends = (np.arange(1, size+1, 1, dtype=int))*resolution
    
    bins = {
        'chrom': chr_names,
        'start': starts,
        'end': ends
    }

    bins = pd.DataFrame(data=bins) 
    
    return bins




































