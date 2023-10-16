import numpy as np
import pandas as pd


# Read in TSVs
kb1 = pd.read_csv('1000base_resolution_final_with_names.tsv', sep='\t')
kb10 = pd.read_csv('10000base_resolution_final_with_names.tsv', sep='\t')
kb100 = pd.read_csv('100000base_resolution_final_with_names.tsv', sep='\t')
mb1 = pd.read_csv('1000000base_resolution_final_with_names.tsv', sep='\t')


# Print out sparseness calculations for all resolutions
print("for 1kb:")
a = kb1.to_numpy()
kb1_rows = (a != 0).sum(1)
print("number of nonzero rows: ", np.count_nonzero(kb1_rows))
print("total number of rows: ", kb1.shape[0])
print("% sparseness: ", 100 * (1 - (np.count_nonzero(kb1_rows) / kb1.shape[0])))

print("==================")

print("for 10kb:")
b = kb10.to_numpy()
kb10_rows = (b != 0).sum(1)
print("number of nonzero rows: ", np.count_nonzero(kb10_rows))
print("total number of rows: ", kb10.shape[0])
print("% sparseness: ", 100 * (1 - (np.count_nonzero(kb10_rows) / kb10.shape[0])))

print("==================")

print("for 100kb:")
c = kb100.to_numpy()
kb100_rows = (c != 0).sum(1)
print("number of nonzero rows: ", np.count_nonzero(kb100_rows))
print("total number of rows: ", kb100.shape[0])
print("% sparseness: ", 100 * (1 - (np.count_nonzero(kb100_rows) / kb100.shape[0])))

print("==================")

print("for 1Mb:")
d = mb1.to_numpy()
mb1_rows = (d != 0).sum(1)
print("number of nonzero rows: ", np.count_nonzero(mb1_rows))
print("total number of rows: ", mb1.shape[0])
print("% sparseness: ", 100 * (1 - (np.count_nonzero(mb1_rows) / mb1.shape[0])))
