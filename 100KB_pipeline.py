import numpy as np
import pandas as pd
# from tqdm import tqdm
print("starting script")
df = pd.read_csv('out.tsv', sep='\t')
df.sort_index(axis='index')
print("finished reading out.tsv")
resolution = 100000 #bases
_newdf = np.zeros((1,9022))
recorded_row = np.zeros(9022)
_sum = 0
current_chromosome = ''
res = ''

# get values 
# for gene in tqdm(df.index): # gene is a string, ie 'chr10:100002601-100002817'
for gene in df.index:
    if (gene.split(':')[0] != current_chromosome):
        _sum = 0
        _newdf = np.vstack([_newdf, recorded_row])
        recorded_row = np.zeros(9022)
        current_chromosome = gene.split(':')[0]

    start = int(gene.split(':')[1].split('-')[0])
    end = int(gene.split(':')[1].split('-')[1].split('|')[0])
    _sum += end - start
    recorded_row = np.add(recorded_row, df.loc[gene].to_numpy())
    if (_sum >= resolution):
        _sum = 0
        _newdf = np.vstack([_newdf, recorded_row])
        recorded_row = np.zeros(9022)

# get chromosome range strings
for gene in df.index:
    if (gene.split(':')[0] != current_chromosome):
        _sum = 0
        res.append(recorded_row + prev)
        current_chromosome = gene.split(':')[0]
        recorded_row = current_chromosome + ':' + gene.split(':')[1].split('-')[0] + '-'

    if (recorded_row == ''):
        recorded_row = current_chromosome + ':' + gene.split(':')[1].split('-')[0] + '-'

    start = int(gene.split(':')[1].split('-')[0])
    end = int(gene.split(':')[1].split('-')[1].split('|')[0])
    _sum += end - start
    if (_sum >= resolution):
        _sum = 0
        res.append(recorded_row + gene.split(':')[1].split('-')[1].split('|')[0])
        recorded_row = ''

    prev = gene.split(':')[1].split('-')[1].split('|')[0]

if recorded_row != '':
    res.append(recorded_row)

if np.count_nonzero(recorded_row) != 0:
    _newdf = np.vstack([_newdf, recorded_row])

_newdf = np.delete(_newdf, (0), axis=0)
print("starting creation of dataframe")
_newdf = pd.DataFrame(_newdf, columns=df.columns, index=res)
print("finished making dataframe")
path = str(resolution) + '_base_resolution.tsv'
print("starting transfer to TSV file")
_newdf.to_csv(path, sep='\t')
print("finished converting to CSV!")
print(_newdf.shape)