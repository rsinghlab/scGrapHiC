import warnings

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA



try:
	from scipy.stats import PearsonRConstantInputWarning, SpearmanRConstantInputWarning
except:
	from scipy.stats import ConstantInputWarning as PearsonRConstantInputWarning

########################################## NULL extractor ########################################################

def null_extractor(mat):
    return mat



########################################### A/B compartments ######################################################
'''
Compartment level analysis and plot funcitons
@author zliu
@data 20210902
'''

# for p(s) curve use log_bins=True , otherwise(e.g. normalize distance for Hi-C matrix ) use log_bins=False
def psDataFromMat(matrix, indices=None, log_bins=True, base=1.1):
    """
    ***FUNCTION COPY FROM HICSTAFF***
    Compute distance law as a function of the genomic coordinate aka P(s).
    Bin length increases exponentially with distance if log_bins is True. Works
    on dense and sparse matrices. Less precise than the one from the pairs.
    Parameters
    ----------
    matrix : numpy.array or scipy.sparse.coo_matrix
        Hi-C contact map of the chromosome on which the distance law is
        calculated.
    indices : None or numpy array
        List of indices on which to compute the distance law. For example
        compartments or expressed genes.
    log_bins : bool
        Whether the distance law should be computed on exponentially larger
        bins.
    Returns
    -------
    numpy array of floats :
        The start index of each bin.
    numpy array of floats :
        The distance law computed per bin on the diagonal
    """

    n = min(matrix.shape)
    included_bins = np.zeros(n, dtype=bool)
    if indices is None:
        included_bins[:] = True
    else:
        included_bins[indices] = True
    D = np.array(
        [
            np.average(matrix.diagonal(j)[included_bins[: n - j]])
            for j in range(n)
        ]
    )
    if not log_bins:
        return np.array(range(len(D))), D
    else:
        n_bins = int(np.log(n) / np.log(base) + 1)
        logbin = np.unique(
            np.logspace(0, n_bins - 1, num=n_bins, base=base, dtype=np.int)
        )
        logbin = np.insert(logbin, 0, 0)
        logbin[-1] = min(n, logbin[-1])
        if n < logbin.shape[0]:
            print("Not enough bins. Increase logarithm base.")
            return np.array(range(len(D))), D
        logD = np.array(
            [
                np.average(D[logbin[i - 1] : logbin[i]])
                for i in range(1, len(logbin))
            ]
        )
        return logbin[:-1], logD

def getOEMatrix(matrix:np.ndarray)->np.ndarray:
    """
    get decay profile normalized pearson correlation matrix
    """
    n=matrix.shape[0]
    dist_matrix = np.zeros((n, n))
    _, dist_vals = psDataFromMat(matrix, log_bins=False)
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = dist_vals[abs(j - i)]
    
    #obs/exp = obs / exp
    matrix = matrix / dist_matrix
    
    return matrix

def getPearsonCorrMatrix(matrix:np.ndarray)->np.ndarray:
    """
    get decay profile normalized pearson correlation matrix
    """
    matrix = getOEMatrix(matrix)
    matrix[np.isnan(matrix)] = 0
    np.fill_diagonal(matrix, 1)
    
    matrix = np.corrcoef(matrix)
    
    np.fill_diagonal(matrix, 1)
    
    return matrix

def addVec(a, b):
    if len(a) < len(b):
        c = b.copy()
        c[:len(a)] += a
    else:
        c = a.copy()
        c[:len(b)] += b
    return c

# call A/B compartmnet 
def ABcompartments(mat, chrom, cgpath, PARAMETERS, n_components=1):
    mat = getOEMatrix(mat)
    
    np.fill_diagonal(mat, 1)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=PearsonRConstantInputWarning
        )
        mat = getPearsonCorrMatrix(mat)
    
    np.fill_diagonal(mat, 1)
    mat[np.isnan(mat)] = 0.0
    
    pca = PCA(n_components=n_components)
    
    y = pca.fit_transform(mat)
    pca_df = pd.DataFrame(pca.components_.T)
    pca_df.columns = ["PC{}".format(i) for i in range(1, n_components+1)]
    
    pca_df["chrom"] = chrom
    pca_df["start"] = np.arange(0, len(pca_df)*PARAMETERS['resolution'], PARAMETERS['resolution'])
    pca_df["end"] = pca_df["start"] + PARAMETERS['resolution']
    pca_df = pca_df[["chrom", "start", "end"] + ["PC{}".format(i) for i in range(1, n_components+1)]]
    
    # correct the sign of PC1
    CG = pd.read_csv(cgpath, sep="\t", header = None)
    
    CG.columns = ["chrom","start", "GC"]
    CG = pca_df[['chrom','start']].merge(CG,how='left',on=['chrom','start'])
    CG = CG.fillna(0)
    
    if np.corrcoef(CG.query('chrom == @chrom')["GC"].values, pca_df["PC1"].values)[1][0] < 0:
        pca_df["PC1"] = -pca_df["PC1"]
        y = -y
    
    y = np.where(y > 0, 1, 0)
    
    return y

from scipy.fftpack import rfft, irfft

def smooth_data_fft(arr, span):  # the scaling of "span" is open to suggestions
    w = rfft(arr)
    spectrum = w ** 2
    cutoff_idx = spectrum < (spectrum.max() * (1 - np.exp(-span / 2000)))
    w[cutoff_idx] = 0
    return irfft(w)


def insulationScore(m, windowsize=500000, res=40000):
    """
    input: contact matrix,windowsize for sliding window, resolution of your contact matrix.
    ourput:

    """
    windowsize_bin = int(windowsize / res)
    score = np.ones((len(m)))
    for i in range(windowsize_bin, len(m) - windowsize_bin):
        with np.errstate(divide='ignore', invalid='ignore'):
            v = np.sum(m[max(0, i - windowsize_bin): i, i + 1: min(len(m) - 1, i + windowsize_bin + 1)]) / (np.sum(
                m[max(0, i - windowsize_bin):min(len(m) - 1, i + windowsize_bin + 1),
                    max(0, i - windowsize_bin):min(len(m) - 1, i + windowsize_bin + 1)]))
            if np.isnan(v):
                v = 0

        score[i] = v
    
    score[score >= 0.99] = 0
    # score = smooth_data_fft(score, 4)
    
    return score