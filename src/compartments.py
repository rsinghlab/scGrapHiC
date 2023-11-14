"""
Compartment level analysis and plot funcitons
@author zliu
@data 20210902
"""
#global dependence
import numpy as np
import pandas as pd

from src.globals import *
from sklearn.decomposition import PCA



# Decay profile
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
def ABcompartment_from_mat(mat, chrom, cgpath, PARAMETERS, n_components=1):
    mat = sqrt_norm(mat)
    mat = oe(mat, None)
    np.fill_diagonal(mat, 1)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=PearsonRConstantInputWarning
        )
        mat = pearson(mat)
    
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
    CG = pd.read_csv(cgpath,sep="\t",header = None)
    
    CG.columns = ["chrom","start", "GC"]
    CG = pca_df[['chrom','start']].merge(CG,how='left',on=['chrom','start'])
    CG = CG.fillna(0)
    
    if np.corrcoef(CG.query('chrom == @chrom')["GC"].values, pca_df["PC1"].values)[1][0] < 0:
        pca_df["PC1"] = -pca_df["PC1"]
        y = -y
    
    max_score = np.max(y)
    y = np.divide(y, max_score)
    
    return y




######################### Higashi Implementation ###################################################3

try:
	from scipy.stats import PearsonRConstantInputWarning, SpearmanRConstantInputWarning
except:
	from scipy.stats import ConstantInputWarning as PearsonRConstantInputWarning
import warnings
from scipy.fftpack import rfft, irfft


def sqrt_norm(matrix):
	coverage = (np.sqrt(np.sum(matrix, axis=-1)))
	with np.errstate(divide='ignore', invalid='ignore'):
		matrix = matrix / coverage.reshape((-1, 1))
		matrix = matrix / coverage.reshape((1, -1))
	matrix[np.isnan(matrix)] = 0.0
	matrix[np.isinf(matrix)] = 0.0
	return matrix


def smooth_data_fft(arr, span):  # the scaling of "span" is open to suggestions
    w = rfft(arr)
    spectrum = w ** 2
    cutoff_idx = spectrum < (spectrum.max() * (1 - np.exp(-span / 2000)))
    w[cutoff_idx] = 0
    return irfft(w)


def kth_diag_indices(a, k):
	rows, cols = np.diag_indices_from(a)
	if k < 0:
		return rows[-k:], cols[:k]
	elif k > 0:
		return rows[:-k], cols[k:]
	else:
		return rows, cols



def oe(matrix, expected = None):
	new_matrix = np.zeros_like(matrix)
	for k in range(len(matrix)):
		rows, cols = kth_diag_indices(matrix, k)
		diag = np.diag(matrix,k)
		if expected is not None:
			expect = expected[k]
		else:
			expect = np.sum(diag) / (np.sum(diag != 0.0) + 1e-15)
		if expect == 0:
			new_matrix[rows, cols] = 0.0
		else:
			new_matrix[rows, cols] = diag / (expect)
	new_matrix = new_matrix + new_matrix.T
	return new_matrix

def pearson(matrix):
	return np.corrcoef(matrix)

def compartment(matrix, expected=None):
    contact = matrix
    # np.fill_diagonal(contact, np.max(contact))
    # contact = KRnormalize(matrix)
    # contact[np.isnan(contact)] = 0.0
    
    contact = sqrt_norm(matrix)
    contact = oe(contact, expected)
    np.fill_diagonal(contact, 1)
    
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=PearsonRConstantInputWarning
        )
        contact = pearson(contact)
    
    np.fill_diagonal(contact, 1)
    contact[np.isnan(contact)] = 0.0
	
    pca = PCA(n_components=1)
    y = pca.fit_transform(contact)
    
    max_score = np.max(y)
    y = np.divide(y, max_score)
    
    return y