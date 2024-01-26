import math

import numpy as np
import scipy.sparse as sps

from scipy.stats import zscore
from sklearn.preprocessing import QuantileTransformer


def is_perfect_square(num):
    if num < 0:
        return False

    # Calculate the square root
    square_root = math.isqrt(num)

    # Check if the square root is equal to its integer value
    return square_root * square_root == num


def smooth_adjacency_matrix(A, threshold=1.0):
    # Vectorization trick handler
    if len(A.shape) ==1 and is_perfect_square(A.shape[0]):
        A = A.reshape(math.isqrt(A.shape[0]), math.isqrt(A.shape[0]))
    
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    # Apply smoothing function to eigenvalues
    smoothed_eigenvalues = soft_thresholding(eigenvalues, threshold)

    # Reconstruct the smoothed adjacency matrix using Cholesky decomposition
    smoothed_A = eigenvectors @ np.diag(smoothed_eigenvalues) @ eigenvectors.T

    smoothed_A = smoothed_A.real # Ensure the result is a real matrix
    
    return smoothed_A.reshape(1, smoothed_A.shape[0], smoothed_A.shape[1]) 

def soft_thresholding(eigenvalues, threshold):
    return np.sign(eigenvalues) * np.maximum(np.abs(eigenvalues) - threshold, 0)


def generate_expected_contact_matrix(matrix, decay_constant=0.2):
    """
        Generates an expected Hi-C contact matrix for the given genome.

        Parameters:
        num_bins (int): The number of  bins in the contact matrix.
        decay_constant (float): The decay constant to use for the expected number of contacts.
        Returns:
        numpy.ndarray: The expected Hi-C contact matrix.
    """
    # Initialize the contact matrix with zeros
    num_bins = matrix.shape[-1]
    contact_matrix = np.zeros((1, num_bins, num_bins))

    # Create a grid of bin indices
    i, j = np.meshgrid(np.arange(num_bins), np.arange(num_bins))

    # Calculate the distance between the bins
    bin_distance = np.abs(i - j)

    # Calculate the expected number of contacts between the bins
    expected_contacts = np.exp(-decay_constant * bin_distance)

    # Update the contact matrix with the expected number of contacts
    contact_matrix[0, i, j] = expected_contacts

    return contact_matrix



def library_size_normalization(matrix, library_size=25000):
    # Vectorization trick handler
    if len(matrix.shape) ==1 and is_perfect_square(matrix.shape[0]):
        matrix = matrix.reshape(math.isqrt(matrix.shape[0]), math.isqrt(matrix.shape[0]))
    
    sum_reads = np.sum(matrix)
    matrix = np.divide(matrix, sum_reads) * library_size
    matrix = np.log1p(matrix)
    matrix = matrix/np.max(matrix)
    
    return matrix



def log2_norm(matrix):
    # Vectorization trick handler
    if len(matrix.shape) ==1 and is_perfect_square(matrix.shape[0]):
        matrix = matrix.reshape(math.isqrt(matrix.shape[0]), math.isqrt(matrix.shape[0]))
        
    return np.log2(1+np.abs(matrix)) * np.sign(matrix)

def log10_norm(matrix):
    # Vectorization trick handler
    if len(matrix.shape) ==1 and is_perfect_square(matrix.shape[0]):
        matrix = matrix.reshape(math.isqrt(matrix.shape[0]), math.isqrt(matrix.shape[0]))
    return np.log10(1+np.abs(matrix)) * np.sign(matrix)

def zscore_norm(matrix):
    # Vectorization trick handler
    if len(matrix.shape) ==1 and is_perfect_square(matrix.shape[0]):
        matrix = matrix.reshape(math.isqrt(matrix.shape[0]), math.isqrt(matrix.shape[0]))
    
    v = matrix.reshape((-1))
    if not (v == v[0]).all():
        matrix = zscore(v).reshape((len(matrix), -1))
    return matrix

def quantile_norm(matrix, n_q=250, dist='uniform', clipping=None):
    # Vectorization trick handler
    if len(matrix.shape) ==1 and is_perfect_square(matrix.shape[0]):
        matrix = matrix.reshape(math.isqrt(matrix.shape[0]), math.isqrt(matrix.shape[0]))
    
    matrix[~np.isnan(matrix)] = QuantileTransformer(n_quantiles=n_q, output_distribution=dist).fit_transform(matrix[~np.isnan(matrix)].reshape((-1, 1))).reshape((-1))
    matrix = QuantileTransformer(n_quantiles=n_q, output_distribution=dist).fit_transform(matrix)
    
    if clipping is not None:
        matrix[matrix > clipping] = clipping
        matrix[matrix < -clipping] = -clipping

    return matrix 


def sqrt_norm(matrix):
    # Vectorization trick handler
    if len(matrix.shape) ==1 and is_perfect_square(matrix.shape[0]):
        matrix = matrix.reshape(math.isqrt(matrix.shape[0]), math.isqrt(matrix.shape[0]))
    
    coverage = (np.sqrt(np.sum(matrix, axis=-1)))

    with np.errstate(divide='ignore', invalid='ignore'):
        matrix = matrix / coverage.reshape((-1, 1))
        matrix = matrix / coverage.reshape((1, -1))

    matrix[np.isnan(matrix)] = 0.0
    matrix[np.isinf(matrix)] = 0.0

    return matrix

def no_norm(matrix):
    # Vectorization trick handler
    if len(matrix.shape) ==1 and is_perfect_square(matrix.shape[0]):
        matrix = matrix.reshape(math.isqrt(matrix.shape[0]), math.isqrt(matrix.shape[0]))
    
    return matrix



def removeRowCSR(mat, i):
	if not isinstance(mat, sps.csr_matrix):
		raise ValueError("works only for CSR format -- use .tocsr() first")
	n = mat.indptr[i + 1] - mat.indptr[i]
	if n > 0:
		mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i + 1]:]
		mat.data = mat.data[:-n]
		mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i + 1]:]
		mat.indices = mat.indices[:-n]
	mat.indptr[i:-1] = mat.indptr[i + 1:]
	mat.indptr[i:] -= n
	mat.indptr = mat.indptr[:-1]
	mat._shape = (mat._shape[0] - 1, mat._shape[1])



def dropcols_coo(M, idx_to_drop):
	idx_to_drop = np.unique(idx_to_drop)
	C = M.tocoo()
	keep = ~np.in1d(C.col, idx_to_drop)
	C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
	C.col -= idx_to_drop.searchsorted(C.col)  # decrement column indices
	C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
	return C.tocsr()


def removeZeroDiagonalCSR(mtx, i=0, toRemovePre=None):
	iteration = 0
	toRemove = []
	ctr = 0

	if toRemovePre is not None:
		for items in toRemovePre:
			toRemove.append(items)

	if i == 0:
		diagonal = mtx.diagonal()
		#        print diagonal
		for values in diagonal:
			if values == 0:
				toRemove.append(ctr)
			ctr += 1

	else:
		rowSums = mtx.sum(axis=0)
		rowSums = list(np.array(rowSums).reshape(-1, ))
		rowSums = list(enumerate(rowSums))
		for value in rowSums:
			if int(value[1]) == 0:
				toRemove.append(value[0])
				rowSums.remove(value)
		rowSums.sort(key=lambda tup: tup[1])
		size = len(rowSums)
		perc = i / 100.0
		rem = int(perc * size)
		while ctr < rem:
			toRemove.append(rowSums[ctr][0])
			ctr += 1
	list(set(toRemove))
	toRemove.sort()
	# print toRemove
	mtx = dropcols_coo(mtx, toRemove)
	for num in toRemove:
		if iteration != 0:
			num -= iteration
		removeRowCSR(mtx, num)
		iteration += 1
	return [mtx, toRemove]


def knightRuizAlg(A, tol=1e-8, f1=False):
    n = A.shape[0]
    e = np.ones((n, 1), dtype=np.float64)
    res = []

    Delta = 3
    delta = 0.1
    x0 = np.copy(e)
    g = 0.9

    etamax = eta = 0.1
    stop_tol = tol * 0.5
    x = np.copy(x0)

    rt = tol ** 2.0
    v = x * (A.dot(x))
    rk = 1.0 - v
    #    rho_km1 = np.dot(rk.T, rk)[0, 0]
    rho_km1 = ((rk.transpose()).dot(rk))[0, 0]
    rho_km2 = rho_km1
    rout = rold = rho_km1

    MVP = 0  # we'll count matrix vector products
    i = 0  # outer iteration count

    k = 0
    while rout > rt:  # outer iteration
        i += 1

        if i > 30:
            break

        k = 0
        y = np.copy(e)
        innertol = max(eta ** 2.0 * rout, rt)

        while rho_km1 > innertol:  # inner iteration by CG
            k += 1
            if k == 1:
                Z = rk / v
                p = np.copy(Z)
                rho_km1 = (rk.transpose()).dot(Z)
            else:
                beta = rho_km1 / rho_km2
                p = Z + beta * p

            if k > 10:
                break

            # update search direction efficiently
            w = x * A.dot(x * p) + v * p
            # alpha = rho_km1 / np.dot(p.T, w)[0,0]
            alpha = rho_km1 / (((p.transpose()).dot(w))[0, 0])
            ap = alpha * p
            # test distance to boundary of cone
            ynew = y + ap

            if np.amin(ynew) <= delta:

                if delta == 0:
                    break

                ind = np.where(ap < 0.0)[0]
                gamma = np.amin((delta - y[ind]) / ap[ind])
                y += gamma * ap
                break

            if np.amax(ynew) >= Delta:
                ind = np.where(ynew > Delta)[0]
                gamma = np.amin((Delta - y[ind]) / ap[ind])
                y += gamma * ap
                break

            y = np.copy(ynew)
            rk -= alpha * w
            rho_km2 = rho_km1
            Z = rk / v
            ho_km1 = ((rk.transpose()).dot(Z))[0, 0]
        
        x *= y
        v = x * (A.dot(x))
        rk = 1.0 - v
        # rho_km1 = np.dot(rk.T, rk)[0,0]
        rho_km1 = ((rk.transpose()).dot(rk))[0, 0]
        rout = rho_km1
        MVP += k + 1

        # update inner iteration stopping criterion
        rat = rout / rold
        rold = rout
        res_norm = rout ** 0.5
        eta_o = eta
        eta = g * rat
        if g * eta_o ** 2.0 > 0.1:
            eta = max(eta, g * eta_o ** 2.0)
        eta = max(min(eta, etamax), stop_tol / res_norm)
        if f1:
            res.append(res_norm)

    return [x, i, k]


def kr_normalize(contact):
    # Vectorization trick handler
    if len(contact.shape) ==1 and is_perfect_square(contact.shape[0]):
        contact = contact.reshape(math.isqrt(contact.shape[0]), math.isqrt(contact.shape[0]))
    
    rawMatrix = sps.csr_matrix(contact)
    mtxAndRemoved = removeZeroDiagonalCSR(rawMatrix, toRemovePre=None)
    rawMatrix = mtxAndRemoved[0]
    result = knightRuizAlg(rawMatrix)
    colVec = result[0]
    x = sps.diags(colVec.flatten(), 0, format='csr')
    normalizedMatrix = x.dot(rawMatrix.dot(x))
    contact = np.array(normalizedMatrix.todense())
    return contact




normalizations ={
    'none': no_norm,
    'log2_norm': log2_norm,
    'log10_norm': log10_norm,
    'zscore_norm' : zscore_norm,
    'quantile_norm': quantile_norm,
    'sqrt_norm': sqrt_norm,
    'library_size_normalization': library_size_normalization
}