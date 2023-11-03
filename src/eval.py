import numpy as np
import math
from skimage.metrics import structural_similarity
import tensorflow as tf
import scipy.stats as stats

def mse(A:np.ndarray, B:np.ndarray):
    """evaluate the Mean Squared Error (MSE) between two scHi-C contact matrices

    Args:
        A (np.ndarray): a 2-dimensional numpy array representing a scHi-C contact matrix
        B (np.ndarray): a 2-dimensional numpy array representing a scHi-C contact matrix

    Returns:
        a scalar representing the MSE
    """
    if (len(A.shape) != 2) and (len(B.shape) != 2):
        raise ValueError("both input matrices are of the wrong shape")
    elif len(A.shape) != 2:
        raise ValueError("first input matrix is of the wrong shape (" + str(len(A.shape)) + " dimensions instead of 2)")
    elif len(B.shape) != 2:
        raise ValueError("second input matrix is of the wrong shape (" + str(len(B.shape)) + " dimensions instead of 2)")
    elif A.shape != B.shape:
        raise KeyError("matrices not of the same size")
    
    mse = np.square(np.subtract(A, B)).mean()
    if math.isnan(mse):
        raise ValueError("one or more input matrices are empty")
    
    return mse

def ssim(A:np.ndarray, B:np.ndarray, win_size:int=7):
    """evaluate the Structural Similarity Index Measure (SSIM) between two scHi-C contact matrices

    Args:
        A (np.ndarray): a 2-dimensional numpy array representing a scHi-C contact matrix
        B (np.ndarray): a 2-dimensional numpy array representing a scHi-C contact matrix
        win_size (int): an odd integer representing the window size for SSIM. Must be less than size of A and B

    Returns:
        a scalar representing the SSIM, normalized to remove negative values
    """
    if (len(A.shape) != 2) and (len(B.shape) != 2):
        raise ValueError("both input matrices are of the wrong shape")
    elif len(A.shape) != 2:
        raise ValueError("first input matrix is of the wrong shape (" + str(len(A.shape)) + " dimensions instead of 2)")
    elif len(B.shape) != 2:
        raise ValueError("second input matrix is of the wrong shape (" + str(len(B.shape)) + " dimensions instead of 2)")
    elif A.shape != B.shape:
        raise KeyError("matrices not of the same size")
    
    ssim = structural_similarity(A, B, multichannel=False, win_size=win_size)
    # ssim = tf.image.ssim(A, B, max_val=max(np.max(A), np.max(B)))
    if math.isnan(ssim):
        raise ValueError("one or more input matrices are empty")
    
    return (ssim + 1) / 2

def kendall_tau(A:np.ndarray, B:np.ndarray, win_size:int=7):
    """evaluate the Kendall's Tau Correlation between two scHi-C contact matrices

    Args:
        A (np.ndarray): a 2-dimensional numpy array representing a scHi-C contact matrix
        B (np.ndarray): a 2-dimensional numpy array representing a scHi-C contact matrix
        win_size (int): an odd integer representing the window size for SSIM. Must be less than size of A and B

    Returns:
        a tuple of scalars representing kendall's tau (normalized to remove negative values) and the p value
    """
    if (len(A.shape) != 2) and (len(B.shape) != 2):
        raise ValueError("both input matrices are of the wrong shape")
    elif len(A.shape) != 2:
        raise ValueError("first input matrix is of the wrong shape (" + str(len(A.shape)) + " dimensions instead of 2)")
    elif len(B.shape) != 2:
        raise ValueError("second input matrix is of the wrong shape (" + str(len(B.shape)) + " dimensions instead of 2)")
    elif A.shape != B.shape:
        raise KeyError("matrices not of the same size")
    
    tau, p_val = stats.kendalltau(A, B)
    if math.isnan(tau):
        raise ValueError("one or more input matrices are empty")
    
    return ((1 + tau) / 2, p_val)