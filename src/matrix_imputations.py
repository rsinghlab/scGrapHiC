import numpy as np
from scipy import ndimage
import scipy.stats as st


# convolution-randomwalk algorithm from schicluster
def gkern(kernlen=3, nsig=1):
    '''
    Returns a 2D Gaussian kernel array.
    '''

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def neighbor_ave_cpu(A, pad, mask="Gaussian", nsig=1):
    '''
    Convolution smooth
    '''

    if pad == 0:
        return A

    maskLength = pad*2 + 1
    maskKern = []
    if mask == "Mean":
        maskKern = np.array([1]*maskLength*maskLength).reshape(3, 3)
    if mask == "Gaussian":
        maskKern = gkern(maskLength, nsig)
    if mask == "Bilateral":
        pass

    return (ndimage.convolve(A, maskKern, mode='constant', cval=0.0) / float(maskLength * maskLength))


def random_walk_cpu(A, rp):
    """
    description:
    input:matrix,restart probability
    output:matrix after random walk
    """

    ngene, _ = A.shape
    A = A - np.diag(np.diag(A))
    A = A + np.diag(np.sum(A, axis=0) == 0)
    P = np.divide(A, np.sum(A, axis=0))
    Q = np.eye(ngene)
    I = np.eye(ngene)
    for i in range(30):
        Q_new = (1 - rp) * I + rp * np.dot(Q, P)
        delta = np.linalg.norm(Q - Q_new)
        Q = Q_new.copy()
        if delta < 1e-6:
            break
    return Q


def matImpute(A, pad, rp, mask='Gaussian'):
    """
    description: run comvolution and random walk for a given matrix
    input: matrix_to_impute,convolution_pad_size,restart_probability_for_random_walk,convolution_mask_type
    """
    A = neighbor_ave_cpu(A, pad, mask)
    if rp == -1:
        Q = A[:]
    else:
        Q = random_walk_cpu(A, rp)

    return Q