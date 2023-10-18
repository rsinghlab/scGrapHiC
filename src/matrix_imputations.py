import cv2

import numpy as np
import scipy.stats as st

from scipy import ndimage
from scipy import signal, interpolate

# Bilateral filter implememtation from: https://github.com/OzgurBagci/fastbilateral/tree/master
def bilateral(image, sigmaspatial, sigmarange, samplespatial=None, samplerange=None):
    """
    :param image: np.array
    :param sigmaspatial: int
    :param sigmarange: int
    :param samplespatial: int || None
    :param samplerange: int || None
    :return: np.array

    Note that sigma values must be integers.

    The 'image' 'np.array' must be given gray-scale. It is suggested that to use OpenCV.
    """

    height = image.shape[0]
    width = image.shape[1]

    samplespatial = sigmaspatial if (samplespatial is None) else samplespatial
    samplerange = sigmarange if (samplerange is None) else samplerange

    flatimage = image.flatten()

    edgemin = np.amin(flatimage)
    edgemax = np.amax(flatimage)
    edgedelta = edgemax - edgemin

    derivedspatial = sigmaspatial / samplespatial
    derivedrange = sigmarange / samplerange

    xypadding = round(2 * derivedspatial + 1)
    zpadding = round(2 * derivedrange + 1)

    samplewidth = int(round((width - 1) / samplespatial) + 1 + 2 * xypadding)
    sampleheight = int(round((height - 1) / samplespatial) + 1 + 2 * xypadding)
    sampledepth = int(round(edgedelta / samplerange) + 1 + 2 * zpadding)

    dataflat = np.zeros(sampleheight * samplewidth * sampledepth)

    (ygrid, xgrid) = np.meshgrid(range(width), range(height))

    dimx = np.around(xgrid / samplespatial) + xypadding
    dimy = np.around(ygrid / samplespatial) + xypadding
    dimz = np.around((image - edgemin) / samplerange) + zpadding

    flatx = dimx.flatten()
    flaty = dimy.flatten()
    flatz = dimz.flatten()

    dim = flatz + flaty * sampledepth + flatx * samplewidth * sampledepth
    dim = np.array(dim, dtype=int)

    dataflat[dim] = flatimage

    data = dataflat.reshape(sampleheight, samplewidth, sampledepth)
    weights = np.array(data, dtype=bool)

    kerneldim = derivedspatial * 2 + 1
    kerneldep = 2 * derivedrange * 2 + 1
    halfkerneldim = round(kerneldim / 2)
    halfkerneldep = round(kerneldep / 2)

    (gridx, gridy, gridz) = np.meshgrid(range(int(kerneldim)), range(int(kerneldim)), range(int(kerneldep)))
    gridx -= int(halfkerneldim)
    gridy -= int(halfkerneldim)
    gridz -= int(halfkerneldep)

    gridsqr = ((gridx * gridx + gridy * gridy) / (derivedspatial * derivedspatial)) \
        + ((gridz * gridz) / (derivedrange * derivedrange))
    kernel = np.exp(-0.5 * gridsqr)

    blurdata = signal.fftconvolve(data, kernel, mode='same')

    blurweights = signal.fftconvolve(weights, kernel, mode='same')
    blurweights = np.where(blurweights == 0, -2, blurweights)

    normalblurdata = blurdata / blurweights
    normalblurdata = np.where(blurweights < -1, 0, normalblurdata)

    (ygrid, xgrid) = np.meshgrid(range(width), range(height))

    dimx = (xgrid / samplespatial) + xypadding
    dimy = (ygrid / samplespatial) + xypadding
    dimz = (image - edgemin) / samplerange + zpadding

    return interpolate.interpn((range(normalblurdata.shape[0]), range(normalblurdata.shape[1]),
                               range(normalblurdata.shape[2])), normalblurdata, (dimx, dimy, dimz))




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


def neighbor_ave_cpu(A, pad, mask="Bilateral", nsig=1):
    '''
    Convolution smooth
    '''

    if pad == 0:
        return A

    # Remove diagonal
    np.fill_diagonal(A, 0)

    maskLength = pad*2 + 1
    maskKern = []
    if mask == "Mean":
        maskKern = np.array([1]*maskLength*maskLength).reshape(3, 3)
    if mask == "Gaussian":
        maskKern = gkern(maskLength, nsig)
    if mask == "Bilateral":
        return bilateral(A, pad, pad)

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
    for _ in range(30):
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