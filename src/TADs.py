import numpy as np
from scipy.fftpack import rfft, irfft






def smooth(x, window_len=10, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    x = np.copy(x)
    if x.ndim != 1:
        print("smooth only accepts 1 dimension arrays.")
        raise EOFError
    
    if x.size < window_len:
        print("Input vector needs to be bigger than window size.")
        raise EOFError
    
    if window_len < 3:
        return x
    
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        print("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        raise EOFError
    
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    
    y = np.convolve(w / w.sum(), s, mode='valid')
    
    # print(int(window_len/2-1), int(window_len/2))

    return y[int(window_len/2-1):-int(window_len/2)]



def smooth_data_fft(arr, span):  # the scaling of "span" is open to suggestions
    w = rfft(arr)
    spectrum = w ** 2
    cutoff_idx = spectrum < (spectrum.max() * (1 - np.exp(-span / 2000)))
    w[cutoff_idx] = 0
    return irfft(w)



def insulation_score(m, resolution):
    # We compute the insulation vector at three scales
    
    scales = [5, 10, 25] # Even numbers because the smoothing wont work 
    
    tracks = np.zeros((m.shape[0], len(scales)))
    
    for idx, scale in enumerate(scales):
        windowsize = scale*resolution
        tracks[:,idx] = _insulationScore(m, windowsize, resolution)

    return tracks






def _insulationScore(m, windowsize=500000, res=40000):
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
    score = smooth_data_fft(score, 4)
    
    max_score = np.max(score)
    score = np.divide(score, max_score)
    
    return score