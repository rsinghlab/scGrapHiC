import math
import cooler
import numpy as np


from src.utils import *
from wrapt_timeout_decorator import * 
from scipy.fftpack import rfft, irfft
from scipy.stats import spearmanr


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


def insulation_score(m, resolution):
    # We compute the insulation vector at three scales
    
    scales = [5, 10, 25] # Even numbers because the smoothing wont work 
    
    tracks = np.zeros((m.shape[0], len(scales)))
    
    for idx, scale in enumerate(scales):
        windowsize = scale*resolution
        tracks[:,idx] = _insulationScore(m, windowsize, resolution)

    return tracks


def insulation_correlation(generated, target, PARAMETERS):
    g_ins_score = insulation_score(generated, PARAMETERS['resolution'])[1]
    t_ins_score = insulation_score(target, PARAMETERS['resolution'])[1]

    return spearmanr(g_ins_score, t_ins_score)[0]



def distance(c1, c2):
    return math.sqrt((float(c1[0]) - float(c2[0]))**2 + (float(c1[1]) - float(c2[1]))**2)


def is_overlapping(coordinate, target_coordinates, rp=0):
    relaxation_parameter = math.sqrt(2*(rp**2))
    try:
        closest = sorted(list(map(lambda x: (x, distance(coordinate, x)), target_coordinates)), key = lambda y: y[1])[0]
    except:
        return ''
    
    coor, d = closest

    if d <= relaxation_parameter:
       return np.array2string(coor, separator=',')
    else:
        return ''

def create_genomic_pixels(dense_matrix, upscale=255):
    """
        Converts a dense matrix into a .bed style sparse matrix file
        @params: dense_matrix <np.array>, input dense matrix
    """
    
    lower_triangular_matrix_coordinates = np.tril_indices(dense_matrix.shape[0], k=-1)
    dense_matrix[lower_triangular_matrix_coordinates] = 0
    
    non_zero_indexes = np.nonzero(dense_matrix)
    bin_ones = non_zero_indexes[0]
    bin_twos = non_zero_indexes[1]
    counts = dense_matrix[np.nonzero(dense_matrix)]*upscale    
    pixels = {
        'bin1_id': bin_ones,
        'bin2_id': bin_twos,
        'count': counts
    }

    pixels = pd.DataFrame(data=pixels)

    return pixels


def create_genomic_bins(
        chromosome_name,
        resolution,
        size
    ):
    """
        The only currently supported type is 'bed' format which is chromosome_id, start, end
        So the function requires input of 'chromosome_name' chromosome name and 'resolution' resolution of of the file. 
        This function also requires size of the chromosome to estimate the maximum number of bins
    """
    chr_names = np.array([chromosome_name]*size)
    starts = (np.arange(0, size, 1, dtype=int))*resolution
    ends = (np.arange(1, size+1, 1, dtype=int))*resolution
    bins = {
        'chrom': chr_names,
        'start': starts,
        'end': ends
    }
    bins = pd.DataFrame(data=bins)
    return bins






def create_cooler_file(matrix, file_identifier, PARAMETERS, output_path):
    output_file = os.path.join(output_path, file_identifier+'.cool')
    
    if os.path.exists(output_file):
        return output_file
    
    
    h, w = matrix.shape
    chrom_pixels = create_genomic_pixels(matrix)
    bins = create_genomic_bins(chr, PARAMETERS['resolution'], h)
    
    cooler.create_cooler(output_file, bins, chrom_pixels,
                    dtypes={"count":"int"}, 
                    assembly="mm10")

    return output_file

def read_chromosight_tsv_file(file_path):
    if os.path.exists(file_path):
        data = open(file_path).read().split('\n')[1:-1]
        data = np.array(list(map(lambda x: [x.split('\t')[i] for i in [6, 7]], data))).astype(np.int64)
        return data
    else: 
        return []



@timeout(60)
def run_chromosight(cooler_file):
    folder = '/'.join(cooler_file.split('/')[:-1])
    file_name = cooler_file.split('/')[-1].split('.')[0]

    borders_output_path = os.path.join(
        folder,
        '{}_borders'.format(file_name)
    )
    
    if not os.path.exists('{}.tsv'.format(borders_output_path)):
        cmd_path = 'chromosight detect --pattern=borders --pearson=0.3 --threads 1 {} {};'.format(
            cooler_file,
            borders_output_path
        )
        os.system(cmd_path)
        
    return borders_output_path+'.tsv'



def overlap_analysis(base, target, rp):
    multi_map = {}
    for coordinate in base:
        coordinate = is_overlapping(coordinate, target, rp)
        if coordinate not in multi_map.keys(): 
            multi_map[coordinate] = 0
            
        multi_map[coordinate] += 1

    fp = multi_map[''] if '' in multi_map.keys() else 0  # When we couldnt find any mapping in target
    fn = len(target)
    tp = 0.000000001
    mm = -1

    for key in multi_map.keys():
        if multi_map[key] >= 1 and key != '':
            tp += 1
            fn -= 1
        if multi_map[key] >= 2:
            mm += 1


    precision = tp/(tp+fp)
    recall =  tp/(tp+fn)
    
    f1 = (2*precision*recall)/(precision + recall)

    accuracy = tp/(tp+fn+fp)

    return precision, recall, f1, accuracy



def tad_sim(generated, target, file_identifier, PARAMETERS, output_path, feature_rp=3):
    generated_cooler_file = create_cooler_file(generated, 'generated_'+file_identifier, PARAMETERS, output_path)
    target_cooler_file = create_cooler_file(target, 'target_'+file_identifier, PARAMETERS, output_path)
    
    generated_tads = run_chromosight(generated_cooler_file)
    target_tads = run_chromosight(target_cooler_file)

    base = read_chromosight_tsv_file(generated_tads)
    target = read_chromosight_tsv_file(target_tads)

    _, _, f1, _ = overlap_analysis(base, target, feature_rp)
    
    return f1



