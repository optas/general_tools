'''
Created on January 17, 2017

@author:    Panos Achlioptas
@contact:   pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''

import numpy as np
from scipy.stats import mode
from scipy import signal

def smooth_normal_outliers(array, dev):
    '''In each row of the input array finds outlier elements and transforms their values.
    An outlier in row[i], is any element of that row that is in magnitude bigger than
    \mu[i] + `dev` times \sigma[i], where \mu[i] is the mean value
    and \sigma the standard deviation of the values of row i.

    Note: It changes the array inline.
    '''

    stds = np.std(array, axis=1)
    means = np.mean(array, axis=1)
    line_elems = np.arange(array.shape[1])
    for i in xrange(array.shape[0]):
        outliers = abs(array[i]) > dev * stds[i] + means[i]
        inliers = np.setdiff1d(line_elems, outliers, assume_unique=False)
        mu_i = np.mean(array[i, inliers])
#         array[i, outliers] = means[i]
        array[i, outliers] = mu_i
    return array


def find_non_homogeneous_vectors(array, thres):
    '''
    '''
    index = []
    n = float(array.shape[1])
    for i, vec in enumerate(array):
        frac = mode(vec)[1][0] / n
        if frac < thres:
            index.append(i)
    return index


def smooth(x, window_len=11, window='hanning'):
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
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def window_average(array, window_size, mode='valid'):
    ''' Runs a sliding window of window_size and reports the average.(stride = 1) .
        valid: all produced values will be from exactly window_size elements (if less trailing remain will be dropped'.
    '''
    kernel = np.ones(window_size)
    conv_out = signal.convolve(array, kernel, mode=mode)
    conv_out /= float(window_size)
    return conv_out