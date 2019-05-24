"""
Created on September 8, 2017

@author: optas
"""
from __future__ import division

import numpy as np
import warnings


def unique_rows(array):
    if array.ndim != 2:
        raise ValueError('Unique rows works with 2D arrays only.')
    array = np.ascontiguousarray(array)
    unique_a = np.unique(array.view([('', array.dtype)] * array.shape[1]))
    return unique_a.view(array.dtype).reshape((unique_a.shape[0], array.shape[1]))


def scale(array, vmin=0, vmax=1):
    """ linearly rescale array to specific maximum and minimum values.
    """
    if vmin >= vmax:
        raise ValueError('vmax must be strictly larger than vmin.')
    
    amax = np.max(array)
    amin = np.min(array)
    
    if amax == amin:
        warnings.warn('An array with all values being the same, cannot be scaled')
        return array
        
    res = vmax - (((vmax - vmin) * (amax - array)) / (amax - amin))
     
    
    cond = np.all(abs(vmax - res) < 10e-5) and np.all(abs(res - vmin) > 10e-5)
        
    if not cond:
        warnings.warn('Scaling failed at granulatiry of 10e-5.')
    
    return res
