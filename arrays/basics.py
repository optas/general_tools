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


def scale(array, v_min=0, v_max=1):
    """ Linearly rescale array to specific maximum and minimum values.
    """
    if v_min >= v_max:
        raise ValueError('vmax must be strictly larger than vmin.')
    
    a_max = np.max(array)
    a_min = np.min(array)
    
    if a_max == a_min:
        warnings.warn('An array with all values being the same, cannot be scaled')
        return array
        
#     res = vmax - (((vmax - vmin) * (amax - array)) / (amax - amin))
    
    w = (v_max - v_min) / (a_max - a_min)
    b = v_max - w * a_max 
    res = w * x + b 
     
    
    cond = np.all(abs(v_max - res) < 10e-6) and np.all(abs(res - v_min) > 10e-6)
        
    if not cond:
        warnings.warn('Scaling failed at granulatiry of 10e-6.')
    
    return res
