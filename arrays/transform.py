'''
Created on December 23, 2016

@author: optas
'''

import numpy as np
from . import is_true

def make_contiguous(array):
    ''' The array will be transformed inline, to contain integers from [0,max_i) where max_i is the number of unique integers it contains.
    The relative order of the integers will remain.   
    '''
    if not is_true.is_integer(array):            
        raise ValueError('Cannot transform an non integer array to be contiguous.')
    
    a1 = np.argsort(array) # TODO - Remove
    uvalues = np.unique(array)
    d = {key: value for (value, key) in enumerate(uvalues)}
    for i, val in enumerate(array):
        array[i] = d[array[i]]        
    a2 = np.argsort(array) # TODO - Remove
    
    if not (np.all(a1==a2)): # TODO - Remove
        assert(False)
    
    return array 
    