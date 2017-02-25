'''
Created on February 24, 2017

@author: optas
'''

import operator
import numpy as np


def sort_dict_by_key(in_dict, reverse=False):
    return sorted(in_dict.items(), key=operator.itemgetter(1), reverse=reverse)


def select_first_last_and_k(in_list, k):
    '''select the first and last element of a list among exactly k elements equally spaced
    in the in_list[1:-1]
    '''
    f = in_list[0]
    e = in_list[-1]
    index = np.floor(np.linspace(1, len(in_list) - 2, k)).astype(np.int16)
    res = [in_list[i] for i in index]
    res.insert(0, f)
    res.append(e)
    return res
