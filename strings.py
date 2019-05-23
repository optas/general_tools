"""
Created on May 6, 2017

@author: optas
"""

import string
import numpy as np
import warnings


def file_ending(in_string):
    """file.jpeg, returns jpeg
    """
    index = in_string[::-1].find('.')
    if index < 1:
        raise ValueError('string does not contain a dot (.) or content after it.')
    return in_string[-index:]


def trim_content_after_last_dot(s):
    """Example: if s = myfile.jpg.png, returns myfile.jpg
    """
    index = s[::-1].find('.') + 1
    s = s[:len(s) - index]
    return s


def random_alphanumeric(n_chars):
    character_pool = string.ascii_uppercase + string.ascii_lowercase + string.digits
    array_pool = np.array([c for c in character_pool])
    res = ''.join(np.random.choice(array_pool, n_chars, replace=True))    
    return res
