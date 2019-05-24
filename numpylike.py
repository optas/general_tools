"""
Created on January 26, 2018

@author: optas
"""

from __future__ import division

from past.utils import old_div
import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x.
    subtract max too avoid after exponentiation numerical issues.
    """
    e_x = np.exp(x - np.max(x))
    return old_div(e_x, e_x.sum())
