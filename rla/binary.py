'''
Functions for working with (random) binary variables.
'''

def binary_matrix(shape, one_prob):
    ''' Create a random binary variable.
    Input:
        one_prob (float) in (0,1) range defining the probability of a '1' 
        to occur.
    '''
    if one_prob >= 1 or one_prob <= 0:
        raise ValueError('probability not in (0,1])')
    
    zero_prob = 1.0 - one_prob
    return  np.random.choice([0, 1], size=shape, p=[zero_prob, one_prob])
