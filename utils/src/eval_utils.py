import numpy as np


def root_mean_sq_err(src, tgt):
    '''
    Root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : root mean squared error
    '''

    return np.sqrt(np.mean((tgt - src) ** 2))

def mean_abs_err(src, tgt):
    '''
    Mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : mean absolute error
    '''

    return np.mean(np.abs(tgt - src))

def inv_root_mean_sq_err(src, tgt):
    '''
    Inverse root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : inverse root mean squared error
    '''

    return np.sqrt(np.mean(((1.0 / tgt) - (1.0 / src)) ** 2))

def inv_mean_abs_err(src, tgt):
    '''
    Inverse mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : inverse mean absolute error
    '''

    return np.mean(np.abs((1.0 / tgt) - (1.0 / src)))

def abs_rel_err(src, tgt):
    '''
    Absolute relative error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : absolute relative error
    '''

    return np.mean(np.abs(src - tgt) / tgt)

def sq_rel_err(src, tgt):
    '''
    Squared relative error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : squared relative error
    '''

    return np.mean(((src - tgt) ** 2) / tgt)

def ratio_out_thresh_err(d, gt, tau=1.25):

    a_ratio = np.maximum((gt/d), (d/gt))
    a_mean = np.mean((a_ratio < tau))
    return a_mean

def root_mean_sq_err_log(src, tgt):
    err = np.log(src) - np.log(tgt)
    rmse_log = np.sqrt(np.mean(err ** 2))
    return rmse_log