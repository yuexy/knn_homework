# coding=utf-8

import numpy as np

def l1_dist(a1, a2):
    return np.sum(np.abs(a1 - a2))

def l2_dist(a1, a2):
    return np.sqrt(np.sum((a1 - a2)**2))
