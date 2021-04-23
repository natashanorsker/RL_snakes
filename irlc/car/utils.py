"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import numpy as np

def err(x, exception=True, tol=1e-5, message="Error too large!"):
    er = np.mean(np.abs(x).flat)
    if er > tol:
        print(message)
        print(x)
        print(er)
        if exception:
            raise Exception(message)
    return er

