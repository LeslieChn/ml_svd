#!/usr/bin/env/python3

import numpy as np
import pandas as pd
import scipy.linalg as la
from project import SVD

# a = np.matrix([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 1, 1, 1, 1])

b = np.random.rand(5,6)

U, S, V = SVD(b)
LU, LS, LV = la.svd(b)

#print("this is our S:", S)
#print("this is lib S:", LS)
print("this is our U:", U)
print("this is lib U:", LU)
print("this is our V:", V)
print("this is lib V:", LV)

# print("linear norm", la.norm(V-LV))
