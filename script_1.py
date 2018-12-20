# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 15:06:51 2018

@author: Douwe
"""
#test comment
import numpy as np
import scipy as sp

h = 1/3

A1 = np.eye(4)*(h**2)
A2 = np.array([[h**2,0,0,0],[-1,4,-1,0],[0,-1,4,-1],[0,0,0,h**2]])
A3 = np.eye(4)
A3[0,0] = 0
A3[3,3] = 0

Asub = np.block([[A2, A3],
                 [A3, A2]])


Atop = np.block([[A1, np.zeros((4,8))],
                 [np.zeros((8,4)), Asub]])

A = (1/(h**2))*np.block([[Atop, np.zeros((12,4))],
              [np.zeros((4,12)), A1]])

eigA = np.linalg.eig(A)

print('Eigenvalues of A: ' + str(eigA))