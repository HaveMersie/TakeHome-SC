# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:53:18 2018

@author: Douwe
"""
import time
start_time = time.time()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from copy import copy
import pprint
import scipy
import scipy.linalg 
import numpy.linalg as la
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

N = 256

h = 1/N

"""Create A"""
Ih = np.identity(N+1)
Ih[0,0] = 0
Ih[N,N] = 0

vdiag = 4*np.ones(N-1)
voffs = -1*np.ones(N-2)
zeros = np.zeros((1,N-1))

Th = np.diag(vdiag)
np.fill_diagonal(Th[:,1:], voffs)
np.fill_diagonal(Th[1:], voffs)


Th = np.block([[h**2, zeros, 0],
               [zeros.T, Th, zeros.T],
               [0, zeros, h**2]])
#print(Th)

zN = np.zeros((N+1, (N+1)**2 - (N+1)))
zShort = np.zeros((N+1, N+1))
#print(np.shape(zN))
I = h**2*np.identity(N+1)
block1 = np.block([[I, zN]])
block2 = np.block([[zShort, Th, -Ih, zN[:, 2*(N+1):]]])

A = np.block([[block1],[block2]])

for i in range(3, N):

    row = np.block([[zN[:, :(i-2)*(N+1)], 
                     -Ih, Th, -Ih, 
                     zN[:, :(N-i)*(N+1)]]])
    

    A = np.block([[A],[row]])

blockN = np.block([[zN, I]])
blockN1 = np.block([[zN[:, 2*(N+1):], -Ih, Th, zShort]]) 
    
A = np.block([[A],[blockN1],[blockN]])/(h**2)

#L = copy(A)
#B = np.zeros(((N+1)**2, (N+1)**2))
#for k in range(N-1):
#    for i in range(k+1, N+1):
#        B[i,k] = L[i,k]/L[k,k]
#        L[i,k] = B[i,k]
#        for j in range(k+1, N+1):
#            L[i,j] = L[i,j] - B[i,k]*L[k,j]
#            

P, L, U = scipy.linalg.lu(A)

#print("A:")
#pprint.pprint(A)
#
#print( "P:")
#pprint.pprint(P.diagonal())
#
#print( "L:")
#pprint.pprint(L.diagonal())
#
#print( "U:")
#pprint.pprint(U)

y = np.zeros(((N+1)**2, 1))
u = np.zeros(((N+1)**2, 1))
u_ex_vec = np.zeros(((N+1)**2, 1))

f = np.zeros(((N+1)**2, 1))
g = np.zeros(((N+1)**2, 1))

X = np.linspace(0, 1, N+1)
Y = np.linspace(0, 1, N+1)

X, Y = np.meshgrid(X, Y)

u_ex = X**3*(1-X) + Y*(1-Y) + np.exp(Y)
g_func = u_ex
f_func = 12*X**2 - 6*X + 2 - np.exp(Y)


for i in range((N+1)**2):
    Ix = i%(N+1)
    Iy = int(i/(N+1))
    I = (Ix, Iy)
    
    if Ix == 0 or Iy == 0 or Ix == N or Iy == N:
        """Boundary"""
        f[i] = g_func[Ix, Iy]

        
    elif I == (1,1) :
        f[i] = f_func[I] + (g_func[1,0] + g_func[0,1])/(h**2)

    elif I == (N-1,1):
        f[i] = f_func[I] + (g_func[N-1,0] + g_func[N,1])/(h**2)

    elif I == (1, N-1):
        f[i] = f_func[I] + (g_func[0,N-1] + g_func[1,N])/(h**2)

    elif I == (N-1, N-1):
        f[i] = f_func[I] + (g_func[N-1,N] + g_func[N,N-1])/(h**2)

    elif Ix == 1   and 1 < Iy < N-1:
        f[i] = f_func[I] + (g_func[0,Iy])/(h**2)

    elif Ix == N-1 and 1 < Iy < N-1:
        f[i] = f_func[I] + (g_func[N,Iy])/(h**2)

    elif Iy == 1   and 1 < Ix < N-1:
        f[i] = f_func[I] + (g_func[Ix,0])/(h**2)

    elif Iy == N-1 and 1 < Ix < N-1:
        f[i] = f_func[I] + (g_func[Ix,N])/(h**2)

    else:
        f[i] = f_func[I]

#for i in range((N+1)**2):
#    y[i] = f[i] - np.dot(L[i, :i], f[:i])
#
#plot_dif = 0*X + 0*Y
#dif = np.zeros(((N+1)**2, 1))
#
#for i in range((N+1)**2):
#    Ix = i%(N+1)
#    Iy = int(i/(N+1))
#    
#    u[i] = (y[i] - np.dot(U[i, i+1:], y[i+1:]))/U[i,i]
#    
#    dif[i] = u[i] - u_ex[Ix, Iy]
#    plot_dif[Ix, Iy] = u[i]# - u_ex[Ix, Iy]
#    
    
u = la.solve(A, f)

plot = 0*X + 0*Y
u_mash =  0*X + 0*Y

for i in range((N+1)**2):
    Ix = i%(N+1)
    Iy = int(i/(N+1))
    
    u_ex_vec[i] = u_ex[Ix, Iy]
    plot[Ix, Iy] = u[i] - u_ex[Ix, Iy]
    u_mash[Ix, Iy] = u[i]
    
M = np.linalg.norm((u_ex_vec - u), ord = inf)
#print(np.linalg.norm((u_ex_vec), ord = inf))
#print(np.linalg.norm((u), ord = inf))

print(M)

#print(np.shape(z))
surf = ax.plot_surface(X, Y, plot, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
title = "plot of (uh - u_ex) with N = " + str(N)
ax.set_title(title)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

print("--- %s seconds ---" % (time.time() - start_time))