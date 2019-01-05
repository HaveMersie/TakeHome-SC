# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:23:53 2018

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
import scipy.sparse as sparse

fig = plt.figure()
ax = fig.add_subplot(111)

N = 16

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

#A_sparse = sparse.lil_matrix(A)

y = np.zeros(((N+1)**2, 1))
u = np.zeros(((N+1)**2, 1))
u_ex_vec = np.zeros(((N+1)**2, 1))

f_vec = np.zeros(((N+1)**2, 1))
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
        f_vec[i] = g_func[Ix, Iy]

        
    elif I == (1,1) :
        f_vec[i] = f_func[I] + (g_func[1,0] + g_func[0,1])/(h**2)

    elif I == (N-1,1):
        f_vec[i] = f_func[I] + (g_func[N-1,0] + g_func[N,1])/(h**2)

    elif I == (1, N-1):
        f_vec[i] = f_func[I] + (g_func[0,N-1] + g_func[1,N])/(h**2)

    elif I == (N-1, N-1):
        f_vec[i] = f_func[I] + (g_func[N-1,N] + g_func[N,N-1])/(h**2)

    elif Ix == 1   and 1 < Iy < N-1:
        f_vec[i] = f_func[I] + (g_func[0,Iy])/(h**2)

    elif Ix == N-1 and 1 < Iy < N-1:
        f_vec[i] = f_func[I] + (g_func[N,Iy])/(h**2)

    elif Iy == 1   and 1 < Ix < N-1:
        f_vec[i] = f_func[I] + (g_func[Ix,0])/(h**2)

    elif Iy == N-1 and 1 < Ix < N-1:
        f_vec[i] = f_func[I] + (g_func[Ix,N])/(h**2)

    else:
        f_vec[i] = f_func[I]


u = np.zeros(((N+1)**2, 1))
 
f_norm = np.linalg.norm(f_vec, ord = 2)

TOL = (np.linalg.norm(f_vec - np.dot(A, u), ord = 2))/f_norm


i = 0 

TOL_vec = [TOL]

#Gauss-Seidel
while TOL > 10**(-6) and i < 100*N:
    for j in range(0, (N+1)**2):
        
        u[j] = (f_vec[j] - np.dot(A[j,:j], u[:j]) - 
                np.dot(A[j, j+1:], u[j+1:]))/A[j,j]
#        r_vec = f_vec - np.dot(A, u)
        
    TOL = np.linalg.norm(f_vec - np.dot(A, u), ord = 2)/f_norm
    TOL_vec.append(TOL)
        
    i += 1
    

print('num iterations = '+ str(i))
        

print('red of iterations')
for i in reversed(range(1,6)):
    red = TOL_vec[-i]/TOL_vec[-i-1]
    print(red)

plot = 0*X + 0*Y
u_mash =  0*X + 0*Y

#for k in range((N+1)**2):
#    Ix = k%(N+1)
#    Iy = int(k/(N+1))
#    
#    u_ex_vec[k] = u_ex[Ix, Iy]
#    plot[Ix, Iy] = u[k] - u_ex[Ix, Iy]
#    u_mash[Ix, Iy] = u[k]
    
#M = np.linalg.norm((u_ex_vec - u), ord = inf)
#print(np.linalg.norm((u_ex_vec), ord = inf))
#print(np.linalg.norm((u), ord = inf))

print('N = ' + str(N))
print('TOL = '+ str(TOL))


#print(np.shape(z))
ax.plot(range(len(TOL_vec)), TOL_vec)
#surf = ax.plot_surface(X, Y, plot, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
title = "Plot of scaled residual versus iteration number \n of Gauss-Seidel with lexicografic ordering, with N = " + str(N)
ax.set_yscale('log')
ax.set_title(title)
ax.set_xlabel('iteration number')
ax.set_ylabel('scaled residual')
#ax.set_zlabel('z')
#file = open(r'D:\Documenten\Studie\MASTER\Scientific Computing\Take home\GitHub\saveN64.txt', 'w')
#
#file.write('num iterations = '+ str(i) + '\n')
#file.write('red of iterations' + '\n')
#for i in reversed(range(1,6)):
#    red = TOL_vec[-i]/TOL_vec[-i-1]
#    file.write(str(red) + '\n')
#
#file.write('N = ' + str(N) + '\n')
#file.write('TOL = '+ str(TOL) + '\n')


#file.close()

print("--- %s seconds ---" % (time.time() - start_time))