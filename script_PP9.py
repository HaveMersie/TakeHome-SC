# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:56:03 2019

@author: Douwe
"""

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




N = 8

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

r_vec = f_vec - np.dot(A, u)

i = 0 

TOL_vec = [TOL]

#Gauss-Seidel

iteration = 100
for i in range(iteration):
    for j in range(0, (N+1)**2):
        
        u[j] = (f_vec[j] - np.dot(A[j,:j], u[:j]) - 
                np.dot(A[j, j+1:], u[j+1:]))/A[j,j]
        r_vec = f_vec - np.dot(A, u)
        
#    TOL = np.linalg.norm(f_vec - np.dot(A, u), ord = 2)/f_norm
#    TOL_vec.append(TOL)
        
    i += 1
           

plot = 0*X + 0*Y
u_mash =  0*X + 0*Y
r_vec_mash =  0*X + 0*Y

for i in range((N+1)**2):
    Ix = i%(N+1)
    Iy = int(i/(N+1))
    
    u_ex_vec[i] = u_ex[Ix, Iy]
    plot[Ix, Iy] = u[i] - u_ex[Ix, Iy]
    u_mash[Ix, Iy] = u[i]
    r_vec_mash[Ix, Iy] = r_vec[i]



#fig = plt.figure()
fig1d = plt.figure()

#ax = fig.add_subplot(111, projection='3d')
ax1 = fig1d.add_subplot(211)

#"""plot exact solution""" 
#surf = ax.plot_surface(X, Y, plot, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
#ax.set_title(title)
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')

ax1.plot(range(len(u)), u - u_ex_vec)
#title = "plot of e_h with N = " + str(N) + " and  " + str(iteration) + 

#ax.set_title(title)
ax1.set_xlabel('index')
ax1.set_ylabel('value')
#ax.set_zlabel('z')


"""make Permuation matrix for N even"""

P = np.zeros(((N+1)**2, (N+1)**2))

for i in range(1, (N+1)**2+1):
    if i%2 == 0:

        P[int(((N+1)**2 + 1+i)/2 -1), i-1] = 1
    else:
        P[int((i+1)/2 -1), i-1] = 1



A_RedBlack = np.dot(np.dot(P,A), P.T)

f_vec = np.dot(P, f_vec)

u_rb = np.zeros(((N+1)**2, 1))
 
f_norm = np.linalg.norm(f_vec, ord = 2)

TOL = (np.linalg.norm(f_vec - np.dot(A_RedBlack, u_rb), ord = 2))/f_norm

r_vec = f_vec - np.dot(A_RedBlack, u_rb)

i = 0 

TOL_vec = [TOL]

#Gauss-Seidel


for i in range(iteration):
    for j in range(0, (N+1)**2):
        
        u_rb[j] = (f_vec[j] - np.dot(A_RedBlack[j,:j], u_rb[:j]) - 
                np.dot(A_RedBlack[j, j+1:], u_rb[j+1:]))/A_RedBlack[j,j]
        r_vec = f_vec - np.dot(A_RedBlack, u_rb)
        
#    TOL = np.linalg.norm(f_vec - np.dot(A_RedBlack, u_rb), ord = 2)/f_norm
#    TOL_vec.append(TOL)
        
    i += 1
           

plot = 0*X + 0*Y
u_mash =  0*X + 0*Y
r_vec_mash =  0*X + 0*Y

for i in range((N+1)**2):
    Ix = i%(N+1)
    Iy = int(i/(N+1))
    
    u_ex_vec[i] = u_ex[Ix, Iy]
    plot[Ix, Iy] = u_rb[i] - u_ex[Ix, Iy]
    u_mash[Ix, Iy] = u_rb[i]
    r_vec_mash[Ix, Iy] = r_vec[i]




#ax = fig.add_subplot(111, projection='3d')
ax2 = fig1d.add_subplot(212)

#"""plot exact solution""" 
#surf = ax.plot_surface(X, Y, plot, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
#ax.set_title(title)
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')

ax2.plot(range(len(u_rb)), u_rb - u_ex_vec)
#title = "plot of e_h with N = " + str(N) + " and  " + str(iteration) + 

#ax.set_title(title)
ax2.set_xlabel('index')
ax2.set_ylabel('value')
#ax.set_zlabel('z')
