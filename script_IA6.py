# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 08:12:29 2019

@author: Douwe
"""

import time
start_time = time.time()

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import numpy.linalg as nla
import scipy.sparse as sparse
import scipy.sparse.linalg as splg

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

#y = np.zeros(((N+1)**2, 1))
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

r_h = f_vec - np.dot(A, u)
TOL = (np.linalg.norm(r_h, ord = 2))/f_norm


"""V-Cycle"""
"""calculate M inverse and B of Gauss-Seidel"""
#A_sparse = sparse.csc_matrix(A)

t0 = time.time()

M_inv = la.solve_triangular(A, np.identity((N+1)**2), 0, True)
s = np.dot(M_inv, f_vec)
B_GS = np.identity((N+1)**2) - np.dot(M_inv, A)

t1 = time.time()
print("--- %s seconds ---" % (t1- t0))


#M = np.tril(A)
#M_sparse = sparse.csc_matrix(M)
#
#t0 = time.time()
#M_inv = splg.inv(M_sparse)

#
#t1 = time.time()
#print("--- %s seconds ---" % (t1- t0))
#M_inv = la.solve_triangular(A, np.identity((N+1)**2), 0, True)
#print("--- %s seconds ---" % (time.time() - t1))

"""defining grid operators"""
I_h_2h_block = np.zeros((int(N/2) + 1, N+1))
zeros = np.zeros((int(N/2) + 1, N+1))

for i in range(int(N/2) + 1):
    I_h_2h_block[i, i*2] = 1

I_h_2h = I_h_2h_block
    
for i in range(2, int(N/2)+2):
    I_h_2h = la.block_diag(I_h_2h, np.block([zeros, I_h_2h_block]))
    
    
    

zeros = np.zeros((N+1, (int(N/2) +1)**2))

I_B1 = np.zeros((N+1, int(N/2) + 1))

for i in range(N+1):
    if i%2 == 0:
        I_B1[i, int(i/2)] = 1
    else:
        I_B1[i, int((i-1)/2)] = 1/2
        I_B1[i, int((i-1)/2) + 1] = 1/2

I_2h_h = np.block([[I_B1, zeros[:, int(N/2)+1:]]])

for i in range(int(N/2)):
    I_2h_h = np.block([[I_2h_h],
                       [zeros[:, :i*(int(N/2)+1)], I_B1/2, I_B1/2, zeros[:, (i+2)*(int(N/2)+1):]],
                       [zeros[:, :(i+1)*(int(N/2)+1)], I_B1, zeros[:, (i+2)*(int(N/2)+1):]]])


TOL_vec = [TOL]

#Gauss-Seidel
A_2h = np.dot(I_h_2h, np.dot(A, I_2h_h))

while TOL > 10**(-6) and i < 100*N:
        
    u = np.dot(B_GS, u) + s
    
    r_h = f_vec - np.dot(A, u)
    r_2h = np.dot(I_h_2h, r_h)
    e_2h = nla.solve(A_2h, r_2h)
    e_h = np.dot(I_2h_h, e_2h)
    u = u + e_h
    u = np.dot(B_GS, u) + s
    
    r_h = f_vec - np.dot(A, u)
    
    
    TOL = np.linalg.norm(r_h, ord = 2)/f_norm
    TOL_vec.append(TOL)
        
    i += 1
    

print('num iterations = '+ str(i))
        

print('red of iterations')
for i in reversed(range(1,6)):
    red = TOL_vec[-i]/TOL_vec[-i-1]
    print(red)

plot = 0*X + 0*Y
u_mash =  0*X + 0*Y


print('N = ' + str(N))
print('TOL = '+ str(TOL))


#print(np.shape(z))
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(range(len(TOL_vec)), TOL_vec)
#surf = ax.plot_surface(X, Y, plot, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
title = "Plot of scaled residual versus iteration number \n of the two-grid V-cycle, with N = " + str(N)
ax.set_yscale('log')
ax.set_title(title)
ax.set_xlabel('iteration number')
ax.set_ylabel('scaled residual')
#ax.set_zlabel('z')


print("--- %s seconds ---" % (time.time() - start_time))