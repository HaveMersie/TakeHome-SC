# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:24:05 2018

@author: Douwe
"""

"""SC IA 1"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = np.linspace(0, 1, 101)
Y = np.linspace(0, 1, 101)

X, Y = np.meshgrid(X, Y)

Z1 = X**3*(1-X) + Y*(1-Y) + np.exp(Y)
Z2 = np.exp(X+Y)
#print(np.shape(z))
surf = ax.plot_surface(X, Y, Z1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')