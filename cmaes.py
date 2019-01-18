import numpy as np
import numpy.linalg as la


from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cma

np.random.seed(0)

"""
The Algorithm

Paramters
Population size  : N
Covariance Matrix: C
mean of best k   : mu

1. Sample N from multivariate normal distribution
2. Calculate fitness and 

"""

def rastrigin(X, A=10):
    return A + np.sum((X**2 - A * np.cos(2 * np.pi * X)), -1)

def rosenbrock(X, a=1, b=100):
    x,y = X[...,0], X[...,1]
    return (a-x)**2 + b*(y-x**2)**2
    

def _plot_points(X, low=-10, high=10, size=1024):
    """ Plots the points on same scale as image """
    tmp = (X - low)/(high-low) * size
    plt.scatter(tmp[:,0], tmp[:,1], color='black', alpha=.5)

if __name__ == '__main__':

    low, high, size = -6, 6, 2048
    spacing = np.linspace(low, high, size)    

    grid = np.stack(np.meshgrid(spacing, spacing), -1)
    function = lambda X: rastrigin(X + [3,2])
    #function = rosenbrock
    Z = function(grid)
    
    #plt.imshow(Z); plt.show()

     
    n = 100     # Population size
    d = 2       # Dimensions
    k = 25      # Size of elite population

    X = np.random.normal(0,1.28, (d, n))

    for i in range(24):
        # Minimize this function
        fitness = function(X.T)
        arg_topk = np.argsort(fitness)[:k]
        topk = X[:,arg_topk]

        #print(f'Iter {i}, score {fitness[arg_topk[0]]}, X = {X[:,arg_topk[0]]}')
        # Covariance of topk but using mean of entire population
        centered = topk - X.mean(1, keepdims=True)
        C = (centered @ centered.T)/(k-1)
        # Eigenvalue decomposition
        w, E = la.eigh(C)
        # Generate new population
        # Sample from multivariate gaussian with mean of topk
        N = np.random.normal(size=(d,n))
        X = topk.mean(1,keepdims=True) + (E @ np.diag(np.sqrt(w)) @ N)
        if i % 1 == 0:
            print(f'iter {i}, z= {fitness[arg_topk[0]]:.2f}, x= {X[:, arg_topk[0]].round(2)}')
            plt.clf()
            plt.imshow(Z, cmap='Oranges')
            _plot_points(X.T, low, high, size)
            plt.pause(.2)
            plt.draw()
            plt.savefig(f'plots/fig-{i}.png')
    """
    es = cma.CMAEvolutionStrategy([0]*2, 1, {'popsize': n})
    for i in range(50):
        X = np.array(es.ask())
        fitness = function(X)
        es.tell(X, fitness)
        if i % 4 == 0:
            print(es.result[0])
            print(fitness.max())
            plt.clf()
            plt.imshow(Z)
            _plot_points(X)
            plt.pause(.2)
            plt.draw()

    """
