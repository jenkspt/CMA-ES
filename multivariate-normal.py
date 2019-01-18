import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


if __name__ == "__main__":
     
    # Sample From Multivariate Gaussian
    n = 1000      # Number of examples
    d = 2         # Number of dimensions
    I = np.identity(d)

    U = np.random.normal(0, 1, size=(n, d))
    # Calculate Covariance Matrix
    mu = U.mean(0)
    C = (U.T @ U)/(n-1) - mu @ mu
    # Calculate Eigenvalue decomposition
    w, E = la.eig(C)
    #assert np.allclose(np.diag(w), I, .2)
    
    # Create somewhat random covariance matrix
    C = I + np.random.normal(0,2, (d,d))
    C = (C + C.T)/2     # make it symmetric

    w, E = la.eig(C)
    # Sample from it
    X = (E @ np.diag(np.sqrt(w)) @ np.random.normal(size=(d,n))).T

    plt.scatter(X[:,0], X[:,1])
    plt.scatter(U[:,0], U[:, 1])

