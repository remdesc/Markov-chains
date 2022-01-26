#import
import numpy as np
from scipy.stats import binom
from random import *
import matplotlib.pyplot as plt
from numpy import linalg as la


def ehrenfest(n_max, K, mu):
    """simulation of Ehrenfest model (exercise 1)

    Args:
        n_max: maximum number of time steps.
        K: number of particles.
        mu: initial distribution.

    Returns:
        nb_particules: an array with the number of particles in compartment 0 at each step.

    """
    
    A = mu.copy() #compartment 0
    B = (1-mu).copy() #compartment 1
    nb_particules = np.zeros(n_max)
    n = 0
    while n < n_max :
        nombre = randint(1,K)
        if A[nombre-1]==1:
            A[nombre-1]=0
            B[nombre-1]=1
        else :
            B[nombre-1]=0
            A[nombre-1]=1
        n+=1
        nb_particules[n-1]=np.count_nonzero(A == 1)
    return nb_particules




def ehrenfest_return_time(n_max, K, mu):
    """simulation of the return time to initial distribution with Ehrenfest model (exercise 1)

    Args:
        n_max: maximum number of time steps.
        K: number of particles.
        mu: initial distribution.

    Returns:
        n: the number of time steps before returning to initial distribution, or False if the initial distribution is not reached before n_max time steps.

    """
    
    nb_distrib = np.count_nonzero(mu == 1)
    A = mu.copy() #compartment 0
    B = (1-mu).copy() #compartment 1
    n = 0
    while n < n_max :
        nombre = randint(1,K)
        if A[nombre-1]==1:
            A[nombre-1]=0
            B[nombre-1]=1
        else :
            B[nombre-1]=0
            A[nombre-1]=1
        n+=1
        if np.count_nonzero(A == 1) == nb_distrib :
            return n
    return False





def simulate_dthmc(P, mu, n_max):
    """simulation of a discrete time homogeneous Markov chain (exercise 2)

    Args:
        n_max: maximum number of time steps.
        P: transition matrix.
        mu: initial distribution.

    Returns:
        trajectory: an array with the states at each time step until n_max.

    """
    
    trajectory = np.zeros(n_max)
    trajectory[0]=np.argwhere(mu==1)[0][0]
    n = 0
    states = np.arange(0,6)
    while n < n_max-1 :
        i = np.argwhere(mu==1)[0][0]
        new_state = np.random.choice(states, p = P[i,:])
        mu = np.zeros(6)
        mu[new_state] = 1
        n+=1
        trajectory[n]=new_state
    return trajectory






def simulate_dthmc_return_time(P, init, n_max):
    """simulation of the return time to initial distribution for a discrete time homogeneous Markov chain (exercise 2)

    Args:
        n_max: maximum number of time steps.
        P: transition matrix.
        mu: initial distribution.

    Returns:
        n: the number of time steps before returning to initial distribution, or False if the initial distribution is not reached before n_max time steps.

    """
    trajectory = np.zeros(n_max)
    trajectory[0]=init
    mu = np.zeros(6)
    mu[init] = 1
    n = 0
    states = np.arange(0,6)
    while n < n_max-1 :
        i = np.argwhere(mu==1)[0][0]
        new_state = np.random.choice(states, p = P[i,:])
        mu = np.zeros(6)
        mu[new_state] = 1
        n+=1
        trajectory[n]=new_state
        if new_state == init :
            return n
    return False