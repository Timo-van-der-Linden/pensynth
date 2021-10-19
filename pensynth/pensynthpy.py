#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions used to run the in-house algorithm to
incrementally compute the Delaunay triangulation

Created on Sun Nov  1 15:03:32 2020
Refactored : 30/07/2021

@author: jeremylhour
"""
import numpy as np
import itertools
import time
from typing import Tuple
from numba import njit
from scipy.spatial.distance import cdist
from scipy.optimize import linprog
from cvxopt import matrix, solvers
from scipy.spatial import Delaunay


# ------------------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------------------
def get_ranks(node: np.array, nodes: np.array) -> Tuple[np.ndarray]:
    """
    get_ranks:
        returns the ranks and anti-ranks of nodes by rank in closeness to node

    @param node (np.array): point for which we want to find the neighbors
    @param nodes (np.array): points that are candidate neighbors
    """
    distance = cdist(node.reshape(1, -1), nodes, "sqeuclidean")[0]
    anti_ranks = np.argsort(distance)
    ranks = np.argsort(anti_ranks)
    return ranks, anti_ranks


def in_hull(x: np.array, points: np.array) -> bool:
    """
    in_hull:
        test if points in x are in hull

    @param x (np.array): should be a 1 x p coordinates of 1 point in p
    dimensions
    @param points (np.array): the m x p array of the coordinates of m points
    in p dimensions
    
    This function is based on Stack Exchange:
    https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl # noqa
    """
    size = points.shape[0]
    A = np.hstack([points, np.ones((size, 1))]).T
    b = np.append(x, 1)
    c = np.zeros(size)
    result = linprog(c, A_eq=A, b_eq=b)
    return result.success


def compute_radius_and_barycenter(nodes: np.ndarray) -> Tuple[np.ndarray]:
    """
    compute_radius_and_barycenter:
        returns radius, coordinates of barycenter
        for circumscribed hypersphere for these points

    @param nodes (np.array): array of dimension (p+1) x p of the p+1 points in
    p dimension

    This function is based on Stack Exchange:
    https://math.stackexchange.com/questions/1087011/calculating-the-radius-of-the-circumscribed-sphere-of-an-arbitrary-tetrahedron # noqa
    """
    size = nodes.shape[1] + 2
    delta = np.ones((size, size))
    delta[0, 0] = 0
    delta[1:, 1:] = cdist(nodes, nodes, "sqeuclidean")
    inverse = np.linalg.inv(delta)
    radius = np.sqrt(np.abs(inverse[0, 0] / 2))
    barycenter = inverse[1:, 0] @ nodes
    return radius, barycenter


def inside_sphere(nodes, barycenter, radius):
    """
    inside_ball:
        find if any of the nodes is inside the given sphere

    @param nodes (np.array): points to check if inside
    @param barycenter (np.array): coordinates of the barycenter
    @param radius (float): radius
    """
    dist_2 = np.diag((nodes - barycenter) @ np.transpose(nodes - barycenter))
    return np.any(np.array([item < radius**2 for item in dist_2]))


def Tzero(w, tol=1e-5):
    """
    Tzero:
        set values under threshold to zero.

    @param w (np.array): numpy array of dimension 1, such that sum(w) = 1.
    @param tol (float): tolerance
    """
    w[w < tol] = 0
    return w/np.sum(w)


# ------------------------------------------------------------------------------
# MAIN FUNCTIONS
# ------------------------------------------------------------------------------
def incremental_pure_synth(X1, X0):
    """
    incremental_pure_synth:
        main algorithm, find the vertices of the simplex that X1 falls into
        returns the points and the antiranks

    @param X1 (np.array): array of dimension p of the treated unit
    @param X0 (np.array): n x p array of untreated units

    """
    # get the ranks and anti-ranks of X0 with respect to their distances to X1
    ranks, anti_ranks = get_ranks(X1, X0)
    n0, p = X0.shape

    # Initialize variables
    foundIt, inHullFlag = False, False

    # BIG LOOP #
    # We loop over number of possible points, starting from p+1
    for k in range(p + 1, n0 + 1):
        # init the_simplex variable if there is a problem
        # when points is not inside convex hull, returns all the points
        the_simplex = tuple(range(k))

        # 1. Set of 'k' nearest neighbors
        X_NN = X0[anti_ranks[:k], ]

        # 2. Check if X1 belongs to that convex hull
        if not inHullFlag:  # this if clause should be removed
            inHullFlag = in_hull(X1, X_NN)

        if not inHullFlag:
            continue  # skip to next iteration if X1 not in convex hull of nearest neighbors

        # 3. For all the subsets of cardinality p+1 that have x in their convex hull...
        #  (since previous simplices did not contain X1,
        #   we need only to consider the simplices that have the new nearest neighbors as a vertex)
        # ...check if a point in X0 is contained in the circumscribing hypersphere of any of these simplices
        for item in itertools.combinations(range(k-1), p):
            candidate = item + (k-1,)
            if in_hull(X1, X_NN[candidate, ]):
                try:
                    radius, center = compute_radius_and_barycenter(X_NN[candidate, ])  # sometimes gives an error if points have the same values for a particular X0
                except:
                    radius = np.nan

                if np.isnan(radius):  # if there is a degenerate case, we stop
                    the_simplex = candidate
                    foundIt = True
                    break

                if not inside_sphere(np.delete(X0, anti_ranks[candidate, ], 0), center, radius):
                    the_simplex = candidate
                    foundIt = True
                    break
        if foundIt:
            break

    anti_ranks_tilde = sorted(anti_ranks[the_simplex, ])
    return X0[anti_ranks_tilde, ], anti_ranks_tilde


def pensynth_weights(X0, X1, pen=0.0, V=None):
    """
    pensynth_weights:
        computes penalized synthetic control weights with penalty pen

    See "A Penalized Synthetic Control Estimator for Disaggregated Data"

    @param X0 (np.array): n x p matrix of untreated units
    @param X1 (np.array): 1 x p matrix of the treated unit
    @param pen (float): lambda, positive tuning parameter
    @param V (np.array): weights for the norm
    """
    if V is None:
        V = np.identity(X0.shape[1])
    n0 = len(X0)

    # OBJECTIVE
    delta = np.diag((X0-X1) @ V @ np.transpose(X0-X1))
    P = matrix(X0 @ V @ np.transpose(X0))
    q = matrix(-X0 @ V @ X1 + (pen/2)*delta)

    # ADDING-UP TO ONE
    A = matrix(1.0, (1, n0))
    b = matrix(1.0)

    # NON-NEGATIVITY
    G = matrix(-np.identity(n0))
    h = matrix(np.zeros(n0))

    # COMPUTE SOLUTION
    solvers.options['show_progress'] = False
    solvers.options['abstol'] = 1e-8
    solvers.options['reltol'] = 1e-8
    solvers.options['maxiters'] = 500
    sol = solvers.qp(P, q, G, h, A, b)
    return Tzero(np.squeeze(np.array(sol['x'])))


if __name__ == '__main__':
    # Test with simulated data
    n = 11
    p = 5

    X = np.random.normal(0, 1, size=(n, p))
    X1 = X[0]
    X0 = np.delete(X, (0), axis=0)

    in_hull_flag = in_hull(X1, X0)
    if in_hull_flag:
        print("Treated is inside convex hull.")
    else:
        print("Treated not in convex hull.")

    print("="*80)
    print("Method 1 : Compute Delaunay Triangulation of X0")
    print("="*80)

    start_time = time.time()
    tri = Delaunay(X0)
    any_simplex = tri.find_simplex(X1)
    print(any_simplex >= 0)
    the_simplex_Delaunay = tri.simplices[any_simplex]
    print(X0[sorted(the_simplex_Delaunay), ])
    print(f"Temps d'exécution total : {(time.time() - start_time):.7f} secondes ---")

    print("="*80)
    print("Method 2 : incremental algorithm")
    print("="*80)

    start_time = time.time()
    simplex, _ = incremental_pure_synth(X1=X1, X0=X0)
    print(simplex)
    print(f"Temps d'exécution total : {(time.time() - start_time):.7f} secondes ---")
