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
from dataclasses import dataclass


@dataclass
class Hypersphere():
    nodes: np.ndarray

    @property
    def nodes(self) -> np.ndarray:
        return self._nodes

    @nodes.setter
    def nodes(self, value: np.ndarray):
        msg = r"nodes must be a matrix of (p+1) x p"
        nodes = np.array(value, dtype=float)
        if nodes.ndim != 2:
            raise ValueError(msg)
        rows, columns = nodes.shape
        if (rows - 1) != columns:
            raise ValueError(msg)
        self._nodes = nodes

    @property
    def rows(self) -> int:
        return self.nodes.shape[0]

    @property
    def columns(self) -> int:
        return self.nodes.shape[1]

    @property
    def distance(self) -> np.ndarray:
        return cdist(self.nodes, self.nodes, "sqeuclidean")

    @property
    def inv_cm_determinant(self) -> np.ndarray:
        """
        The inverse of the Cayley-Menger Determinant. This function is based
        on Stack Exchange:
        https://math.stackexchange.com/questions/1087011/calculating-the-radius-of-the-circumscribed-sphere-of-an-arbitrary-tetrahedron # noqa
        """
        size = self.columns + 2
        determinant = np.ones((size, size))
        determinant[0, 0] = 0
        determinant[1:, 1:] = self.distance
        inverse = np.linalg.inv(determinant)
        return inverse
    
    @property
    def radius(self) -> float:
        return np.sqrt(np.abs(self.inv_cm_determinant[0, 0] / 2))
    
    @property
    def barycenter(self) -> np.ndarray:
        return self.inv_cm_determinant[1:, 0] @ self.nodes
    
    def contains(self, nodes: np.ndarray) -> bool:
        """
        Find if any of the nodes is inside the given sphere.
    
        @param nodes (np.array): points to check if inside
        """
        if nodes.shape[1] != self.columns:
            raise ValueError(f"nodes must be matrix of n x {self.columns}")
        difference = nodes - self.barycenter
        distance = (difference ** 2).sum(axis=1)
        return np.any(distance < (self.radius ** 2))


def get_ranks(node: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray]:
    """
    Returns the ranks and anti-ranks of nodes by rank in closeness to node.

    @param node (np.array): point for which we want to find the neighbors
    @param nodes (np.array): points that are candidate neighbors
    """
    distance = cdist(node.reshape(1, -1), nodes, "sqeuclidean")[0]
    anti_ranks = np.argsort(distance)
    ranks = np.argsort(anti_ranks)
    return ranks, anti_ranks


def in_hull(x: np.ndarray, points: np.ndarray) -> bool:
    """
    Tests if x is in hull formed by points.

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


def set_zero_weights(weights: np.ndarray, tolerance: int=1e-5) -> np.ndarray:
    """
    Set weights within the tolerance to zero and ensure the weights sum to one.

    @param w (np.array): numpy array of dimension 1, such that sum(w) = 1.
    @param tol (float): tolerance
    """
    matrix[matrix <= tolerance] = 0
    return matrix / np.sum(matrix)


def incremental_pure_synth(X1, X0):
    """
    Main algorithm. Find the vertices of the simplex that X1 falls into returns
    the points and the antiranks.

    @param X1 (np.array): array of dimension p of the treated unit
    @param X0 (np.array): n x p array of untreated units

    """
    # we don't even use the ranks
    ranks, anti_ranks = get_ranks(X1, X0)
    n0, p = X0.shape
    simplex = None

    for k in range(p + 1, n0 + 1):
        if simplex is not None:
            break
        k_nearest_neighbors = X0[anti_ranks[:k], ]
        if in_hull(X1, k_nearest_neighbors):
            # 3. For all the subsets of cardinality p+1 that have x in their
            # convex hull... (since previous simplices did not contain X1, we
            # need only to consider the simplices that have the new nearest
            # neighbors as a vertex) ...check if a point in X0 is contained in
            # the circumscribing hypersphere of any of these simplices
            for subset in itertools.combinations(range(k - 1), p):
                candidate = subset + (k - 1, )
                hypersphere = Hypersphere(k_nearest_neighbors[candidate, ])
                if in_hull(X1, hypersphere.nodes):
                    irrelevant_nodes = np.delete(X0, anti_ranks[candidate, ], axis=0)
                    if not hypersphere.contains(irrelevant_nodes):
                        simplex = candidate

    if simplex is None:
        simplex = tuple(range(k))
    anti_ranks_tilde = sorted(anti_ranks[simplex, ])
    result = (X0[anti_ranks_tilde, ], anti_ranks_tilde)
    return result


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
    return set_zero_weights(np.squeeze(np.array(sol['x'])))


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
