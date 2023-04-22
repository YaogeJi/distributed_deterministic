import numpy as np
from networkx import *
from networkx.convert_matrix import to_numpy_array
from utils import MaxIterError
import copy

def metropolis_hasting(G):
    for i in G.edges:
        degree = max(G.degree(i[0]), G.degree(i[1]))
        G.add_edge(*i, weight=1/(degree + 1))
    adjacent_matrix = to_numpy_array(G)
    weighted_matrix = np.eye(adjacent_matrix.shape[0]) - np.diag(sum(adjacent_matrix)) + adjacent_matrix
    return weighted_matrix
    

def lazy_metropolis(G):
    for i in G.edges:
        degree = max(G.degree(i[0]), G.degree(i[1]))
        G.add_edge(*i, weight=1/degree)
    adjacent_matrix = to_numpy_array(G)
    weighted_matrix = np.eye(adjacent_matrix.shape[0]) - np.diag(sum(adjacent_matrix)) + adjacent_matrix
    weighted_matrix = 0.5 * weighted_matrix + 0.5 * np.eye(adjacent_matrix.shape[0])
    return weighted_matrix
    


class ExistedNetwork:
    def __init__(self, network_name, m, mixing, net_arg=None, rho=0, eps=1e-3, max_attempts=int(1e6), seed=0):
        self.network_name = network_name
        self.m = m
        self.net_arg = net_arg
        self.rho = rho
        self.mixing = mixing
        self.eps = eps
        self.max_attempts = max_attempts
        self.seed = seed
        self._generate()

    def __call__(self):
        return self.G
    
    @property
    def w(self):
        return eval(self.mixing)(self.G)
    
    @property
    def connectivity(self):
        eigenvalue, _ = np.linalg.eig(self.w)
        sorted_eigenvalue = np.sort(np.abs(eigenvalue))
        return sorted_eigenvalue[-2]

    def _generate(self): 
        if self.network_name == 'complete_graph':
            self.G = eval(self.network_name)(self.m)
        elif self.network_name == 'star_graph':
            self.G = eval(self.network_name)(self.m - 1)
        elif self.network_name == 'path_graph':
            self.G = eval(self.network_name)(self.m)
        elif self.network_name in ['fast_gnp_random_graph', 'random_geometric_graph']:
            for i in range(self.max_attempts):
                self.G = eval(self.network_name)(self.m, self.net_arg, self.seed+i)
                if np.abs(self.connectivity - self.rho) < self.eps:
                    break
            else:
                raise MaxIterError("Not able to generate graph with given connectivity.")
            


