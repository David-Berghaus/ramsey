import numpy as np
from scipy.special import comb
import networkx as nx
from networkx.algorithms.operators.unary import complement
from networkx.algorithms.components.connected import is_connected


def get_cliques_and_count(G_nx, k): #See: https://stackoverflow.com/a/58782120
    count = 0
    cliques = []
    for clique in nx.find_cliques(G_nx):
        cliques.append(clique)
        if len(clique) == k:
            count += 1
        elif len(clique) > k:
            count += comb(len(clique), k, exact=True)
    return cliques, count

