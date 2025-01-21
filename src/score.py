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

def get_score_and_cliques(G_nx, r, b, not_connected_punishment):
    G_complement = complement(G_nx)
    graphs_are_connected = True
    if not is_connected(G_nx):
        graphs_are_connected = False
        G_nx = connect_graph(G_nx) # Artificially connect the graph to ensure stability. But we will punish the network later
    if not is_connected(G_complement):
        graphs_are_connected = False
        G_complement = connect_graph(G_complement) # Artificially connect the graph to ensure stability. But we will punish the network later
    cliques_r, count_r = get_cliques_and_count(G_nx, r)
    cliques_b, count_b = get_cliques_and_count(G_complement, b)
    score = -(count_r + count_b) if graphs_are_connected else not_connected_punishment
    return score, cliques_r, cliques_b, graphs_are_connected

def connect_graph(G):
    # Get the connected components of the graph
    components = list(nx.connected_components(G))
    
    # Iterate through components and connect them
    for i in range(len(components) - 1):
        # Take a node from the current component and the next component
        node_from_current = next(iter(components[i]))
        node_from_next = next(iter(components[i + 1]))
        
        # Add an edge between these nodes
        G.add_edge(node_from_current, node_from_next)
    
    return G