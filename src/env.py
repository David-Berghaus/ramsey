import os
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.optimize as opt
import networkx as nx
from stable_baselines3.common.logger import Logger
import time
import networkx as nx
from networkx.readwrite.graph6 import write_graph6

from score import get_chromatic_number, get_score, maximum_clique_size, get_embed_dim
    
class CustomGraphGANEnv(gym.Env):
    """Custom Environment for a graph."""
    def __init__(self, n, dir="", model_id=0, max_embed_dim_threshold=4, min_chromatic_number_threshold=6, logger=None):
        super(CustomGraphGANEnv, self).__init__()
        # Define action and observation space
        self.num_entries = n*(n-1)//2
        self.action_space = spaces.MultiBinary(n=self.num_entries)
        self.n = n
        self.observation_space = spaces.Box(low=-4, high=4, shape=(self.n,), dtype=np.float32)
        self.observation_space_np = get_observation_space(self.n)
        self.dir = dir
        self.max_embed_dim_threshold = max_embed_dim_threshold
        self.min_chromatic_number_threshold = min_chromatic_number_threshold
        self.model_id = model_id
        self.graph_storage_file = os.path.join(self.dir, f"graphs_{self.model_id}.g6")
        self.logger = logger
        self.iteration_count = 0
        self.time_spent_computing_score = 0
        self.encountered_graphs = set()
        self.found_unique_graphs = 0

    def step(self, action):
        self.iteration_count += 1
        self.observation_space_np = get_observation_space(self.n)
        G, A = obs_space_to_graph(action, self.n, return_adjacency_matrix=True, use_GAN=True)
        # Transform A to boolean matrix
        A = A > 0
        if not nx.is_connected(G):
            return self.observation_space_np, -10, True, False, {}
        start_time = time.time()
        chromatic_number = get_chromatic_number(G)
        max_clique_size = maximum_clique_size(G)
        # embed_dim = get_embed_dim(A, 5)
        embed_dim = max_clique_size
        num_edges = G.number_of_edges()

        reward = get_score(chromatic_number, embed_dim, max_clique_size, num_edges, chromatic_weighting_fact=1.0, clique_size_weighting_fact=1.0, num_edges_weighting_fact=1.0/(4*self.n))
        self.time_spent_computing_score += time.time() - start_time

        info = {}
        graph6_str = nx.to_graph6_bytes(G).decode('utf-8')
        if graph6_str not in self.encountered_graphs:
            if chromatic_number >= self.min_chromatic_number_threshold and embed_dim <= self.max_embed_dim_threshold:
                self.encountered_graphs.add(graph6_str)
                # Store the graph by appending graph_storage_file
                with open(self.graph_storage_file, "a") as f:
                    f.write(graph6_str + "\n")
                    # f.write(str(action) + "\n")
                # The graphs can now be loaded via graphs = nx.read_graph6('graphs_0.g6')
                self.found_unique_graphs += 1
                reward += 10
                if embed_dim <= 3:
                    reward += 10
        else:
            reward -= 10
        info["chromatic_number"] = chromatic_number
        info["embed_dim"] = embed_dim

        # Log chromatic number and max clique size
        if self.logger is not None:
            self.logger.record("env/chromatic_number", chromatic_number)
            self.logger.record("env/embed_dim", embed_dim)
            self.logger.record("env/max_clique_size", max_clique_size)
            self.logger.record("env/time_spent_computing_score", self.time_spent_computing_score)
            self.logger.record("env/num_edges", num_edges)
            self.logger.record("env/found_unique_graphs", self.found_unique_graphs)
            self.logger.dump(self.iteration_count)

        return self.observation_space_np, reward, True, False, info
        
    def reset(self, **kwargs):
        return self.observation_space_np, {}
    
def get_observation_space(num_entries, low=-4, high=4):
    res = np.random.randn(num_entries)
    return np.clip(res, low, high)

def flattened_off_diagonal_to_adjacency_matrix(flattened_off_diagonal: npt.ArrayLike, n: int) -> npt.ArrayLike:
    """Converts a flattened off-diagonal to an adjacency matrix."""
    adjacency_matrix = np.zeros((n, n))
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            adjacency_matrix[i, j] = flattened_off_diagonal[count]
            adjacency_matrix[j, i] = flattened_off_diagonal[count]
            count += 1
    return adjacency_matrix

def obs_space_to_graph(obs_space, n, use_GAN=False, return_adjacency_matrix=False):
    if use_GAN:
        adj_matrix = flattened_off_diagonal_to_adjacency_matrix(obs_space, n)
    else:
        adj_matrix = flattened_off_diagonal_to_adjacency_matrix(obs_space[n*(n-1)//2:], n)
    G = nx.from_numpy_array(adj_matrix)
    if return_adjacency_matrix:
        return G, adj_matrix
    return G


