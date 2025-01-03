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

from score import get_score_and_cliques

class AdjacencyMatrixFlippingEnv(gym.Env):
    """Custom Environment for flipping entries in the adjacency matrix. The agent receives a binary vector and suggests a bit to flip."""
    def __init__(self, n, dir="", model_id=0, r=3, b=3, logger=None):
        super(AdjacencyMatrixFlippingEnv, self).__init__()
        # Define action and observation space
        self.num_entries = n*(n-1)//2
        self.action_space = spaces.Discrete(self.num_entries)
        self.observation_space = spaces.MultiBinary(self.num_entries)
        self.n = n
        self.r = r
        self.b = b
        self.observation_space_np = np.random.randint(2, size=self.num_entries)
        self.best_observation_space = np.copy(self.observation_space_np)    
        self.best_recorded_score = -float("inf")
        
        self.reward_factor = None # Factor with which we normalize the rewards
        self.max_steps = 10 # Maximum number of steps per episode TODO: Check if this is a good value
        self.not_connected_punishment = -100 # Punishment for not connected graphs TODO: Check if this is a good value
        
        self.dir = dir
        self.model_id = model_id
        self.graph_storage_file = os.path.join(self.dir, f"graphs_{self.model_id}.g6")
        self.logger = logger
        self.iteration_count = 0
        

    def step(self, action):
        self.observation_space_np[action] = 1-self.observation_space_np[action]
        G = obs_space_to_graph(self.observation_space_np, self.n)
        score, cliques_r, cliques_b, is_connected = get_score_and_cliques(G, self.r, self.b, self.not_connected_punishment)
        if not is_connected:
            return self.observation_space_np, self.not_connected_punishment, True, False, {}
        
        done = self.step_count > self.max_steps
        if self.reward_factor is None:
            self.reward_factor = int(score/10.0) #To Do: check if this is a good reward factor
        reward = self.reward_factor/(self.reward_factor+score)
        if not done:
            #reward /= self.max_steps
            reward = 0 #It seems to be a bit better not to give any reward for intermediate steps, however we still compute it to search for the best state     
        if score >= self.best_recorded_score:
            self.best_observation_space = np.copy(self.observation_space_np) #Update state even if score is equal to avoid getting stuck in local minima
        
        self.iteration_count += 1
        info = {}
        # graph6_str = nx.to_graph6_bytes(G).decode('utf-8')
        # if graph6_str not in self.encountered_graphs:
        #     if chromatic_number >= self.min_chromatic_number_threshold and embed_dim <= self.max_embed_dim_threshold:
        #         self.encountered_graphs.add(graph6_str)
        #         # Store the graph by appending graph_storage_file
        #         with open(self.graph_storage_file, "a") as f:
        #             f.write(graph6_str + "\n")
        #             # f.write(str(action) + "\n")
        #         # The graphs can now be loaded via graphs = nx.read_graph6('graphs_0.g6')
            

        # # Log chromatic number and max clique size
        # if self.logger is not None:
        #     self.logger.record("env/chromatic_number", chromatic_number)
        #     self.logger.dump(self.iteration_count)

        return self.observation_space_np, reward, True, False, info
        
    def reset(self, **kwargs):
        self.observation_space_np = np.copy(self.best_observation_space)
        self.step_count = 0
        return self.observation_space_np, {}
    
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

def obs_space_to_graph(obs_space, n, return_adjacency_matrix=False):
    adj_matrix = flattened_off_diagonal_to_adjacency_matrix(obs_space, n)
    G = nx.from_numpy_array(adj_matrix)
    if return_adjacency_matrix:
        return G, adj_matrix
    return G


