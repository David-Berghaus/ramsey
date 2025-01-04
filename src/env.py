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
    def __init__(self, n, r, b, not_connected_punishment, dir="", model_id=0, logger=None):
        super(AdjacencyMatrixFlippingEnv, self).__init__()
        
        self.reward_factor = None # Factor with which we normalize the rewards
        self.max_steps = 10 # Maximum number of steps per episode TODO: Check if this is a good value
        self.not_connected_punishment = not_connected_punishment # Punishment for not connected graphs TODO: Check if this is a good value
        
        # Define action and observation space
        self.num_entries = n*(n-1)//2
        self.action_space = spaces.Discrete(self.num_entries)
        self.observation_space = spaces.MultiBinary(self.num_entries)
        self.n = n
        self.r = r
        self.b = b
        self.observation_space_np = np.random.randint(2, size=self.num_entries)
        self.best_observation_space = np.copy(self.observation_space_np)    
        self.best_recorded_score, _, _, _ = get_score_and_cliques(obs_space_to_graph(self.observation_space_np, self.n), self.r, self.b, self.not_connected_punishment)
        
        self.dir = dir
        self.model_id = model_id
        self.graph_storage_file = os.path.join(self.dir, f"graphs_{self.model_id}.g6")
        self.logger = logger
        self.iteration_count = 0 # Total iteration count
        self.step_count = 0 # Iteration count per episode
        

    def step(self, action):
        # Flip the selected bit in the observation space
        self.observation_space_np[action] = 1 - self.observation_space_np[action]
        G = obs_space_to_graph(self.observation_space_np, self.n)
        
        # Compute the new score after the agent's action
        score, cliques_r, cliques_b, is_connected = get_score_and_cliques(G, self.r, self.b, self.not_connected_punishment)
        
        if not is_connected:
            # Apply a strong negative reward if the graph is disconnected
            reward = self.not_connected_punishment
            done = True  # End the episode
            return self.observation_space_np, reward, done, False, {}
        
        # Compute reward based on the change in score
        if hasattr(self, 'previous_score'):
            reward = score - self.previous_score
        else:
            reward = 0  # No reward for the first action
        
        # Update the previous score for the next step
        self.previous_score = score
        
        # Check if the episode should end
        self.iteration_count += 1
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        # Update the best recorded score and state
        if score > self.best_recorded_score:
            self.best_recorded_score = score
            self.best_observation_space = np.copy(self.observation_space_np)
    
        info = {}
    
        # Log the current score if a logger is available
        if self.logger is not None:
            self.logger.record("env/score", score)
            self.logger.record("env/reward", reward)
            self.logger.record("env/best_score", self.best_recorded_score)
            self.logger.dump(self.iteration_count)
        
        return self.observation_space_np, reward, done, False, info
        
    def reset(self, **kwargs):
        self.observation_space_np = np.copy(self.best_observation_space)
        self.step_count = 0
        
        # Recalculate the score for the reset state
        G = obs_space_to_graph(self.observation_space_np, self.n)
        self.previous_score = self.best_recorded_score
        
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


