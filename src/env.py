import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from networkx.readwrite.graph6 import write_graph6

from score import get_score_and_cliques

class AdjacencyMatrixFlippingEnv(gym.Env):
    """Custom Environment for flipping entries in the adjacency matrix."""

    def __init__(self, n, r, b, not_connected_punishment, num_local_searches_before_reset, dir="", model_id=0, logger=None, env_id=None):
        super(AdjacencyMatrixFlippingEnv, self).__init__()
        
        self.reward_factor = None
        self.max_steps = n * (n - 1) // 2
        self.not_connected_punishment = not_connected_punishment
        self.num_local_searches_before_reset = num_local_searches_before_reset
        self.num_local_searches = 0

        # Define action and observation space
        self.num_entries = n * (n - 1) // 2
        self.action_space = spaces.Discrete(self.num_entries)
        self.observation_space = spaces.MultiBinary(self.num_entries)
        self.n = n
        self.r = r
        self.b = b
        self.observation_space_np = np.random.randint(2, size=self.num_entries)
        self.best_observation_space = np.copy(self.observation_space_np)    
        self.best_recorded_score, _, _, _ = get_score_and_cliques(
            obs_space_to_graph(self.observation_space_np, self.n), self.r, self.b, self.not_connected_punishment
        )
        self.previous_score = self.best_recorded_score
        
        self.dir = dir
        self.model_id = model_id
        self.graph_storage_file = os.path.join(self.dir, f"graphs_{self.model_id}.g6")
        self.logger = logger
        self.env_id = env_id
        self.iteration_count = 0
        self.step_count = 0

    def step(self, action):
        done = self.step_count >= self.max_steps
        # Flip the selected bit in the observation space
        self.observation_space_np[action] = 1 - self.observation_space_np[action]
        G = obs_space_to_graph(self.observation_space_np, self.n)
        
        # Compute the new score after the agent's action
        score, cliques_r, cliques_b, is_connected = get_score_and_cliques(G, self.r, self.b, self.not_connected_punishment)
        
        if not is_connected:
            # Apply a strong negative reward if the graph is disconnected
            reward = self.not_connected_punishment
            done = True
        else:
            if not done:        
                reward = score - self.previous_score
            else:
                reward = score
            
            # Update the previous score for the next step
            self.previous_score = score
            
            # Check if the episode should end
            self.iteration_count += 1
            self.step_count += 1

            
            # Update the best recorded score and state
            if score > self.best_recorded_score:
                self.best_recorded_score = score
                self.best_observation_space = np.copy(self.observation_space_np)
    
        info = {
            'score': score,
            'reward': reward,
            'best_score': self.best_recorded_score
        }
    
        return self.observation_space_np, reward, done, False, info
      
    def reset(self, **kwargs):
        # Reset the environment to a random one
        self.observation_space_np = np.random.randint(2, size=self.num_entries)
        # Recalculate the score for the reset state
        G = obs_space_to_graph(self.observation_space_np, self.n)
        self.previous_score, _, _, _ = get_score_and_cliques(
            G, self.r, self.b, self.not_connected_punishment
        )
            
        self.step_count = 0
    
        return self.observation_space_np, {}
        
    # def reset(self, **kwargs):
    #     if self.num_local_searches > self.num_local_searches_before_reset:
    #         # Reset the environment to a random one
    #         self.observation_space_np = np.random.randint(2, size=self.num_entries)
    #         self.num_local_searches = 0
    #         # Recalculate the score for the reset state
    #         G = obs_space_to_graph(self.observation_space_np, self.n)
    #         self.best_recorded_score, _, _, _ = get_score_and_cliques(
    #             G, self.r, self.b, self.not_connected_punishment
    #         )
    #         self.previous_score = self.best_recorded_score
    #     else:
    #         self.observation_space_np = np.copy(self.best_observation_space)
    #         self.previous_score = self.best_recorded_score
            
    #     self.step_count = 0
    #     self.num_local_searches += 1
    
    #     return self.observation_space_np, {}

def flattened_off_diagonal_to_adjacency_matrix(flattened_off_diagonal: np.ndarray, n: int) -> np.ndarray:
    """Converts a flattened off-diagonal to an adjacency matrix."""
    adjacency_matrix = np.zeros((n, n))
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
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