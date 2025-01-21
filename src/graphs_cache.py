import numpy as np
import hashlib
from networkx.linalg.graphmatrix import adjacency_matrix
import networkx as nx

from score import get_score_and_cliques
from env import obs_space_to_graph

def hash_tensor(tensor):
    # Ensure the tensor is on the CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    # Convert the tensor to bytes directly
    tensor_bytes = tensor.detach().numpy().tobytes()
    # Create a hash object
    hash_object = hashlib.sha256()  # Use a faster hashing algorithm if needed
    hash_object.update(tensor_bytes)
    return hash_object.hexdigest()

class GraphsCache:
    def __init__(self, max_size):
        self.cache = {}
        self.order = []
        self.max_size = max_size

    def get(self, observation_space, r, b, n):
        key = hash_tensor(observation_space)
        if key in self.cache:
            return self.cache[key]
        G = obs_space_to_graph(observation_space, n)
        _, cliques_r, cliques_b, is_connected = get_score_and_cliques(G, r, b, -1000)
        adj_matrix = adjacency_matrix(G).toarray()
        self.put(key, (cliques_r, cliques_b, adj_matrix))
        return cliques_r, cliques_b, adj_matrix

    def put(self, key, value):
        if key not in self.cache:
            if len(self.order) >= self.max_size:
                # Remove the oldest item
                oldest_key = self.order.pop(0)
                del self.cache[oldest_key]
        else:
            # If key already exists, remove it from order
            self.order.remove(key)

        # Add the new key-value pair
        self.cache[key] = value
        self.order.append(key)