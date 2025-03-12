import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np

from clique_prediction_models import RamseyGraphGNNWithCliqueAttention
from env import obs_space_to_graph
from graphs_cache import GraphsCache, hash_tensor
from score import get_score_and_cliques

def get_one_hot_encoding(n, i):
    """Returns the one-hot encoding of the integer i."""
    encoding = np.zeros(n)
    encoding[i] = 1
    return encoding
       

class RamseyGraphGNNEdgeScorer(RamseyGraphGNNWithCliqueAttention):
    def __init__(self, n_vertices=17, r=4, b=4, hidden_dim=64, num_layers=3, 
                 clique_attention_context_len=16, node_attention_context_len=8, num_heads=2):
        super(RamseyGraphGNNEdgeScorer, self).__init__(
            n_vertices=n_vertices, r=r, b=b, hidden_dim=hidden_dim,
            num_layers=num_layers, clique_attention_context_len=clique_attention_context_len,
            node_attention_context_len=node_attention_context_len, num_heads=num_heads
        )
        
        # Add edge scoring projections
        self.edge_query = nn.Linear(hidden_dim, hidden_dim)
        self.edge_key = nn.Linear(hidden_dim, hidden_dim)
    
    def score_edges(self, graph_embedding):
        """
        Score all possible edge flips using attention between node embeddings.
        
        Args:
            graph_embedding (torch.Tensor): Enhanced node embeddings [n, hidden_dim]
            
        Returns:
            torch.Tensor: Edge scores [n, n]
        """
        # Project node embeddings into query/key space
        node_queries = self.edge_query(graph_embedding)  # [n, hidden_dim]
        node_keys = self.edge_key(graph_embedding)       # [n, hidden_dim]
        
        # Compute all pairwise scores with a single matrix multiplication
        edge_scores = torch.matmul(node_queries, node_keys.transpose(-2, -1))  # [n, n]
        
        # Mask self-loops (diagonal)
        mask = torch.eye(self.n, device=edge_scores.device).bool()
        edge_scores = edge_scores.masked_fill(mask, -float('inf'))
        
        return edge_scores
    
    def forward_rl(self, x):
        """
        Forward pass for reinforcement learning to score potential edge flips.
        """
        # Build adjacency matrices
        adj_matrices = self.build_adjacency_matrix_batch(x)
        
        batch_size = x.shape[0]
        device = x.device
        
        edge_scores_batch = []
        for i in range(batch_size):
            # Get GNN node embeddings
            node_embeddings = self.extract_node_features(adj_matrices[i])
            
            # Get cliques
            graph_vec = x[i]
            
            # Use cache to avoid recomputation
            graph_key = hash_tensor(graph_vec.cpu().numpy())
            if graph_key in self.graphs_cache.cache:
                cliques_r, cliques_b, _ = self.graphs_cache.cache[graph_key]
            else:
                # Convert to NetworkX graph
                G = obs_space_to_graph(graph_vec.cpu().numpy(), self.n)
                # Get cliques
                _, cliques_r, cliques_b, _ = get_score_and_cliques(G, self.r, self.b, self.not_connected_punishment)
                # Cache the results
                self.graphs_cache.put(graph_key, (cliques_r, cliques_b, None))
            
            # Apply node-clique cross-attention
            enhanced_node_embeddings = self.node_clique_cross_attention(node_embeddings, cliques_r + cliques_b)
            
            # Process cliques using node embeddings
            r_clique_embed = self.process_cliques(
                cliques_r, enhanced_node_embeddings, self.r, 
                self.r_query, self.r_key, self.r_value, 
                self.r_downward, device
            )
            
            b_clique_embed = self.process_cliques(
                cliques_b, enhanced_node_embeddings, self.b,
                self.b_query, self.b_key, self.b_value, 
                self.b_downward, device
            )
            
            # Keep full node embeddings (no mean pooling)
            graph_embedding = enhanced_node_embeddings + r_clique_embed + b_clique_embed
            
            # Score all potential edges using these node embeddings
            edge_scores = self.score_edges(graph_embedding)
            
            # Extract upper triangular part (excluding diagonal) for action space
            triu_indices = torch.triu_indices(self.n, self.n, offset=1, device=device)
            edge_scores_flat = edge_scores[triu_indices[0], triu_indices[1]]
            
            edge_scores_batch.append(edge_scores_flat)
        
        return torch.stack(edge_scores_batch)


# SB3 Feature Extractor Wrapper
class RamseyGNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=1, n_vertices=17, r=4, b=4, 
                 hidden_dim=64, num_layers=3, clique_attention_context_len=16, 
                 node_attention_context_len=8, num_heads=2):
        # The features_dim will be the number of possible edges n*(n-1)/2
        features_dim = n_vertices * (n_vertices - 1) // 2
        
        super().__init__(observation_space, features_dim)
        
        # Create the GNN model with edge scoring
        self.gnn_model = RamseyGraphGNNEdgeScorer(
            n_vertices=n_vertices, r=r, b=b, hidden_dim=hidden_dim,
            num_layers=num_layers, clique_attention_context_len=clique_attention_context_len,
            node_attention_context_len=node_attention_context_len, num_heads=num_heads
        )
    
    def forward(self, observations):
        # Call the RL-specific forward method
        return self.gnn_model.forward_rl(observations)
    