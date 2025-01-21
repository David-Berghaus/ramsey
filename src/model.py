import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np

from graphs_cache import GraphsCache

def get_one_hot_encoding(n, i):
    """Returns the one-hot encoding of the integer i."""
    encoding = np.zeros(n)
    encoding[i] = 1
    return encoding
       

class NodeMeanPoolCliqueAttentionFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n, r, b, not_connected_punishment, features_dim, num_heads, node_attention_context_len, clique_attention_context_len):
        super(NodeMeanPoolCliqueAttentionFeatureExtractor, self).__init__(observation_space, features_dim)
        self.n = n
        self.r = r
        self.b = b
        self.not_connected_punishment = not_connected_punishment
        self.graphs_cache = GraphsCache(10000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # Some hyperparameters
        embed_dim = features_dim//2
        num_heads = 2
        self.node_attention_context_len = node_attention_context_len
        self.clique_attention_context_len = clique_attention_context_len
        
        self.node_embedder_1 = nn.Linear(n, embed_dim-1)
        self.clique_size_embedder_1 = nn.Linear(1, 1)
        
        self.clique_Wq_1 = nn.Linear(embed_dim, embed_dim)
        self.clique_Wk_1 = nn.Linear(embed_dim, embed_dim)
        self.clique_Wv_1 = nn.Linear(embed_dim, embed_dim)
        self.clique_attention_1 = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.clique_layer_norm_11 = torch.nn.LayerNorm(embed_dim)
        self.clique_nn_1 = nn.Sequential( # NN after attention
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.clique_layer_norm_12 = torch.nn.LayerNorm(embed_dim)
        self.clique_downward_projection_1 = nn.Linear(self.clique_attention_context_len*embed_dim, embed_dim)
        
        self.graph_conv_downward_projection_1 = nn.Linear(n*(embed_dim-1), embed_dim)
        self.graph_embedding_layer_norm_1 = torch.nn.LayerNorm(embed_dim)
        
        if r == b: # We can reuse the same parameters
            self.node_embedder_2 = self.node_embedder_1
            self.clique_size_embedder_2 = self.clique_size_embedder_1
            
            self.clique_Wq_2 = self.clique_Wq_1
            self.clique_Wk_2 = self.clique_Wk_1
            self.clique_Wv_2 = self.clique_Wv_1
            self.clique_attention_2 = self.clique_attention_1
            
            self.clique_layer_norm_21 = self.clique_layer_norm_11
            self.clique_nn_2 = self.clique_nn_1
            self.clique_layer_norm_22 = self.clique_layer_norm_12
            self.clique_downward_projection_2 = self.clique_downward_projection_1
            
            self.graph_conv_downward_projection_2 = self.graph_conv_downward_projection_1
            self.graph_embedding_layer_norm_2 = self.graph_embedding_layer_norm_1
        else:
            self.node_embedder_2 = nn.Linear(n, embed_dim-1)
            self.clique_size_embedder_2 = nn.Linear(1, 1)
            
            self.clique_Wq_2 = nn.Linear(embed_dim, embed_dim)
            self.clique_Wk_2 = nn.Linear(embed_dim, embed_dim)
            self.clique_Wv_2 = nn.Linear(embed_dim, embed_dim)
            self.clique_attention_2 = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.clique_layer_norm_21 = torch.nn.LayerNorm(embed_dim)
            self.clique_nn_2 = nn.Sequential( # NN after attention
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.clique_layer_norm_22 = torch.nn.LayerNorm(embed_dim)
            self.clique_downward_projection_2 = nn.Linear(self.clique_attention_context_len*embed_dim, embed_dim)
            
            self.graph_conv_downward_projection_2 = nn.Linear(n*(embed_dim-1), embed_dim)
            self.graph_embedding_layer_norm_2 = torch.nn.LayerNorm(embed_dim)
        
        
    def forward(self, observations):
        node_embeddings_1 = self.node_embeddings(True) #[n, embed_dim]
        node_embeddings_2 = self.node_embeddings(False) #[n, embed_dim]
        # print(node_embeddings_1)
        if torch.isnan(node_embeddings_1).any() or torch.isnan(node_embeddings_2).any():
            breakpoint()
        
        graph_embeddings = []
        cliques_r_batch, cliques_b_batch, adj_matrix_batch = [], [], []
        for observation in observations:
            cliques_r, cliques_b, adj_matrix = self.graphs_cache.get(observation, self.r, self.b, self.n)
            cliques_r_batch.append(cliques_r)
            cliques_b_batch.append(cliques_b)
            adj_matrix_batch.append(adj_matrix)
        adj_matrix_batch = torch.tensor(adj_matrix_batch, device=self.device, dtype=torch.float32)
        
        clique_embeddings_1, clique_masks_batch_1 = self.clique_mean_encoder(True, node_embeddings_1, cliques_r_batch) # [batch_size, clique_attention_context_len, embed_dim], [batch_size, clique_attention_context_len]    
        clique_embeddings_2, clique_masks_batch_2 = self.clique_mean_encoder(False, node_embeddings_2, cliques_b_batch) # [batch_size, clique_attention_context_len, embed_dim], [batch_size, clique_attention_context_len]
        if torch.isnan(clique_embeddings_1).any() or torch.isnan(clique_embeddings_2).any():
            breakpoint()
        
        cliques_embeddings_1 = self.cliques_attention_encoder(True, clique_embeddings_1, clique_masks_batch_1) # [batch_size, embed_dim]cliques_embeddings_1 = self.cliques_attention_encoder(False, clique_embeddings_2, clique_masks_batch_2) # [batch_size, embed_dim]
        cliques_embeddings_2 = self.cliques_attention_encoder(False, clique_embeddings_2, clique_masks_batch_2) # [batch_size, embed_dim]
        if torch.isnan(cliques_embeddings_1).any() or torch.isnan(cliques_embeddings_2).any():
            breakpoint()

        graph_edge_conv_embedding_1 = self.graph_edge_conv_encoder(True, node_embeddings_1, adj_matrix_batch) # [batch_size, embed_dim]
        graph_edge_conv_embedding_2 = self.graph_edge_conv_encoder(False, node_embeddings_2, adj_matrix_batch) # [batch_size, embed_dim]
        if torch.isnan(graph_edge_conv_embedding_1).any() or torch.isnan(graph_edge_conv_embedding_2).any():
            breakpoint()
        
        graph_embedding_1 = self.graph_embeddings(True, cliques_embeddings_1, graph_edge_conv_embedding_1) # [batch_size, embed_dim]
        graph_embedding_2 = self.graph_embeddings(False, cliques_embeddings_2, graph_edge_conv_embedding_2) # [batch_size, embed_dim]
        
        graph_embeddings = torch.cat([graph_embedding_1, graph_embedding_2], dim=1) # [batch_size, 2*embed_dim]
        
        # check if there are any NaN values in the graph embeddings
        if torch.isnan(graph_embeddings).any():
            breakpoint()
        return graph_embeddings
    
    
    def node_embeddings(self, use_r):
        nodes_one_hot = torch.zeros(self.n, self.n, device=self.device, dtype=torch.float32)
        for i in range(self.n):
            nodes_one_hot[i, i] = 1
        if use_r:
            return self.node_embedder_1(nodes_one_hot)
        else:
            return self.node_embedder_2(nodes_one_hot)
        
    
    def graph_embeddings(self, use_r, cliques_embeddings, graph_edge_conv_embedding):
        """
        Combine the embeddings of the cliques and the graph.
        TODO: See if a linear projection works better than simple addition.
        """
        graph_embeddings = cliques_embeddings + graph_edge_conv_embedding
        if use_r:
            return self.graph_embedding_layer_norm_1(graph_embeddings)
        else:
            return self.graph_embedding_layer_norm_2(graph_embeddings)
        
    
    def graph_edge_conv_encoder(self, use_r, node_embeddings, adj_matrix_batch):
        """
        Create embeddings of the graphs similar to a graph convolutional network.
        """
        if use_r:
            adj_tilde_batch = adj_matrix_batch + torch.eye(self.n, device=self.device, dtype=torch.float32)
        else:
            adj_tilde_batch = 1 - adj_matrix_batch
        degree_matrix_diag = torch.sum(adj_tilde_batch, dim=1)
        # Add a small epsilon for numerical stability
        degree_matrix_diag += 1e-8
        # Compute the reciprocal of the square root directly
        degree_matrix_batch = torch.diag_embed(1.0 / torch.sqrt(degree_matrix_diag))
        normalized_adj_matrix_batch = torch.matmul(degree_matrix_batch, adj_tilde_batch)
        normalized_adj_matrix_batch = torch.matmul(normalized_adj_matrix_batch, degree_matrix_batch)
        graph_embeddings = torch.matmul(normalized_adj_matrix_batch, node_embeddings)
        if use_r:
            graph_embeddings = self.graph_conv_downward_projection_1(graph_embeddings.flatten(1))
        else:
            graph_embeddings = self.graph_conv_downward_projection_2(graph_embeddings.flatten(1))
        return graph_embeddings
            
       
        
    def clique_mean_encoder_slow(self, use_r, node_embeddings, cliques_batch):
        """
        Compute the mean over the nodes in the cliques.
        
        Input:
        node_embeddings: [n, embed_dim-1]
        cliques_batch: List of graphs, where each graph is a list of cliques, where each clique is a list of nodes.
        
        Output:
        clique_embeddings_batch: [batch_size, clique_attention_context_len, embed_dim]
        clique_masks_batch: [batch_size, clique_attention_context_len]
        """
        clique_mask_batch = torch.zeros(len(cliques_batch), self.clique_attention_context_len, dtype=torch.bool, device=self.device)
        max_clique_size = max(len(clique) for cliques in cliques_batch for clique in cliques)
        if use_r:   
            clique_size_embeddings = self.clique_size_embedder_1(torch.tensor([[i] for i in range(max_clique_size+1)], device=self.device, dtype=torch.float32))
        else:
            clique_size_embeddings = self.clique_size_embedder_2(torch.tensor([[i] for i in range(max_clique_size+1)], device=self.device, dtype=torch.float32))
        embed_dim = node_embeddings.shape[-1]+1
        res = torch.zeros(len(cliques_batch), self.clique_attention_context_len, embed_dim, device=self.device, dtype=torch.float32) # [batch_size, clique_attention_context_len, embed_dim-1]
        for i in range(len(cliques_batch)):
            num_cliques = len(cliques_batch[i])
            for j in range(min(self.clique_attention_context_len,num_cliques)):
                clique = cliques_batch[i][j]
                clique_size = len(clique)
                res[i, j, :embed_dim-1] = torch.mean(node_embeddings[clique], dim=0)
                res[i, j, embed_dim-1] = clique_size_embeddings[clique_size]
            if num_cliques < self.clique_attention_context_len:
                clique_mask_batch[i, num_cliques:] = True
        
        return res, clique_mask_batch
    
    def clique_mean_encoder(self, use_r, node_embeddings, cliques_batch):
        """
        Compute the mean over the nodes in the cliques.
        
        Input:
        node_embeddings: [n, embed_dim-1]
        cliques_batch: List of graphs, where each graph is a list of cliques, 
                    where each clique is a list of nodes.
        
        Output:
        clique_embeddings_batch: [batch_size, clique_attention_context_len, embed_dim]
        clique_masks_batch: [batch_size, clique_attention_context_len]
        """
        batch_size = len(cliques_batch)
        context_len = self.clique_attention_context_len

        # Step 1: Extend node_embeddings with a padding node
        padding_embedding = torch.zeros(1, node_embeddings.size(1), device=node_embeddings.device)
        node_embeddings = torch.cat([node_embeddings, padding_embedding], dim=0)  # New padding node at index `padding_idx`
        padding_idx = node_embeddings.size(0) - 1  # Index of the padding node

        # Step 2: Ensure each graph has exactly context_len cliques by padding with empty cliques
        padded_cliques_batch = [
            graph[:context_len] + [[] for _ in range(context_len - len(graph))]
            if len(graph) < context_len else graph[:context_len]
            for graph in cliques_batch
        ]

        # Step 3: Flatten all cliques and prepare masks
        all_cliques = []
        clique_masks = []
        for graph in padded_cliques_batch:
            for clique in graph:
                if clique:
                    all_cliques.append(clique)
                    clique_masks.append(False)
                else:
                    all_cliques.append([padding_idx])  # Pad with padding_idx
                    clique_masks.append(True)
        
        total_cliques = batch_size * context_len

        # Step 4: Compute clique sizes
        clique_sizes = torch.tensor(
            [len(clique) if not mask else 0 for clique, mask in zip(all_cliques, clique_masks)], 
            device=self.device
        )

        max_clique_size = clique_sizes.max().item() if clique_sizes.numel() > 0 else 0

        # Step 5: Get clique size embeddings
        size_indices = torch.arange(0, max_clique_size + 1, device=self.device).unsqueeze(1).float()
        if use_r:
            clique_size_embeddings = self.clique_size_embedder_1(size_indices)  # [max_size+1, embed_dim_size]
        else:
            clique_size_embeddings = self.clique_size_embedder_2(size_indices)

        # Step 6: Handle case with no cliques
        if total_cliques == 0:
            embed_dim = node_embeddings.shape[-1] + 1
            clique_embeddings_batch = torch.zeros(batch_size, context_len, embed_dim, device=self.device)
            clique_masks_batch = torch.ones(batch_size, context_len, dtype=torch.bool, device=self.device)
            return clique_embeddings_batch, clique_masks_batch

        # Step 7: Determine the maximum clique size for padding
        max_clique_size = max(len(clique) for clique in all_cliques) if all_cliques else 0

        # Step 8: Pad cliques with padding_idx to have uniform size
        padded_cliques = [
            clique + [padding_idx] * (max_clique_size - len(clique)) if len(clique) < max_clique_size else clique
            for clique in all_cliques
        ]

        cliques_tensor = torch.tensor(padded_cliques, device=self.device, dtype=torch.long)  # [total_cliques, max_clique_size]

        # Step 9: Create mask for valid nodes (padding_idx indicates padding)
        mask = cliques_tensor != padding_idx  # [total_cliques, max_clique_size]

        # Step 10: Compute sum of node embeddings where mask is True
        node_sum = node_embeddings[cliques_tensor] * mask.unsqueeze(-1)  # [total_cliques, max_clique_size, embed_dim-1]
        node_mean = node_sum.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)  # [total_cliques, embed_dim-1]

        # Step 11: Assign size embeddings
        size_embeds = clique_size_embeddings[clique_sizes]  # [total_cliques, embed_dim_size]

        # Step 12: Combine embeddings
        embed_dim = node_embeddings.shape[-1] + 1
        res = torch.cat([node_mean, size_embeds], dim=-1)  # [total_cliques, embed_dim]

        # Step 13: Reshape to [batch_size, context_len, embed_dim]
        clique_embeddings_batch = res.view(batch_size, context_len, embed_dim)

        # Step 14: Create masks: True where cliques are padded (originally missing)
        clique_masks_batch = torch.tensor(clique_masks, device=self.device).view(batch_size, context_len)

        return clique_embeddings_batch, clique_masks_batch

    
    def cliques_attention_encoder(self, use_r, clique_embeddings, clique_masks):
        """
        Perform attention over the cliques in the graph.
        """
        num_cliques = clique_embeddings.shape[0]
        if num_cliques > self.clique_attention_context_len:
            clique_embeddings = clique_embeddings[:,:self.clique_attention_context_len,:]
        original_embeddings = clique_embeddings
        if use_r:
            # Flatten the first two dimenions to get the shape [batch_size * clique_attention_context_len, embed_dim] before applying the projections
            clique_embeddings_reshaped = clique_embeddings.view(-1, clique_embeddings.shape[-1])
            Q = self.clique_Wq_1(clique_embeddings_reshaped)
            K = self.clique_Wk_1(clique_embeddings_reshaped)
            V = self.clique_Wv_1(clique_embeddings_reshaped)
            # Transform the queries, keys and values back to the original shape
            Q = Q.view(clique_embeddings.shape)
            K = K.view(clique_embeddings.shape)
            V = V.view(clique_embeddings.shape)
                      
            attention_output = self.clique_attention_1(Q, K, V, key_padding_mask=clique_masks)[0]
            # Apply layer normalization and residual connection
            attention_output = self.clique_layer_norm_11(attention_output) + original_embeddings
            
            # Apply a feed forward network
            attention_output_before_ff = attention_output
            attention_output = self.clique_nn_1(attention_output)
            # Apply layer normalization and residual connection
            attention_output = self.clique_layer_norm_12(attention_output) + attention_output_before_ff
            
            # Flatten and project down to the original embedding size
            attention_output = attention_output.flatten(1)
            
            graph_embedding = self.clique_downward_projection_1(attention_output)
        else:
            # Flatten the first two dimenions to get the shape [batch_size * clique_attention_context_len, embed_dim] before applying the projections
            clique_embeddings_reshaped = clique_embeddings.view(-1, clique_embeddings.shape[-1])
            Q = self.clique_Wq_2(clique_embeddings_reshaped)
            K = self.clique_Wk_2(clique_embeddings_reshaped)
            V = self.clique_Wv_2(clique_embeddings_reshaped)
            # Transform the queries, keys and values back to the original shape
            Q = Q.view(clique_embeddings.shape)
            K = K.view(clique_embeddings.shape)
            V = V.view(clique_embeddings.shape)
                       
            attention_output = self.clique_attention_2(Q, K, V, key_padding_mask=clique_masks)[0]
            # Apply layer normalization and residual connection
            attention_output = self.clique_layer_norm_21(attention_output) + original_embeddings
            
            # Apply a feed forward network
            attention_output_before_ff = attention_output
            attention_output = self.clique_nn_2(attention_output)
            # Apply layer normalization and residual connection
            attention_output = self.clique_layer_norm_22(attention_output) + attention_output_before_ff
            
            # Flatten and project down to the original embedding size
            attention_output = attention_output.flatten(1)
            
            graph_embedding = self.clique_downward_projection_2(attention_output)
        # It can happen for disconnected graphs that we have no cliques (which we will punish later)
        # Replace all NaN values with zeros
        graph_embedding[torch.isnan(graph_embedding)] = 0
        return graph_embedding