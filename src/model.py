import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np

from score import get_score_and_cliques
from env import obs_space_to_graph


def get_one_hot_encoding(n, i):
    """Returns the one-hot encoding of the integer i."""
    encoding = np.zeros(n)
    encoding[i] = 1
    return encoding

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n, r, b, not_connected_punishment, features_dim, num_heads, node_attention_context_len, clique_attention_context_len):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.n = n
        self.r = r
        self.b = b
        self.not_connected_punishment = not_connected_punishment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # Some hyperparameters
        embed_dim = features_dim//2
        num_heads = 2
        self.node_attention_context_len = node_attention_context_len
        self.clique_attention_context_len = clique_attention_context_len
        
        self.node_embedder_1 = nn.Linear(n, embed_dim)
        self.node_Wq_1 = nn.Linear(embed_dim, embed_dim)
        self.node_Wk_1 = nn.Linear(embed_dim, embed_dim)
        self.node_Wv_1 = nn.Linear(embed_dim, embed_dim)
        self.node_attention_1 = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.node_layer_norm_11 = torch.nn.LayerNorm(embed_dim)
        self.node_nn_1 = nn.Sequential( # NN after attention
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.node_layer_norm_12 = torch.nn.LayerNorm(embed_dim)
        self.node_downward_projection_1 = nn.Linear(self.node_attention_context_len*embed_dim, embed_dim-1)
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
        
        self.node_embedder_2 = nn.Linear(n, embed_dim)
        self.node_Wq_2 = nn.Linear(embed_dim, embed_dim)
        self.node_Wk_2 = nn.Linear(embed_dim, embed_dim)
        self.node_Wv_2 = nn.Linear(embed_dim, embed_dim)
        self.node_attention_2 = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.node_layer_norm_21 = torch.nn.LayerNorm(embed_dim)
        self.node_nn_2 = nn.Sequential( # NN after attention
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.node_layer_norm_22 = torch.nn.LayerNorm(embed_dim)
        self.node_downward_projection_2 = nn.Linear(self.node_attention_context_len*embed_dim, embed_dim-1)
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
        
        
    def forward(self, observations):
        node_embeddings_1 = self.node_embeddings(True) #[n, embed_dim]
        node_embeddings_2 = self.node_embeddings(False) #[n, embed_dim]
        
        graph_embeddings = []
        cliques_r_batch, cliques_b_batch = [], []
        for observation in observations:
            G = obs_space_to_graph(observation, self.n)
            _, cliques_r, cliques_b, _ = get_score_and_cliques(G, self.r, self.b, self.not_connected_punishment)
            cliques_r_batch.append(cliques_r)
            cliques_b_batch.append(cliques_b)
            
        clique_embeddings_1, clique_masks_batch_1 = self.clique_attention_encoder(True, node_embeddings_1, cliques_r_batch) # [batch_size, clique_attention_context_len, embed_dim], [batch_size, clique_attention_context_len]
        graph_embedding_1 = self.graph_attention_encoder(True, clique_embeddings_1, clique_masks_batch_1) # [batch_size, embed_dim]
        
        clique_embeddings_2, clique_masks_batch_2 = self.clique_attention_encoder(False, node_embeddings_2, cliques_b_batch) # [batch_size, clique_attention_context_len, embed_dim], [batch_size, clique_attention_context_len]
        graph_embedding_2 = self.graph_attention_encoder(False, clique_embeddings_2, clique_masks_batch_2) # [batch_size, embed_dim]

        graph_embeddings = torch.cat([graph_embedding_1, graph_embedding_2], dim=1)
        return graph_embeddings
    
    
    def node_embeddings(self, use_r):
        nodes_one_hot = torch.zeros(self.n, self.n, device=self.device, dtype=torch.float32)
        for i in range(self.n):
            nodes_one_hot[i, i] = 1
        if use_r:
            return self.node_embedder_1(nodes_one_hot)
        else:
            return self.node_embedder_2(nodes_one_hot)
       
        
    def clique_attention_encoder(self, use_r, node_embeddings, cliques_batch):
        """
        Perform attention over the nodes in the cliques.
        """
        if use_r:
            Q = self.node_Wq_1(node_embeddings)
            K = self.node_Wk_1(node_embeddings)
            V = self.node_Wv_1(node_embeddings)
            
            queries_batch, keys_batch, values_batch, original_embeddings_batch, key_padding_masks_batch = self.select_clique_node_embeddings(node_embeddings, cliques_batch, Q, K, V)
            
            attention_output = self.node_attention_1(queries_batch, keys_batch, values_batch, key_padding_mask=key_padding_masks_batch)[0] #[batch_size * num_cliques, node_attention_context_len, embed_dim]
            
            # Apply layer normalization and residual connection
            attention_output = self.node_layer_norm_11(attention_output) + original_embeddings_batch
            
            # Apply a feed forward network
            attention_output_before_ff = attention_output
            attention_output = self.node_nn_1(attention_output)
            # Apply layer normalization and residual connection
            attention_output = self.node_layer_norm_12(attention_output) + attention_output_before_ff
            
            # Flatten and project down to the original embedding size
            attention_output = attention_output.flatten(1)
            
            clique_embeddings_batch = self.node_downward_projection_1(attention_output) # [batch_size * num_cliques, embed_dim-1]
            max_clique_size = max(len(clique) for cliques in cliques_batch for clique in cliques)
            clique_size_embeddings = self.clique_size_embedder_1(torch.tensor([[i] for i in range(max_clique_size+1)], device=self.device, dtype=torch.float32))
        else:
            Q = self.node_Wq_2(node_embeddings)
            K = self.node_Wk_2(node_embeddings)
            V = self.node_Wv_2(node_embeddings)
            
            queries_batch, keys_batch, values_batch, original_embeddings_batch, key_padding_masks_batch = self.select_clique_node_embeddings(node_embeddings, cliques_batch, Q, K, V)
            
            attention_output = self.node_attention_2(queries_batch, keys_batch, values_batch, key_padding_mask=key_padding_masks_batch)[0] #[batch_size * num_cliques, node_attention_context_len, embed_dim]
            
            # Apply layer normalization and residual connection
            attention_output = self.node_layer_norm_21(attention_output) + original_embeddings_batch
            
            # Apply a feed forward network
            attention_output_before_ff = attention_output
            attention_output = self.node_nn_2(attention_output)
            # Apply layer normalization and residual connection
            attention_output = self.node_layer_norm_22(attention_output) + attention_output_before_ff
            
            # Flatten and project down to the original embedding size
            attention_output = attention_output.flatten(1)
            
            clique_embeddings_batch = self.node_downward_projection_2(attention_output) # [batch_size * num_cliques, embed_dim-1]
            max_clique_size = max(len(clique) for cliques in cliques_batch for clique in cliques)
            clique_size_embeddings = self.clique_size_embedder_2(torch.tensor([[i] for i in range(max_clique_size+1)], device=self.device, dtype=torch.float32))
        
        # Create an output array of shape [batch_size, clique_attention_context_len, embed_dim]
        clique_size_embeddings_batch = []
        clique_masks_batch = []
        curr_index = 0
        for batch_id in range(len(cliques_batch)):
            num_cliques = len(cliques_batch[batch_id])
            clique_sizes = [len(clique) for clique in cliques_batch[batch_id]]
            clique_size_embeddings = clique_size_embeddings[clique_sizes]
            clique_embeddings = clique_embeddings_batch[curr_index:curr_index+num_cliques]
            clique_mask = torch.zeros(self.clique_attention_context_len, dtype=torch.bool, device=self.device)
            if num_cliques < self.clique_attention_context_len:
                clique_embeddings = torch.cat([clique_embeddings, torch.zeros(self.clique_attention_context_len - num_cliques, clique_embeddings.shape[1], device=self.device)])
                clique_size_embeddings = torch.cat([clique_size_embeddings, torch.zeros(self.clique_attention_context_len - num_cliques, clique_size_embeddings.shape[1], device=self.device)])
                clique_mask[num_cliques:] = True
            elif num_cliques > self.clique_attention_context_len:
                clique_embeddings = clique_embeddings[:self.clique_attention_context_len]
                clique_size_embeddings = clique_size_embeddings[:self.clique_attention_context_len]
            curr_index += num_cliques
            clique_size_embeddings_batch.append(torch.cat([clique_embeddings, clique_size_embeddings], dim=1))
            clique_masks_batch.append(clique_mask)
            
        return torch.stack(clique_size_embeddings_batch), torch.stack(clique_masks_batch)
    
    def select_clique_node_embeddings(self, node_embeddings, cliques_batch, Q, K, V):
        """
        Select the node embeddings for every clique (and add padding if necessary) and return the queries, keys, values and original embeddings.
        """
        queries_batch, keys_batch, values_batch, original_embeddings_batch, key_padding_masks_batch = [], [], [], [], []
        for cliques in cliques_batch:
            queries, keys, values = [], [], []
            original_embeddings = []
            key_padding_masks = []
            for clique in cliques:      
                Q_clique = Q[clique]
                K_clique = K[clique]
                V_clique = V[clique]
                original_embedding = node_embeddings[clique]
                key_padding_mask = torch.zeros(self.node_attention_context_len, dtype=torch.bool, device=self.device)
                if len(clique) < self.node_attention_context_len: # Add padding
                    Q_clique = torch.cat([Q_clique, torch.zeros(self.node_attention_context_len - len(clique), Q_clique.shape[1], device=self.device)])
                    K_clique = torch.cat([K_clique, torch.zeros(self.node_attention_context_len - len(clique), K_clique.shape[1], device=self.device)])
                    V_clique = torch.cat([V_clique, torch.zeros(self.node_attention_context_len - len(clique), V_clique.shape[1], device=self.device)])
                    original_embedding = torch.cat([original_embedding, torch.zeros(self.node_attention_context_len - len(clique), original_embedding.shape[1], device=self.device)])
                    key_padding_mask[len(clique):] = True
                elif len(clique) > self.node_attention_context_len:
                    Q_clique = Q_clique[:self.node_attention_context_len]
                    K_clique = K_clique[:self.node_attention_context_len]
                    V_clique = V_clique[:self.node_attention_context_len]
                    original_embedding = original_embedding[:self.node_attention_context_len]
                queries.append(Q_clique)
                keys.append(K_clique)
                values.append(V_clique)
                original_embeddings.append(original_embedding)
                key_padding_masks.append(key_padding_mask)
            queries_batch.append(torch.stack(queries))
            keys_batch.append(torch.stack(keys))
            values_batch.append(torch.stack(values))
            original_embeddings_batch.append(torch.stack(original_embeddings))
            key_padding_masks_batch.append(torch.stack(key_padding_masks))
        queries_batch = torch.cat(queries_batch)
        keys_batch = torch.cat(keys_batch)
        values_batch = torch.cat(values_batch)
        original_embeddings_batch = torch.cat(original_embeddings_batch)
        key_padding_masks_batch = torch.cat(key_padding_masks_batch)
        return queries_batch, keys_batch, values_batch, original_embeddings_batch, key_padding_masks_batch
    
    
    def graph_attention_encoder(self, use_r, clique_embeddings, clique_masks):
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
            
            return self.clique_downward_projection_1(attention_output)
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
            
            return self.clique_downward_projection_2(attention_output)