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
    def __init__(self, observation_space, n, r, b, not_connected_punishment, features_dim):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.n = n
        self.r = r
        self.b = b
        self.not_connected_punishment = not_connected_punishment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        embed_dim = 128
        num_heads = 2
        self.node_attention_context_len = 10
        self.clique_attention_context_len = 128
        
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
        self.node_downward_projection_1 = nn.Linear(self.node_attention_context_len*embed_dim, embed_dim)
        
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
        self.node_downward_projection_2 = nn.Linear(self.node_attention_context_len*embed_dim, embed_dim)
        
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
        assert observations.shape[0] == 1, "Only batch size 1 is supported at this point"
        G = obs_space_to_graph(observations[0], self.n)
        _, cliques_r, cliques_b, _ = get_score_and_cliques(G, self.r, self.b, self.not_connected_punishment)
        
        node_embeddings_1 = self.node_embeddings(True) #[n, embed_dim]
        clique_embeddings_1 = self.clique_attention_encoder(True, node_embeddings_1, cliques_r) # [num_cliques_r, embed_dim]
        graph_embedding_1 = self.graph_attention_encoder(True, clique_embeddings_1) # [embed_dim]
        
        node_embeddings_2 = self.node_embeddings(False) #[n, embed_dim]
        clique_embeddings_2 = self.clique_attention_encoder(False, node_embeddings_2, cliques_b) # [num_cliques_b, embed_dim]
        graph_embedding_2 = self.graph_attention_encoder(False, clique_embeddings_2) # [embed_dim]

        graph_embedding = torch.cat([graph_embedding_1, graph_embedding_2], dim=0)
        return graph_embedding
    
    def node_embeddings(self, use_r):
        nodes_one_hot = torch.zeros(self.n, self.n, device=self.device, dtype=torch.float32)
        for i in range(self.n):
            nodes_one_hot[i, i] = 1
        if use_r:
            return self.node_embedder_1(nodes_one_hot)
        else:
            return self.node_embedder_2(nodes_one_hot)
        
    def clique_attention_encoder(self, use_r, node_embeddings, cliques):
        """
        Perform attention over the nodes in the cliques.
        """
        if use_r:
            Q = self.node_Wq_1(node_embeddings)
            K = self.node_Wk_1(node_embeddings)
            V = self.node_Wv_1(node_embeddings)
            
            queries, keys, values, original_embeddings, key_padding_masks = self.select_clique_node_embeddings(node_embeddings, cliques, Q, K, V)
            
            attention_output = self.node_attention_1(queries, keys, values, key_padding_mask=key_padding_masks)[0]
            # Apply layer normalization and residual connection
            attention_output = self.node_layer_norm_11(attention_output) + original_embeddings
            
            # Apply a feed forward network
            attention_output_before_ff = attention_output
            attention_output = self.node_nn_1(attention_output)
            # Apply layer normalization and residual connection
            attention_output = self.node_layer_norm_12(attention_output) + attention_output_before_ff
            
            # Flatten and project down to the original embedding size
            attention_output = attention_output.flatten(1)
            
            return self.node_downward_projection_1(attention_output)
        else:
            Q = self.node_Wq_2(node_embeddings)
            K = self.node_Wk_2(node_embeddings)
            V = self.node_Wv_2(node_embeddings)
            
            queries, keys, values, original_embeddings, key_padding_masks = self.select_clique_node_embeddings(node_embeddings, cliques, Q, K, V)
            
            attention_output = self.node_attention_2(queries, keys, values, key_padding_mask=key_padding_masks)[0]
            # Apply layer normalization and residual connection
            attention_output = self.node_layer_norm_21(attention_output) + original_embeddings
            
            # Apply a feed forward network
            attention_output_before_ff = attention_output
            attention_output = self.node_nn_2(attention_output)
            # Apply layer normalization and residual connection
            attention_output = self.node_layer_norm_22(attention_output) + attention_output_before_ff
            
            # Flatten and project down to the original embedding size
            attention_output = attention_output.flatten(1)
            
            return self.node_downward_projection_2(attention_output)
    
    def select_clique_node_embeddings(self, node_embeddings, cliques, Q, K, V):
        """
        Select the node embeddings for every clique (and add padding if necessary) and return the queries, keys, values and original embeddings.
        """
        queries, keys, values = [], [], []
        original_embeddings = []
        key_padding_masks = []
        for clique in cliques:      
            Q_clique = Q[clique]
            K_clique = K[clique]
            V_clique = V[clique]
            original_embedding = node_embeddings[clique]
            if len(clique) < self.node_attention_context_len: # Add padding
                Q_clique = torch.cat([Q_clique, torch.zeros(self.node_attention_context_len - len(clique), Q_clique.shape[1], device=self.device)])
                K_clique = torch.cat([K_clique, torch.zeros(self.node_attention_context_len - len(clique), K_clique.shape[1], device=self.device)])
                V_clique = torch.cat([V_clique, torch.zeros(self.node_attention_context_len - len(clique), V_clique.shape[1], device=self.device)])
                original_embedding = torch.cat([original_embedding, torch.zeros(self.node_attention_context_len - len(clique), original_embedding.shape[1], device=self.device)])
                key_padding_mask = torch.zeros(self.node_attention_context_len, dtype=torch.bool, device=self.device)
                key_padding_mask[len(clique):] = True
            else:
                key_padding_mask = None
                if len(clique) > self.node_attention_context_len:
                    Q_clique = Q_clique[:self.node_attention_context_len]
                    K_clique = K_clique[:self.node_attention_context_len]
                    V_clique = V_clique[:self.node_attention_context_len]
                    original_embedding = original_embedding[:self.node_attention_context_len]
            queries.append(Q_clique)
            keys.append(K_clique)
            values.append(V_clique)
            original_embeddings.append(original_embedding)
            key_padding_masks.append(key_padding_mask)
        queries = torch.stack(queries)
        keys = torch.stack(keys)
        values = torch.stack(values)
        original_embeddings = torch.stack(original_embeddings)
        key_padding_masks = torch.stack(key_padding_masks)
        return queries, keys, values, original_embeddings, key_padding_masks
    
    def graph_attention_encoder(self, use_r, clique_embeddings):
        """
        Perform attention over the cliques in the graph.
        """
        num_cliques = clique_embeddings.shape[0]
        key_padding_mask = None
        if num_cliques > self.clique_attention_context_len:
            clique_embeddings = clique_embeddings[:self.clique_attention_context_len]
        original_embeddings = clique_embeddings
        if use_r:
            Q = self.clique_Wq_1(clique_embeddings)
            K = self.clique_Wk_1(clique_embeddings)
            V = self.clique_Wv_1(clique_embeddings)
            
            num_cliques = clique_embeddings.shape[0]
            if num_cliques < self.clique_attention_context_len: # Add padding
                Q = torch.cat([Q, torch.zeros(self.clique_attention_context_len - num_cliques, Q.shape[1], device=self.device)])
                K = torch.cat([K, torch.zeros(self.clique_attention_context_len - num_cliques, K.shape[1], device=self.device)])
                V = torch.cat([V, torch.zeros(self.clique_attention_context_len - num_cliques, V.shape[1], device=self.device)])
                original_embeddings = torch.cat([clique_embeddings, torch.zeros(self.clique_attention_context_len - num_cliques, clique_embeddings.shape[1], device=self.device)])
                key_padding_mask = torch.zeros(self.clique_attention_context_len, dtype=torch.bool, device=self.device)
                key_padding_mask[num_cliques:] = True
                       
            attention_output = self.clique_attention_1(Q[None], K[None], V[None], key_padding_mask=key_padding_mask)[0]
            # Apply layer normalization and residual connection
            attention_output = self.clique_layer_norm_11(attention_output) + original_embeddings[None]
            
            # Apply a feed forward network
            attention_output_before_ff = attention_output
            attention_output = self.clique_nn_1(attention_output)
            # Apply layer normalization and residual connection
            attention_output = self.clique_layer_norm_12(attention_output) + attention_output_before_ff
            
            # Flatten and project down to the original embedding size
            attention_output = attention_output.flatten(1)
            
            return self.clique_downward_projection_1(attention_output)[0]
        else:
            Q = self.clique_Wq_2(clique_embeddings)
            K = self.clique_Wk_2(clique_embeddings)
            V = self.clique_Wv_2(clique_embeddings)
            
            num_cliques = clique_embeddings.shape[0]
            if num_cliques < self.clique_attention_context_len: # Add padding
                Q = torch.cat([Q, torch.zeros(self.clique_attention_context_len - num_cliques, Q.shape[1], device=self.device)])
                K = torch.cat([K, torch.zeros(self.clique_attention_context_len - num_cliques, K.shape[1], device=self.device)])
                V = torch.cat([V, torch.zeros(self.clique_attention_context_len - num_cliques, V.shape[1], device=self.device)])
                original_embeddings = torch.cat([clique_embeddings, torch.zeros(self.clique_attention_context_len - num_cliques, clique_embeddings.shape[1], device=self.device)])
                key_padding_mask = torch.zeros(self.clique_attention_context_len, device=self.device)
                key_padding_mask[num_cliques:] = 1
                       
            attention_output = self.clique_attention_2(Q[None], K[None], V[None], key_padding_mask=key_padding_mask)[0]
            # Apply layer normalization and residual connection
            attention_output = self.clique_layer_norm_21(attention_output) + original_embeddings[None]
            
            # Apply a feed forward network
            attention_output_before_ff = attention_output
            attention_output = self.clique_nn_2(attention_output)
            # Apply layer normalization and residual connection
            attention_output = self.clique_layer_norm_22(attention_output) + attention_output_before_ff
            
            # Flatten and project down to the original embedding size
            attention_output = attention_output.flatten(1)
            
            return self.clique_downward_projection_2(attention_output)[0]