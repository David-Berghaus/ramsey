import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import comb

# Import existing functionality
from env import obs_space_to_graph, flattened_off_diagonal_to_adjacency_matrix
from score import get_score_and_cliques, get_cliques_and_count
from model import NodeMeanPoolCliqueAttentionFeatureExtractor
from graphs_cache import GraphsCache, hash_tensor

# Custom Dataset class for graph adjacency matrices
class GraphCliqueDataset(Dataset):
    def __init__(self, adjacency_matrices, clique_counts):
        """
        Args:
            adjacency_matrices (list): List of flattened adjacency matrices
            clique_counts (list): List of tuples (count_original, count_complement)
        """
        self.adjacency_matrices = adjacency_matrices
        self.clique_counts = clique_counts
        
    def __len__(self):
        return len(self.adjacency_matrices)
    
    def __getitem__(self, idx):
        return self.adjacency_matrices[idx], self.clique_counts[idx]

# Function to generate dataset of random graphs with their maximum clique sizes
def generate_clique_dataset(n_vertices=17, n_samples=1000, seed=42):
    """
    Generate a dataset of random graphs and their 4-clique counts.
    
    Args:
        n_vertices (int): Number of vertices in each graph
        n_samples (int): Number of graph samples to generate
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (flattened_adjacency_matrices, clique_counts)
    """
    np.random.seed(seed)
    
    # Number of entries in flattened adjacency matrix
    n_entries = n_vertices * (n_vertices - 1) // 2
    
    # Initialize arrays to store data
    flattened_matrices = []
    clique_counts = []
    
    # Parameters for Ramsey analysis
    r = 4  # We're looking for 4-cliques
    b = 4  # And 4-independent sets (4-cliques in complement)
    
    for _ in tqdm(range(n_samples), desc="Generating graphs"):
        # Generate random adjacency matrix
        flattened_adj = np.random.randint(2, size=n_entries)
        
        # Convert to graph
        G = obs_space_to_graph(flattened_adj, n_vertices)
        G_complement = nx.complement(G)
        
        # Get clique counts using the same function as in the RL approach
        _, cliques_r, cliques_b, _ = get_score_and_cliques(G, r, b, -1000)
        
        # Get counts of cliques of size exactly r and b
        count_r = 0
        for clique in cliques_r:
            if len(clique) == r:
                count_r += 1
            elif len(clique) > r:
                count_r += comb(len(clique), r, exact=True)
                
        count_b = 0
        for clique in cliques_b:
            if len(clique) == b:
                count_b += 1
            elif len(clique) > b:
                count_b += comb(len(clique), b, exact=True)
        
        # Save data
        flattened_matrices.append(flattened_adj)
        clique_counts.append((count_r, count_b))
    
    return np.array(flattened_matrices), np.array(clique_counts)

# Model 1: Custom architecture adapted from the existing RL implementation
class CustomCliquePredictor(nn.Module):
    def __init__(self, n_vertices=17, features_dim=256):
        super(CustomCliquePredictor, self).__init__()
        
        # Store features_dim as an instance variable
        self.features_dim = features_dim
        
        # Create dummy observation space for the feature extractor
        dummy_obs_space = None  # This will be handled in forward
        
        # Number of entries in flattened adjacency matrix
        self.n_entries = n_vertices * (n_vertices - 1) // 2
        
        # Parameters for feature extractor
        self.n = n_vertices
        self.r = 4  # Parameter for R(4,4)
        self.b = 4  # Parameter for R(4,4)
        self.not_connected_punishment = -1000
        self.node_attention_context_len = 20
        self.clique_attention_context_len = 20
        self.num_heads = 2
        
        # Create feature extractor (will be initialized in forward)
        self.feature_extractor = None
        
        # Since the features_dim is split into two equal parts for R and B cliques
        # Each regression head should take features_dim//2 as input
        regressor_input_dim = features_dim // 2
        
        # Regression heads for clique size prediction
        self.original_regressor = nn.Sequential(
            nn.Linear(regressor_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.complement_regressor = nn.Sequential(
            nn.Linear(regressor_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.graphs_cache = GraphsCache(10000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Batch of flattened adjacency matrices [batch_size, n_entries]
            
        Returns:
            tuple: (clique_size_original, clique_size_complement)
        """
        batch_size = x.shape[0]
        
        # Initialize feature extractor if not already done
        if self.feature_extractor is None:
            # Dummy observation space
            from gymnasium import spaces
            observation_space = spaces.MultiBinary(self.n_entries)
            
            self.feature_extractor = NodeMeanPoolCliqueAttentionFeatureExtractor(
                observation_space=observation_space,
                n=self.n,
                r=self.r,
                b=self.b,
                not_connected_punishment=self.not_connected_punishment,
                features_dim=self.features_dim,
                num_heads=self.num_heads,
                node_attention_context_len=self.node_attention_context_len,
                clique_attention_context_len=self.clique_attention_context_len
            ).to(self.device)
        
        # Use the feature extractor to get embeddings
        # The output is of shape [batch_size, features_dim]
        all_features = self.feature_extractor(x)
        
        # The first half of the features correspond to original graph
        # The second half correspond to the complement graph
        features_half = all_features.shape[1] // 2
        
        # Apply regression heads
        clique_size_original = self.original_regressor(all_features[:, :features_half])
        clique_size_complement = self.complement_regressor(all_features[:, features_half:])
        
        return torch.cat((clique_size_original, clique_size_complement), dim=1)

# Model 2: MLP Baseline
class MLPCliquePredictor(nn.Module):
    def __init__(self, n_vertices=17, hidden_dims=[512, 512, 256, 128]):
        super(MLPCliquePredictor, self).__init__()
        
        # Number of entries in flattened adjacency matrix
        self.n_entries = n_vertices * (n_vertices - 1) // 2
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build MLP layers
        layers = []
        input_dim = self.n_entries
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Output layers for original and complement clique sizes
        self.original_head = nn.Linear(hidden_dims[-1], 1)
        self.complement_head = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Batch of flattened adjacency matrices [batch_size, n_entries]
            
        Returns:
            tuple: (clique_size_original, clique_size_complement)
        """
        features = self.mlp(x)
        
        clique_size_original = self.original_head(features)
        clique_size_complement = self.complement_head(features)
        
        return torch.cat((clique_size_original, clique_size_complement), dim=1)

# Model 3: GNN with Clique Attention from NodeMeanPoolCliqueAttentionFeatureExtractor
class RamseyGraphGNNWithCliqueAttention(nn.Module):
    """
    An optimized hybrid model that combines traditional GNN layers with the clique attention mechanism
    from NodeMeanPoolCliqueAttentionFeatureExtractor for Ramsey graph analysis.
    """
    def __init__(self, n_vertices=17, r=4, b=4, hidden_dim=64, num_layers=3, 
                 clique_attention_context_len=8, node_attention_context_len=8):
        super(RamseyGraphGNNWithCliqueAttention, self).__init__()
        self.n = n_vertices
        self.r = r
        self.b = b
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.clique_attention_context_len = clique_attention_context_len
        self.node_attention_context_len = node_attention_context_len
        self.not_connected_punishment = -1000  # Same as in NodeMeanPoolCliqueAttentionFeatureExtractor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graphs_cache = GraphsCache(10000)
        
        # SIMPLIFIED GNN Node Feature Extraction - using fewer features for efficiency
        # Initial node feature dimensions
        node_feat_dim = 16
        
        # GNN layers
        self.node_embedding = nn.Linear(1, node_feat_dim)  # Initial node feature embedding
        
        # Message passing layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = node_feat_dim if i == 0 else hidden_dim
            self.gnn_layers.append(nn.Linear(layer_input_dim * 2, hidden_dim))  # Message function
            
        # Graph-level readout
        self.graph_readout = nn.Linear(hidden_dim, hidden_dim)
        
        # SIMPLIFIED CLIQUE ATTENTION - inspired by the attention mechanism in the feature extractor
        # but optimized for this specific task
        
        # Clique embedding layer
        self.r_clique_embedding = nn.Linear(self.r, hidden_dim)
        self.b_clique_embedding = nn.Linear(self.b, hidden_dim)
        
        # Attention mechanisms
        self.r_query = nn.Linear(hidden_dim, hidden_dim)
        self.r_key = nn.Linear(hidden_dim, hidden_dim)
        self.r_value = nn.Linear(hidden_dim, hidden_dim)
        
        self.b_query = nn.Linear(hidden_dim, hidden_dim)
        self.b_key = nn.Linear(hidden_dim, hidden_dim)
        self.b_value = nn.Linear(hidden_dim, hidden_dim)
        
        # Downward projections for clique attention
        self.r_downward = nn.Linear(clique_attention_context_len * hidden_dim, hidden_dim)
        self.b_downward = nn.Linear(clique_attention_context_len * hidden_dim, hidden_dim)
        
        # Final prediction layers
        self.final_r_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.final_b_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.r_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.b_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Create a module for the nonlinearity used throughout
        self.act = nn.ReLU()
        
        # Cache for graphs
        self.n_entries = n_vertices * (n_vertices - 1) // 2
    
    def extract_node_features(self, adj_matrix):
        """
        Extract GNN node features from the adjacency matrix.
        This simplified GNN aggregates messages from neighbors and updates node representations.
        
        Args:
            adj_matrix (torch.Tensor): Adjacency matrix [n, n]
            
        Returns:
            torch.Tensor: Node features tensor [n, hidden_dim]
        """
        n = adj_matrix.size(0)
        
        # Initial node features (just a single feature of 1.0 for each node)
        node_features = torch.ones(n, 1, device=adj_matrix.device)
        node_features = self.node_embedding(node_features)  # [n, node_feat_dim]
        
        # GNN message passing
        for i, layer in enumerate(self.gnn_layers):
            # Aggregate messages from neighbors
            messages = []
            for j in range(n):
                # Get neighbors (where adjacency matrix entry is 1)
                neighbors = (adj_matrix[j] == 1).nonzero(as_tuple=True)[0]
                
                if len(neighbors) > 0:
                    # Aggregate neighbor features
                    neighbor_feats = node_features[neighbors]
                    neighbor_aggr = neighbor_feats.mean(dim=0)
                else:
                    # No neighbors - use zero vector
                    neighbor_aggr = torch.zeros_like(node_features[0])
                
                # Concatenate node's own features with aggregated neighbor features
                combined = torch.cat([node_features[j], neighbor_aggr])
                messages.append(combined)
            
            # Stack messages and apply layer
            stacked_messages = torch.stack(messages)
            node_features = self.act(layer(stacked_messages))
        
        return node_features
    
    def forward(self, x):
        """
        Forward pass of the GNN with Clique Attention model.
        
        Args:
            x (torch.Tensor): Batch of flattened adjacency matrices [batch_size, n_entries]
            
        Returns:
            torch.Tensor: Predicted 4-clique counts [batch_size, 2]
        """
        batch_size = x.shape[0]
        
        # List to store per-graph embeddings
        all_gnn_embeddings = []
        r_attention_embeddings = []
        b_attention_embeddings = []
        
        # Process each graph in the batch
        for i in range(batch_size):
            graph_vec = x[i]
            
            # Convert flattened representation to adjacency matrix
            adj_mat = flattened_off_diagonal_to_adjacency_matrix(graph_vec, self.n)
            adj_tensor = torch.tensor(adj_mat, dtype=torch.float32, device=x.device)
            
            # Get GNN node embeddings
            node_embeddings = self.extract_node_features(adj_tensor)
            
            # Graph-level readout (mean pooling)
            graph_embedding = node_embeddings.mean(dim=0)
            graph_embedding = self.graph_readout(graph_embedding)
            
            # Store GNN embedding
            all_gnn_embeddings.append(graph_embedding)
            
            # Get r-cliques
            G = obs_space_to_graph(graph_vec.cpu().numpy(), self.n)
            G_comp = nx.complement(G)
            
            # Use the cache to avoid recomputation
            graph_key = hash_tensor(graph_vec.cpu().numpy())
            if graph_key in self.graphs_cache.cache:
                cliques_r, cliques_b, _ = self.graphs_cache.cache[graph_key]
            else:
                _, cliques_r, cliques_b, _ = get_score_and_cliques(G, self.r, self.b, self.not_connected_punishment)
                self.graphs_cache.put(graph_key, (cliques_r, cliques_b, None))
            
            # Process r-cliques for original graph
            r_clique_vectors = []
            for clique in cliques_r:
                if len(clique) >= self.r:
                    # One-hot encode the clique nodes
                    clique_vec = torch.zeros(self.n, device=x.device)
                    clique_vec[list(clique)] = 1.0
                    r_clique_vectors.append(clique_vec[:self.r])  # Truncate/pad to fixed size r
            
            if r_clique_vectors:
                # Stack clique vectors and get embeddings
                r_clique_stack = torch.stack(r_clique_vectors[:self.clique_attention_context_len])
                
                # Clique embedding
                r_embedded = self.r_clique_embedding(r_clique_stack)
                
                # Self-attention mechanism
                q = self.r_query(r_embedded)
                k = self.r_key(r_embedded)
                v = self.r_value(r_embedded)
                
                # Compute attention scores
                scores = torch.matmul(q, k.transpose(-2, -1))
                attention = torch.softmax(scores, dim=-1)
                
                # Apply attention
                attn_out = torch.matmul(attention, v)
                
                # Flatten and project
                attn_out_reshaped = attn_out.reshape(-1)
                # Ensure we have the right dimensionality for r_downward
                expected_dim = self.clique_attention_context_len * self.hidden_dim
                if attn_out_reshaped.shape[0] != expected_dim:
                    # Pad or truncate to match the expected dimensions
                    if attn_out_reshaped.shape[0] < expected_dim:
                        padding = torch.zeros(expected_dim - attn_out_reshaped.shape[0], device=x.device)
                        attn_out_reshaped = torch.cat([attn_out_reshaped, padding])
                    else:
                        attn_out_reshaped = attn_out_reshaped[:expected_dim]
                
                r_graph_embed = self.r_downward(attn_out_reshaped)
            else:
                # If all cliques are masked, use zeros
                r_graph_embed = torch.zeros(self.hidden_dim, device=x.device)
            
            r_attention_embeddings.append(r_graph_embed)
            
            # Process b-cliques for complement graph
            b_clique_vectors = []
            for clique in cliques_b:
                if len(clique) >= self.b:
                    # One-hot encode the clique nodes
                    clique_vec = torch.zeros(self.n, device=x.device)
                    clique_vec[list(clique)] = 1.0
                    b_clique_vectors.append(clique_vec[:self.b])  # Truncate/pad to fixed size b
            
            if b_clique_vectors:
                # Stack clique vectors and get embeddings
                b_clique_stack = torch.stack(b_clique_vectors[:self.clique_attention_context_len])
                
                # Clique embedding
                b_embedded = self.b_clique_embedding(b_clique_stack)
                
                # Self-attention mechanism
                q = self.b_query(b_embedded)
                k = self.b_key(b_embedded)
                v = self.b_value(b_embedded)
                
                # Compute attention scores
                scores = torch.matmul(q, k.transpose(-2, -1))
                attention = torch.softmax(scores, dim=-1)
                
                # Apply attention
                attn_out = torch.matmul(attention, v)
                
                # Flatten and project
                attn_out_reshaped = attn_out.reshape(-1)
                # Ensure we have the right dimensionality for b_downward
                expected_dim = self.clique_attention_context_len * self.hidden_dim
                if attn_out_reshaped.shape[0] != expected_dim:
                    # Pad or truncate to match the expected dimensions
                    if attn_out_reshaped.shape[0] < expected_dim:
                        padding = torch.zeros(expected_dim - attn_out_reshaped.shape[0], device=x.device)
                        attn_out_reshaped = torch.cat([attn_out_reshaped, padding])
                    else:
                        attn_out_reshaped = attn_out_reshaped[:expected_dim]
                
                b_graph_embed = self.b_downward(attn_out_reshaped)
            else:
                # If all cliques are masked, use zeros
                b_graph_embed = torch.zeros(self.hidden_dim, device=x.device)
            
            b_attention_embeddings.append(b_graph_embed)
        
        # Stack the embeddings
        gnn_embeddings = torch.stack(all_gnn_embeddings)
        r_attention_embeddings = torch.stack(r_attention_embeddings)
        b_attention_embeddings = torch.stack(b_attention_embeddings)
        
        # Combine GNN and attention embeddings
        r_combined = torch.cat([gnn_embeddings, r_attention_embeddings], dim=1)
        b_combined = torch.cat([gnn_embeddings, b_attention_embeddings], dim=1)
        
        # Project to final embeddings
        r_embedding = self.final_r_projection(r_combined)
        b_embedding = self.final_b_projection(b_combined)
        
        # Predict clique counts
        r_counts = self.r_head(r_embedding)
        b_counts = self.b_head(b_embedding)
        
        # Combine predictions
        predictions = torch.cat([r_counts, b_counts], dim=1)
        
        return predictions

# Function to train a model
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu', model_name="model"):
    """
    Train a model for predicting clique sizes.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        epochs (int): Number of training epochs
        lr (float): Learning rate
        device (str): Device to train on ('cpu' or 'cuda')
        model_name (str): Name for saving the model
        
    Returns:
        tuple: (trained_model, training_losses, validation_losses)
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    training_losses = []
    validation_losses = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for adjacency_matrices, max_clique_sizes in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            adjacency_matrices = adjacency_matrices.float().to(device)
            max_clique_sizes = max_clique_sizes.float().to(device)
            
            # Forward pass
            outputs = model(adjacency_matrices)
            loss = criterion(outputs, max_clique_sizes)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        training_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for adjacency_matrices, max_clique_sizes in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                adjacency_matrices = adjacency_matrices.float().to(device)
                max_clique_sizes = max_clique_sizes.float().to(device)
                
                outputs = model(adjacency_matrices)
                loss = criterion(outputs, max_clique_sizes)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        validation_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            torch.save(model.state_dict(), f"{model_name}_best.pt")
    
    # Load the best model
    model.load_state_dict(best_model_state)
    
    return model, training_losses, validation_losses

# Evaluation function
def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate a trained model.
    
    Args:
        model (nn.Module): The trained model
        test_loader (DataLoader): Test data loader
        device (str): Device to evaluate on
        
    Returns:
        tuple: (mse, mae, predictions, actual_values)
    """
    model = model.to(device)
    model.eval()
    
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    
    total_mse = 0.0
    total_mae = 0.0
    
    all_predictions = []
    all_actual = []
    
    with torch.no_grad():
        for adjacency_matrices, max_clique_sizes in tqdm(test_loader, desc="Evaluating"):
            adjacency_matrices = adjacency_matrices.float().to(device)
            max_clique_sizes = max_clique_sizes.float().to(device)
            
            outputs = model(adjacency_matrices)
            
            mse = criterion_mse(outputs, max_clique_sizes)
            mae = criterion_mae(outputs, max_clique_sizes)
            
            total_mse += mse.item()
            total_mae += mae.item()
            
            all_predictions.extend(outputs.cpu().numpy())
            all_actual.extend(max_clique_sizes.cpu().numpy())
    
    avg_mse = total_mse / len(test_loader)
    avg_mae = total_mae / len(test_loader)
    
    return avg_mse, avg_mae, np.array(all_predictions), np.array(all_actual)

# Function to visualize results
def visualize_results(predictions, actual_values, model_name="Model"):
    """
    Visualize the prediction results.
    
    Args:
        predictions (np.array): Predicted clique counts
        actual_values (np.array): Actual clique counts
        model_name (str): Name of the model for plot titles
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot for original graph clique counts
    axes[0].scatter(actual_values[:, 0], predictions[:, 0], alpha=0.5)
    axes[0].plot([min(actual_values[:, 0]), max(actual_values[:, 0])],
                [min(actual_values[:, 0]), max(actual_values[:, 0])], 'r--')
    axes[0].set_xlabel('Actual 4-Clique Count (Original)')
    axes[0].set_ylabel('Predicted 4-Clique Count')
    axes[0].set_title(f'{model_name} - Original Graph')
    
    # Plot for complement graph clique counts
    axes[1].scatter(actual_values[:, 1], predictions[:, 1], alpha=0.5)
    axes[1].plot([min(actual_values[:, 1]), max(actual_values[:, 1])],
                [min(actual_values[:, 1]), max(actual_values[:, 1])], 'r--')
    axes[1].set_xlabel('Actual 4-Clique Count (Complement)')
    axes[1].set_ylabel('Predicted 4-Clique Count')
    axes[1].set_title(f'{model_name} - Complement Graph')
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_predictions.png")
    plt.close() 