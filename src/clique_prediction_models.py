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
import os

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
                 clique_attention_context_len=16, node_attention_context_len=8, num_heads=2):
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
        self.num_heads = num_heads
        
        # IMPROVED GNN Node Feature Extraction
        # Initial node feature dimensions
        node_feat_dim = 16
        
        # GNN layers
        self.node_embedding = nn.Linear(1, node_feat_dim)  # Initial node feature embedding
        
        # Message passing layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = node_feat_dim if i == 0 else hidden_dim
            self.gnn_layers.append(nn.Linear(layer_input_dim * 2, hidden_dim))  # Message function
        
        # Size-aware clique embeddings
        self.clique_size_embedder_r = nn.Linear(1, hidden_dim)
        self.clique_size_embedder_b = nn.Linear(1, hidden_dim)
            
        # Multi-head attention with layer normalization
        self.r_attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.b_attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        
        self.r_layer_norm1 = nn.LayerNorm(hidden_dim)
        self.r_layer_norm2 = nn.LayerNorm(hidden_dim)
        self.b_layer_norm1 = nn.LayerNorm(hidden_dim)
        self.b_layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward networks for attention
        self.r_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.b_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Query, Key, Value projections
        self.r_query = nn.Linear(hidden_dim, hidden_dim)
        self.r_key = nn.Linear(hidden_dim, hidden_dim)
        self.r_value = nn.Linear(hidden_dim, hidden_dim)
        
        self.b_query = nn.Linear(hidden_dim, hidden_dim)
        self.b_key = nn.Linear(hidden_dim, hidden_dim)
        self.b_value = nn.Linear(hidden_dim, hidden_dim)
        
        # Downward projections for clique attention
        self.r_downward = nn.Linear(clique_attention_context_len * hidden_dim, hidden_dim)
        self.b_downward = nn.Linear(clique_attention_context_len * hidden_dim, hidden_dim)
        
        # Node-clique cross-attention
        self.node_to_clique_r = nn.Linear(hidden_dim, hidden_dim)
        self.node_to_clique_b = nn.Linear(hidden_dim, hidden_dim)
        
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
        
        # Pre-compute indices for faster adjacency matrix building
        self.row_indices, self.col_indices = [], []
        idx = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.row_indices.append(i)
                self.row_indices.append(j)
                self.col_indices.append(j)
                self.col_indices.append(i)
                idx += 1
        self.row_indices = torch.tensor(self.row_indices)
        self.col_indices = torch.tensor(self.col_indices)
        
        # Pre-allocate tensors for padding attention mechanism
        self.zero_hidden = torch.zeros(self.hidden_dim)
        self.expected_attn_dim = self.clique_attention_context_len * self.hidden_dim
    
    def extract_node_features(self, adj_matrix):
        """
        Extract GNN node features from the adjacency matrix using proper normalized graph convolution.
        
        Args:
            adj_matrix (torch.Tensor): Adjacency matrix [n, n]
            
        Returns:
            torch.Tensor: Node features tensor [n, hidden_dim]
        """
        # Add self-loops
        adj_tilde = adj_matrix + torch.eye(self.n, device=adj_matrix.device)
        
        # Compute degree matrix
        degree = torch.sum(adj_tilde, dim=1)
        degree_sqrt_inv = torch.diag(1.0 / torch.sqrt(degree + 1e-8))
        
        # Normalize adjacency matrix
        normalized_adj = torch.matmul(degree_sqrt_inv, adj_tilde)
        normalized_adj = torch.matmul(normalized_adj, degree_sqrt_inv)
        
        # Initial node features (just a single feature of 1.0 for each node)
        node_features = torch.ones(self.n, 1, device=adj_matrix.device)
        node_features = self.node_embedding(node_features)  # [n, node_feat_dim]
        
        # GNN message passing with normalized adjacency
        for i, layer in enumerate(self.gnn_layers):
            # Create a matrix of neighbor messages using normalized adjacency
            neighbor_sums = torch.matmul(normalized_adj, node_features)
            
            # Create a matrix where each row contains [node_feature, neighbor_sum]
            combined_features = torch.cat([node_features, neighbor_sums], dim=1)
            
            # Apply the layer
            node_features = self.act(layer(combined_features))
        
        return node_features
    
    def build_adjacency_matrix_batch(self, graph_vec_batch):
        """
        Efficiently builds adjacency matrices for a batch of graph vectors using sparse operations.
        
        Args:
            graph_vec_batch (torch.Tensor): Batch of graph vectors [batch_size, n_entries]
            
        Returns:
            list: List of adjacency matrices as tensors
        """
        batch_size = graph_vec_batch.shape[0]
        adj_matrices = []
        
        for b in range(batch_size):
            graph_vec = graph_vec_batch[b]
            values = torch.cat([graph_vec, graph_vec])  # Double the values for symmetric entries
            indices = torch.stack([
                self.row_indices.to(graph_vec.device), 
                self.col_indices.to(graph_vec.device)
            ])
            
            # Create sparse tensor and convert to dense
            adj_matrix = torch.sparse.FloatTensor(
                indices, values, (self.n, self.n)
            ).to_dense()
            
            adj_matrices.append(adj_matrix)
            
        return adj_matrices
    
    def node_clique_cross_attention(self, node_embeddings, cliques):
        """
        Apply node-clique cross-attention to enhance node embeddings with clique information.
        
        Args:
            node_embeddings (torch.Tensor): Node embeddings [n, hidden_dim]
            cliques (list): List of cliques
            
        Returns:
            torch.Tensor: Enhanced node embeddings [n, hidden_dim]
        """
        n_nodes = node_embeddings.shape[0]
        n_cliques = min(len(cliques), self.clique_attention_context_len)
        
        if n_cliques == 0:
            return node_embeddings
            
        membership = torch.zeros(n_nodes, n_cliques, device=node_embeddings.device)
        
        # Fill membership matrix
        for c_idx, clique in enumerate(cliques[:n_cliques]):
            for n_idx in clique:
                if n_idx < n_nodes:
                    membership[n_idx, c_idx] = 1.0
        
        # Use membership for attention
        attention_weights = torch.softmax(membership, dim=1)
        
        # Compute clique representations
        clique_reps = []
        for clique in cliques[:n_cliques]:
            clique_nodes = list(clique)
            clique_rep = node_embeddings[clique_nodes].mean(dim=0)
            clique_reps.append(clique_rep)
        
        if not clique_reps:
            return node_embeddings
            
        clique_matrix = torch.stack(clique_reps)
        enhanced_node_embeddings = node_embeddings + torch.matmul(attention_weights, clique_matrix)
        return enhanced_node_embeddings
    
    def process_cliques(self, cliques, node_embeddings, max_size, query, key, value, downward, device):
        """
        Process cliques using node embeddings for clique representation and multi-head attention
        
        Args:
            cliques (list): List of cliques
            node_embeddings (torch.Tensor): Node embeddings [n, hidden_dim]
            max_size (int): Maximum clique size (r or b)
            query, key, value (nn.Module): Attention components
            downward (nn.Module): Downward projection
            device (torch.device): Device to use
            
        Returns:
            torch.Tensor: Clique embedding
        """
        # Get valid cliques
        valid_cliques = [clique for clique in cliques if len(clique) >= max_size]
        
        if not valid_cliques:
            return self.zero_hidden.to(device)
        
        # Limit to context length
        valid_cliques = valid_cliques[:self.clique_attention_context_len]
        
        # Use node embeddings for clique representation
        clique_vectors = []
        clique_sizes = []
        
        for clique in valid_cliques:
            nodes_in_clique = list(clique)[:max_size]
            # Extract node embeddings for clique nodes and mean pool
            clique_embedding = node_embeddings[nodes_in_clique].mean(dim=0)
            clique_vectors.append(clique_embedding)
            clique_sizes.append(len(clique))
        
        clique_stack = torch.stack(clique_vectors)
        
        # Add size information to embeddings
        if max_size == self.r:
            size_embedder = self.clique_size_embedder_r
        else:
            size_embedder = self.clique_size_embedder_b
            
        clique_size_tensor = torch.tensor(clique_sizes, dtype=torch.float32, device=device).view(-1, 1)
        size_embeddings = size_embedder(clique_size_tensor)
        
        # Combine with node-based embedding
        clique_stack = clique_stack + size_embeddings
        
        # Apply multi-head attention
        if max_size == self.r:
            attention = self.r_attention
            layer_norm1 = self.r_layer_norm1
            layer_norm2 = self.r_layer_norm2
            nn_layer = self.r_nn
        else:
            attention = self.b_attention
            layer_norm1 = self.b_layer_norm1
            layer_norm2 = self.b_layer_norm2
            nn_layer = self.b_nn
            
        # Transform for attention
        Q = query(clique_stack).unsqueeze(0)  # Add batch dimension
        K = key(clique_stack).unsqueeze(0)
        V = value(clique_stack).unsqueeze(0)
        
        # Store original for residual connection
        original_embeddings = clique_stack.unsqueeze(0)
        
        # Apply attention mechanism
        attention_output, _ = attention(Q, K, V)
        
        # Apply layer normalization and residual connection
        attention_output = layer_norm1(attention_output) + original_embeddings
        
        # Apply a feed forward network
        attention_output_before_ff = attention_output
        attention_output = nn_layer(attention_output)
        
        # Apply layer normalization and residual connection
        attention_output = layer_norm2(attention_output) + attention_output_before_ff
        
        # Remove batch dimension and flatten
        attention_output = attention_output.squeeze(0)
        
        # Prepare for downward projection
        num_cliques = attention_output.size(0)
        
        # Reshape and handle different dimensions
        if num_cliques * self.hidden_dim == self.expected_attn_dim:
            # Perfect fit
            attn_flat = attention_output.reshape(-1)
        elif num_cliques * self.hidden_dim < self.expected_attn_dim:
            # Need padding
            attn_flat = attention_output.reshape(-1)
            padding = torch.zeros(
                self.expected_attn_dim - attn_flat.size(0), 
                device=device
            )
            attn_flat = torch.cat([attn_flat, padding])
        else:
            # Need truncation
            attn_flat = attention_output[:self.clique_attention_context_len].reshape(-1)
            if attn_flat.size(0) > self.expected_attn_dim:
                attn_flat = attn_flat[:self.expected_attn_dim]
        
        # Apply downward projection
        graph_embedding = downward(attn_flat)
        
        # Replace NaN values with zeros
        graph_embedding[torch.isnan(graph_embedding)] = 0
        
        return graph_embedding
    
    def forward(self, x):
        """
        Forward pass of the enhanced GNN with Clique Attention model.
        
        Args:
            x (torch.Tensor): Batch of flattened adjacency matrices [batch_size, n_entries]
            
        Returns:
            torch.Tensor: Predicted clique counts [batch_size, 2]
        """
        # Build adjacency matrices
        adj_matrices = self.build_adjacency_matrix_batch(x)
        
        batch_size = x.shape[0]
        device = x.device
        
        result = []
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
            
            b_graph_embed = self.process_cliques(
                cliques_b, enhanced_node_embeddings, self.b,
                self.b_query, self.b_key, self.b_value, 
                self.b_downward, device
            )
            
            # Create graph embedding using mean pooling for permutation invariance
            graph_embedding = enhanced_node_embeddings.mean(dim=0)
            
            r_embedding = graph_embedding + r_clique_embed
            b_embedding = graph_embedding + b_clique_embed
            
            # Predict clique counts
            r_count = self.r_head(r_embedding)
            b_count = self.b_head(b_embedding)
            
            # Combine predictions
            result.append(torch.cat([r_count, b_count], dim=0))
        
        return torch.stack(result)

# Function to train a model
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, device='cpu', model_name="model",
               patience=5, min_delta=0.001, overfitting_threshold=3, output_dir=None):
    """
    Train the model with early stopping.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Maximum number of epochs to train for
        lr: Learning rate
        device: Device to train on (cpu/cuda)
        model_name: Name of the model for saving
        patience: Number of epochs with no improvement after which training will be stopped
        min_delta: Minimum change in validation loss to qualify as improvement
        overfitting_threshold: Number of consecutive epochs showing overfitting before stopping
        output_dir: Directory to save model weights and plots. If None, saves in the current directory.
    
    Returns:
        model: Trained model
        training_losses: List of training losses
        validation_losses: List of validation losses
        stopped_early: Boolean indicating if training stopped early
        reason: Reason for stopping
    """
    # Create weights directory if output_dir is provided
    weights_path = f"{model_name}_best.pt"
    if output_dir is not None:
        weights_dir = os.path.join(output_dir, "model_weights")
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, f"{model_name}_best.pt")
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    training_losses = []
    validation_losses = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    # Early stopping variables
    epochs_no_improve = 0
    early_stopped = False
    stopped_reason = "Completed all epochs"
    
    # Overfitting detection variables
    consecutive_overfitting = 0
    
    # Convergence detection variables
    val_loss_change_history = []
    convergence_window = 5  # Number of epochs to consider for convergence
    convergence_threshold = 0.0005  # Maximum relative change in validation loss to consider converged
    
    print(f"\nStarting training with adaptive termination:")
    print(f"- Max epochs: {epochs}")
    print(f"- Early stopping patience: {patience}")
    print(f"- Overfitting detection threshold: {overfitting_threshold} consecutive epochs")
    print(f"- Convergence window: {convergence_window} epochs with less than {convergence_threshold*100:.4f}% change")
    
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
        
        # Calculate relative change in validation loss
        if epoch > 0:
            val_loss_change = abs((validation_losses[epoch] - validation_losses[epoch-1]) / validation_losses[epoch-1])
            val_loss_change_history.append(val_loss_change)
        
        # Check for improvement
        if val_loss < best_val_loss - min_delta:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} (improved)")
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            torch.save(model.state_dict(), weights_path)
            epochs_no_improve = 0
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} (no improvement)")
            epochs_no_improve += 1
        
        # Check for overfitting: training loss decreases while validation loss increases
        if epoch > 0:
            train_improved = training_losses[epoch] < training_losses[epoch-1]
            val_worsened = validation_losses[epoch] > validation_losses[epoch-1]
            
            if train_improved and val_worsened:
                consecutive_overfitting += 1
                print(f"Warning: Possible overfitting detected ({consecutive_overfitting}/{overfitting_threshold})")
            else:
                consecutive_overfitting = 0
        
        # Check for convergence
        if len(val_loss_change_history) >= convergence_window:
            recent_changes = val_loss_change_history[-convergence_window:]
            if all(change < convergence_threshold for change in recent_changes):
                print(f"Convergence detected: Validation loss has stabilized over {convergence_window} epochs")
                early_stopped = True
                stopped_reason = f"Model converged (validation loss stabilized over {convergence_window} epochs)"
                break
        
        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs due to no improvement for {patience} epochs")
            early_stopped = True
            stopped_reason = f"No improvement for {patience} epochs"
            break
            
        # Overfitting check
        if consecutive_overfitting >= overfitting_threshold:
            print(f"Training stopped after {epoch+1} epochs due to overfitting detection")
            early_stopped = True
            stopped_reason = f"Overfitting detected for {overfitting_threshold} consecutive epochs"
            break
    
    # Print final training status
    if early_stopped:
        print(f"\nTraining terminated early: {stopped_reason}")
        print(f"Completed {epoch+1}/{epochs} epochs")
    else:
        print(f"\nCompleted all {epochs} epochs")
    
    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.6f}")
    
    return model, training_losses, validation_losses, early_stopped, stopped_reason

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
            
            # Append batch predictions and actuals
            batch_predictions = outputs.cpu().numpy()
            batch_actuals = max_clique_sizes.cpu().numpy()
            
            all_predictions.append(batch_predictions)
            all_actual.append(batch_actuals)
    
    avg_mse = total_mse / len(test_loader)
    avg_mae = total_mae / len(test_loader)
    
    # Concatenate all batches into a single array
    all_predictions = np.vstack(all_predictions)
    all_actual = np.vstack(all_actual)
    
    print(f"Evaluation complete:")
    print(f"  MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}")
    print(f"  Predictions shape: {all_predictions.shape}")
    print(f"  Actual values shape: {all_actual.shape}")
    
    return avg_mse, avg_mae, all_predictions, all_actual

# Function to visualize results
def visualize_results(predictions, actual_values, model_name="Model", output_dir=None):
    """
    Visualize the prediction results.
    
    Args:
        predictions (np.array): Predicted clique counts
        actual_values (np.array): Actual clique counts
        model_name (str): Name of the model for plot titles
        output_dir (str): Directory to save the plot. If None, saves in the current directory or 'plots' folder.
    """
    try:
        # Convert inputs to numpy arrays if they aren't already
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        
        # Print debug information
        print(f"Visualizing results for {model_name}:")
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Actual values shape: {actual_values.shape}")
        
        # Ensure both arrays have the right shape
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 2)
        if len(actual_values.shape) == 1:
            actual_values = actual_values.reshape(-1, 2)
        
        # Set larger font sizes for paper-quality plots
        plt.rcParams.update({
            'font.size': 14,             # Default font size
            'axes.titlesize': 18,        # Title font size
            'axes.labelsize': 16,        # Axis label font size
            'xtick.labelsize': 14,       # X-tick label font size
            'ytick.labelsize': 14,       # Y-tick label font size
            'legend.fontsize': 14,       # Legend font size
            'figure.titlesize': 20,      # Figure title font size
        })
            
        # Create figure and axes
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))  # Slightly increased figure size
        
        # Plot for original graph clique counts
        axes[0].scatter(actual_values[:, 0], predictions[:, 0], alpha=0.5, s=60)  # Increased marker size
        
        # Compute the ranges with some safety checks
        x_min = np.min(actual_values[:, 0])
        x_max = np.max(actual_values[:, 0])
        
        # Add diagonal line
        if x_min != x_max:  # Only plot line if there's a range
            axes[0].plot([x_min, x_max], [x_min, x_max], 'r--', linewidth=2)  # Increased line width
        
        # Set buffer space around points
        x_range = x_max - x_min
        if x_range == 0:
            x_range = 1.0  # Default range if all values are the same
            
        axes[0].set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
        
        # Set y-axis limits with safety check for identical values
        y_min = np.min(predictions[:, 0])
        y_max = np.max(predictions[:, 0])
        y_range = y_max - y_min
        if y_range == 0:
            y_range = 1.0  # Default range if all values are the same
        
        axes[0].set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        
        axes[0].set_xlabel('Actual 4-Clique Count (Original)', fontsize=16, fontweight='bold')
        axes[0].set_ylabel('Predicted 4-Clique Count', fontsize=16, fontweight='bold')
        axes[0].set_title(f'{model_name} - Original Graph', fontsize=18, fontweight='bold')
        axes[0].grid(True, linestyle='--', alpha=0.7)
        
        # Make tick labels larger and more visible
        axes[0].tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
        
        # Plot for complement graph clique counts
        axes[1].scatter(actual_values[:, 1], predictions[:, 1], alpha=0.5, s=60)  # Increased marker size
        
        # Compute the ranges with some safety checks
        x_min = np.min(actual_values[:, 1])
        x_max = np.max(actual_values[:, 1])
        
        # Add diagonal line
        if x_min != x_max:  # Only plot line if there's a range
            axes[1].plot([x_min, x_max], [x_min, x_max], 'r--', linewidth=2)  # Increased line width
            
        # Set buffer space around points
        x_range = x_max - x_min
        if x_range == 0:
            x_range = 1.0  # Default range if all values are the same
            
        axes[1].set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
        
        # Set y-axis limits with safety check for identical values
        y_min = np.min(predictions[:, 1])
        y_max = np.max(predictions[:, 1])
        y_range = y_max - y_min
        if y_range == 0:
            y_range = 1.0  # Default range if all values are the same
            
        axes[1].set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        
        axes[1].set_xlabel('Actual 4-Clique Count (Complement)', fontsize=16, fontweight='bold')
        axes[1].set_ylabel('Predicted 4-Clique Count', fontsize=16, fontweight='bold')
        axes[1].set_title(f'{model_name} - Complement Graph', fontsize=18, fontweight='bold')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        
        # Make tick labels larger and more visible
        axes[1].tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
        
        # Adjust layout to make room for larger fonts
        plt.tight_layout(pad=3.0)
        
        # Determine save path
        if output_dir is not None:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{model_name}_predictions.pdf")
        else:
            # Ensure the plots directory exists
            if not os.path.exists('plots'):
                os.makedirs('plots')
            save_path = os.path.join('plots', f"{model_name}_predictions.pdf")
            
        print(f"  Saving plot to: {os.path.abspath(save_path)}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Increased DPI for higher quality
        
        # Return fig for optional display in Jupyter notebooks or further modification
        return fig
    
    except Exception as e:
        print(f"Error in visualize_results: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Create a simple error figure with larger text
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error generating plot: {str(e)}", 
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=16)  # Larger error message font
        plt.axis('off')
        
        # Save error plot to correct location
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            error_path = os.path.join(output_dir, f"{model_name}_error.pdf")
        else:
            error_path = f"{model_name}_error.pdf"
            
        plt.savefig(error_path, dpi=300)  # Higher DPI for error plot too
        plt.close()
        return None
    finally:
        plt.close() 