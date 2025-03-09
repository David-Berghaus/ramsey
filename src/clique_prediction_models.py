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
from graphs_cache import GraphsCache

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

# Main function to run the experiment
def run_clique_prediction_experiment(n_samples=5000, batch_size=32, epochs=20, device='cpu'):
    """
    Run the full experiment comparing both models.
    
    Args:
        n_samples (int): Number of graph samples to generate
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        device (str): Device to run on ('cpu' or 'cuda')
        
    Returns:
        dict: Results dictionary with model performances
    """
    print("Generating dataset...")
    adjacency_matrices, clique_counts = generate_clique_dataset(n_samples=n_samples)
    
    # Split into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        adjacency_matrices, clique_counts, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )
    
    # Create datasets
    train_dataset = GraphCliqueDataset(X_train, y_train)
    val_dataset = GraphCliqueDataset(X_val, y_val)
    test_dataset = GraphCliqueDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create and train the custom model
    print("\nTraining Custom Model...")
    custom_model = CustomCliquePredictor(n_vertices=17, features_dim=256)
    custom_model, custom_train_losses, custom_val_losses = train_model(
        custom_model, train_loader, val_loader, epochs=epochs, 
        device=device, model_name="custom_model"
    )
    
    # Create and train the MLP model
    print("\nTraining MLP Model...")
    mlp_model = MLPCliquePredictor(n_vertices=17)
    mlp_model, mlp_train_losses, mlp_val_losses = train_model(
        mlp_model, train_loader, val_loader, epochs=epochs,
        device=device, model_name="mlp_model"
    )
    
    # Evaluate both models
    print("\nEvaluating Custom Model...")
    custom_mse, custom_mae, custom_preds, custom_actual = evaluate_model(
        custom_model, test_loader, device=device
    )
    
    print("\nEvaluating MLP Model...")
    mlp_mse, mlp_mae, mlp_preds, mlp_actual = evaluate_model(
        mlp_model, test_loader, device=device
    )
    
    # Visualize results
    visualize_results(custom_preds, custom_actual, model_name="Custom_Model")
    visualize_results(mlp_preds, mlp_actual, model_name="MLP_Model")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(custom_train_losses, label='Custom Train')
    plt.plot(custom_val_losses, label='Custom Val')
    plt.plot(mlp_train_losses, label='MLP Train')
    plt.plot(mlp_val_losses, label='MLP Val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.close()
    
    # Report results
    results = {
        'Custom Model': {
            'MSE': custom_mse,
            'MAE': custom_mae,
            'Training Time (epochs)': epochs
        },
        'MLP Model': {
            'MSE': mlp_mse,
            'MAE': mlp_mae,
            'Training Time (epochs)': epochs
        }
    }
    
    print("\n=== Results ===")
    print(f"Custom Model - Test MSE: {custom_mse:.4f}, Test MAE: {custom_mae:.4f}")
    print(f"MLP Model - Test MSE: {mlp_mse:.4f}, Test MAE: {mlp_mae:.4f}")
    
    return results

if __name__ == "__main__":
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run the experiment
    results = run_clique_prediction_experiment(
        n_samples=5000,
        batch_size=32, 
        epochs=20,
        device=device
    ) 