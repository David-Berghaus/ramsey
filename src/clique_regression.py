import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import networkx as nx
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from score import get_score_and_cliques
from env import obs_space_to_graph

class GraphDataset(Dataset):
    """
    Dataset for random graph samples with their clique counts
    """
    def __init__(self, n, r, b, num_samples=10000):
        self.n = n
        self.r = r
        self.b = b
        self.num_samples = num_samples
        self.num_entries = n * (n - 1) // 2
        
        # Generate random graphs and their clique counts
        self.data = []
        for _ in tqdm(range(num_samples), desc="Generating random graphs"):
            # Generate random adjacency matrix
            adj_vector = np.random.randint(2, size=self.num_entries)
            
            # Convert to graph and get clique counts
            G = obs_space_to_graph(adj_vector, self.n)
            _, cliques_r, cliques_b, is_connected = get_score_and_cliques(
                G, self.r, self.b, -1000
            )
            
            # Count cliques
            r_count = sum(1 for clique in cliques_r if len(clique) >= r)
            b_count = sum(1 for clique in cliques_b if len(clique) >= b)
            
            # Store graph and clique counts if the graph is connected
            if is_connected:
                self.data.append((adj_vector, r_count, b_count))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        adj_vector, r_count, b_count = self.data[idx]
        return torch.FloatTensor(adj_vector), torch.FloatTensor([r_count, b_count])

class SimpleGraphEncoder(nn.Module):
    """
    A simpler graph encoder that uses the same architecture principles as the RL model
    but without the complex clique attention mechanisms
    """
    def __init__(self, n, input_dim, hidden_dim=64, output_dim=64):
        super(SimpleGraphEncoder, self).__init__()
        self.n = n
        
        # Node level encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Graph level encoder
        self.graph_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, adj_vector):
        # Convert adjacency vector to matrix
        batch_size = adj_vector.shape[0]
        adj_matrices = []
        
        for b in range(batch_size):
            # Convert flat vector to adjacency matrix
            adj_matrix = torch.zeros((self.n, self.n), device=adj_vector.device)
            idx = 0
            for i in range(self.n):
                for j in range(i+1, self.n):
                    adj_matrix[i, j] = adj_matrix[j, i] = adj_vector[b, idx]
                    idx += 1
            adj_matrices.append(adj_matrix)
        
        adj_matrices = torch.stack(adj_matrices)
        
        # Node features - use degree as initial feature
        node_features = []
        for b in range(batch_size):
            degrees = adj_matrices[b].sum(dim=1, keepdim=True)
            # Add extra features if needed
            node_features.append(degrees)
        
        node_features = torch.stack(node_features)
        
        # Process node features
        node_embeddings = self.node_encoder(node_features)
        
        # Global pooling
        graph_embedding = torch.mean(node_embeddings, dim=1)
        
        # Final graph embedding
        output = self.graph_encoder(graph_embedding)
        
        return output

class CliqueRegressionModel(nn.Module):
    """
    Model for predicting clique counts using a simpler architecture inspired by the RL model
    """
    def __init__(self, n, r, b, features_dim=64):
        super(CliqueRegressionModel, self).__init__()
        self.n = n
        self.r = r
        self.b = b
        self.num_entries = n * (n - 1) // 2
        
        # Graph encoder for original graph
        self.graph_encoder = SimpleGraphEncoder(n=n, input_dim=1, hidden_dim=64, output_dim=features_dim//2)
        
        # Regression head to predict clique counts
        self.regression_head = nn.Sequential(
            nn.Linear(features_dim//2, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Output r and b counts
        )
        
    def forward(self, x):
        # Extract features using the graph encoder
        features = self.graph_encoder(x)
        
        # Predict clique counts
        predictions = self.regression_head(features)
        
        return predictions

def train_regression_model(n=17, r=4, b=4, batch_size=32, num_epochs=100, 
                          learning_rate=1e-4, num_samples=5000):
    """
    Train a regression model to predict clique counts
    """
    # Set up directories for saving models and results
    time_stamp = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    base_dir = "data/regression"
    base_path = os.path.join(base_dir, f"{n}_{r}_{b}", time_stamp)
    os.makedirs(base_path, exist_ok=True)
    
    # Create dataset and dataloader
    dataset = GraphDataset(n=n, r=r, b=b, num_samples=num_samples)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CliqueRegressionModel(n=n, r=r, b=b)
    model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                epoch_val_loss += loss.item()
        
        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss}, Val Loss: {epoch_val_loss}")
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(base_path, f"model_epoch_{epoch+1}.pt"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(base_path, "model_final.pt"))
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title(f'Training and Validation Loss for Clique Regression (n={n}, r={r}, b={b})')
    plt.legend()
    plt.savefig(os.path.join(base_path, 'loss_curve.png'))
    
    return model, train_losses, val_losses

def evaluate_model(model, n=17, r=4, b=4, num_samples=100):
    """
    Evaluate the regression model on random samples
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    num_entries = n * (n - 1) // 2
    mse_loss = 0
    actual_rs = []
    pred_rs = []
    actual_bs = []
    pred_bs = []
    
    for _ in tqdm(range(num_samples), desc="Evaluating model"):
        # Generate random adjacency matrix
        adj_vector = np.random.randint(2, size=num_entries)
        
        # Convert to graph and get clique counts
        G = obs_space_to_graph(adj_vector, n)
        _, cliques_r, cliques_b, is_connected = get_score_and_cliques(
            G, r, b, -1000
        )
        
        if not is_connected:
            continue
            
        # Count cliques
        r_count = sum(1 for clique in cliques_r if len(clique) >= r)
        b_count = sum(1 for clique in cliques_b if len(clique) >= b)
        
        # Predict using the model
        with torch.no_grad():
            tensor_adj = torch.FloatTensor(adj_vector).unsqueeze(0).to(device)
            prediction = model(tensor_adj).cpu().numpy()[0]
        
        # Calculate MSE
        actual = np.array([r_count, b_count])
        mse = np.mean((prediction - actual) ** 2)
        mse_loss += mse
        
        # Store for plotting
        actual_rs.append(r_count)
        pred_rs.append(prediction[0])
        actual_bs.append(b_count)
        pred_bs.append(prediction[1])
    
    mse_loss /= num_samples
    
    # Create scatter plots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(actual_rs, pred_rs, alpha=0.5)
    plt.plot([min(actual_rs), max(actual_rs)], [min(actual_rs), max(actual_rs)], 'r--')
    plt.xlabel('Actual r-clique count')
    plt.ylabel('Predicted r-clique count')
    plt.title(f'r-clique Prediction (MSE: {mse_loss:.4f})')
    
    plt.subplot(1, 2, 2)
    plt.scatter(actual_bs, pred_bs, alpha=0.5)
    plt.plot([min(actual_bs), max(actual_bs)], [min(actual_bs), max(actual_bs)], 'r--')
    plt.xlabel('Actual b-clique count')
    plt.ylabel('Predicted b-clique count')
    plt.title('b-clique Prediction')
    
    plt.tight_layout()
    
    return mse_loss, (actual_rs, pred_rs, actual_bs, pred_bs)

if __name__ == "__main__":
    # Train the regression model
    trained_model, train_losses, val_losses = train_regression_model(
        n=17, 
        r=4, 
        b=4,
        batch_size=32,
        num_epochs=50,
        learning_rate=1e-4,
        num_samples=5000
    )
    
    # Evaluate the model
    mse, (actual_rs, pred_rs, actual_bs, pred_bs) = evaluate_model(
        trained_model,
        n=17,
        r=4,
        b=4,
        num_samples=100
    )
    
    print(f"Final MSE: {mse}") 