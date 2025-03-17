import argparse
import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import time
from datetime import datetime
import networkx as nx
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import shutil
from sklearn.model_selection import train_test_split

from clique_prediction_models import (
    MLPCliquePredictor,
    RamseyGraphGNNWithCliqueAttention,
    generate_clique_dataset,
    GraphCliqueDataset,
    train_model,
    evaluate_model,
    visualize_results
)

def parse_args():
    parser = argparse.ArgumentParser(description='Clique Prediction in Ramsey Graphs')
    
    # Dataset parameters
    parser.add_argument('--n_samples', type=int, default=5000, 
                        help='Number of graph samples to generate')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for training')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension for neural network layers')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of layers in the GNN')
    parser.add_argument('--clique_attention_context', type=int, default=16, 
                        help='Context length for clique attention mechanism')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs to wait for improvement before early stopping')
    parser.add_argument('--min_delta', type=float, default=0.001,
                        help='Minimum change in validation loss to be considered as improvement')
    parser.add_argument('--overfitting_threshold', type=int, default=3,
                        help='Number of consecutive epochs with improving train loss but worsening val loss')
    
    # Execution options
    parser.add_argument('--no_gpu', action='store_true',
                        help='Disable GPU usage even if available')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    # Model selection flags
    parser.add_argument('--mlp_only', action='store_true',
                        help='Train only the MLP model')
    parser.add_argument('--ramsey_gnn_only', action='store_true',
                        help='Train only the Ramsey GNN with Clique Attention model')
    
    args = parser.parse_args()
    
    # Check for conflicting model flags
    model_flags = [args.mlp_only, args.ramsey_gnn_only]
    if sum(model_flags) > 1:
        parser.error("Only one model flag can be specified at a time")
    
    return args

def create_output_dir(base_output_dir, model_name=None):
    """Create the output directory structure if it doesn't exist"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if model_name:
        # Create model-specific directory with timestamp
        output_dir = os.path.join(base_output_dir, model_name, timestamp)
    else:
        # Create a general directory with timestamp only for comparative runs
        output_dir = os.path.join(base_output_dir, f"comparison_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for different types of results
    os.makedirs(os.path.join(output_dir, "model_weights"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tensorboard"), exist_ok=True)
    
    return output_dir

def measure_prediction_time(model, test_data, device, n_runs=100):
    """
    Measure the average prediction time for a model.
    
    Args:
        model (nn.Module): The model to evaluate
        test_data (torch.Tensor): Test data
        device (str): Device to run on
        n_runs (int): Number of runs for averaging
        
    Returns:
        float: Average prediction time in milliseconds
    """
    model = model.to(device)
    model.eval()
    
    # Warm-up runs
    for _ in range(10):
        with torch.no_grad():
            _ = model(test_data)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = model(test_data)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return np.mean(times), np.std(times)

def analyze_performance_on_ramsey_graphs(models, n_vertices=17):
    """
    Analyze model performance on specific graphs that are close to the Ramsey number R(4,4).
    
    Args:
        models (dict): Dictionary of models to evaluate
        n_vertices (int): Number of vertices
        
    Returns:
        dict: Results dictionary
    """
    # Import necessary functions
    from env import obs_space_to_graph
    
    results = {}
    
    # Create some special test cases
    # 1. A graph with a 4-clique
    n_entries = n_vertices * (n_vertices - 1) // 2
    g1 = np.zeros(n_entries)
    
    # Add a 4-clique (nodes 0, 1, 2, 3) by setting edges between them
    edge_idx = 0
    for i in range(n_vertices):
        for j in range(i+1, n_vertices):
            if i < 4 and j < 4:  # Connect vertices 0,1,2,3 to form a 4-clique
                g1[edge_idx] = 1
            edge_idx += 1
    
    # 2. A balanced graph (close to a Ramsey graph)
    # This is a simplified approximation - real R(4,4) graphs are complex
    np.random.seed(42)
    g2 = np.random.randint(2, size=n_entries)
    
    test_graphs = [g1, g2]
    test_names = ["4-Clique Graph", "Random Graph"]
    
    # Calculate actual clique sizes
    actual_sizes = []
    for g in test_graphs:
        G = obs_space_to_graph(g, n_vertices)
        G_comp = nx.complement(G)
        
        max_clique_original = max([len(c) for c in nx.find_cliques(G)], default=0)
        max_clique_complement = max([len(c) for c in nx.find_cliques(G_comp)], default=0)
        
        actual_sizes.append((max_clique_original, max_clique_complement))
    
    # Evaluate each model
    for model_name, model in models.items():
        model_results = []
        
        for i, g in enumerate(test_graphs):
            test_graph = torch.tensor(g, dtype=torch.float32).unsqueeze(0).to(model.device)
            
            with torch.no_grad():
                prediction = model(test_graph).cpu().numpy()[0]
            
            actual = actual_sizes[i]
            error = np.abs(prediction - actual)
            
            model_results.append({
                "graph_name": test_names[i],
                "actual": actual,
                "predicted": prediction.tolist(),
                "absolute_error": error.tolist()
            })
        
        results[model_name] = model_results
    
    return results

def save_training_results(model, model_name, output_dir, train_losses, val_losses, 
                         mse, mae, predictions, actual, args, early_stopped=False, stopped_reason=""):
    """
    Save all training results, including model weights, metrics, and visualizations.
    
    Args:
        model (nn.Module): The trained model
        model_name (str): Name of the model
        output_dir (str): Output directory
        train_losses (list): Training losses
        val_losses (list): Validation losses
        mse (float): Mean Squared Error on test set
        mae (float): Mean Absolute Error on test set
        predictions (numpy.ndarray): Model predictions
        actual (numpy.ndarray): Actual values
        args (argparse.Namespace): Command line arguments
        early_stopped (bool): Whether training was terminated early
        stopped_reason (str): Reason for early stopping if applicable
    """
    print(f"Saving results for {model_name}...")
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(output_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Save model weights
    weights_path = os.path.join(output_dir, "model_weights", f"{model_name}_weights.pt")
    torch.save(model.state_dict(), weights_path)
    
    # Create a copy of the best model for easy loading
    best_model_path = os.path.join(output_dir, "model_weights", f"{model_name}_best.pt")
    if os.path.exists(f"{model_name}_best.pt"):
        shutil.move(f"{model_name}_best.pt", best_model_path)
        print(f"Moved best model to: {best_model_path}")
    
    # Save metrics to JSON
    metrics = {
        "test_mse": float(mse),
        "test_mae": float(mae),
        "epochs_trained": len(train_losses),
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
        "best_val_loss": float(min(val_losses)),
        "best_epoch": int(np.argmin(val_losses) + 1),
        "early_stopped": early_stopped,
        "stopped_reason": stopped_reason,
        "training_parameters": {
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "patience": args.patience if hasattr(args, 'patience') else 5,
            "min_delta": args.min_delta if hasattr(args, 'min_delta') else 0.001,
            "overfitting_threshold": args.overfitting_threshold if hasattr(args, 'overfitting_threshold') else 3
        }
    }
    
    with open(os.path.join(output_dir, "metrics", f"{model_name}_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save losses to JSON
    losses = {
        "train_losses": [float(loss) for loss in train_losses],
        "val_losses": [float(loss) for loss in val_losses]
    }
    
    with open(os.path.join(output_dir, "metrics", f"{model_name}_losses.json"), 'w') as f:
        json.dump(losses, f, indent=4)
    
    # Create and save plots
    # Training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.axvline(x=metrics["best_epoch"]-1, color='g', linestyle='--', 
               label=f'Best Model (Epoch {metrics["best_epoch"]})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'{model_name} Training Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{model_name}_training_curve.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create visualization of predictions vs. actual values
    print(f"Predictions shape: {predictions.shape}, Actual values shape: {actual.shape}")
    
    # Convert to numpy arrays if they aren't already
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().numpy()
    
    # Ensure shapes are correct (n_samples, 2)
    if len(predictions.shape) == 1 and len(predictions) % 2 == 0:
        predictions = predictions.reshape(-1, 2)
    if len(actual.shape) == 1 and len(actual) % 2 == 0:
        actual = actual.reshape(-1, 2)
    
    try:
        # Generate the visualization
        fig = visualize_results(predictions, actual, model_name, output_dir=plots_dir)
        
        # Save the plot in the correct directory
        if fig is not None:
            plot_path = os.path.join(plots_dir, f"{model_name}_predictions.pdf")
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Predictions plot saved to: {plot_path}")
    except Exception as e:
        print(f"Error creating prediction plot: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Results for {model_name} saved to {output_dir}")

def train_model_with_tensorboard(model, train_loader, val_loader, output_dir, args, model_name):
    """
    Train a model with TensorBoard logging.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        output_dir (str): Output directory for saving results
        args (argparse.Namespace): Command line arguments
        model_name (str): Name of the model for logging
        
    Returns:
        tuple: (model, training_losses, validation_losses, stopped_early, reason)
    """
    # Create TensorBoard writer
    tensorboard_dir = os.path.join(output_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    # Get device from model
    device = next(model.parameters()).device
    
    # Train the model
    model, train_losses, val_losses, early_stopped, stopped_reason = train_model(
        model, train_loader, val_loader, epochs=args.epochs, 
        lr=args.lr, device=device, model_name=model_name,
        patience=args.patience, min_delta=args.min_delta, 
        overfitting_threshold=args.overfitting_threshold,
        output_dir=output_dir
    )
    
    # Log training and validation losses to TensorBoard
    for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        writer.add_scalar(f"{model_name}/train_loss", train_loss, epoch)
        writer.add_scalar(f"{model_name}/val_loss", val_loss, epoch)
    
    # Close the writer
    writer.close()
    
    return model, train_losses, val_losses, early_stopped, stopped_reason

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Generate dataset
    print(f"Generating dataset with {args.n_samples} samples...")
    adjacency_matrices, clique_counts = generate_clique_dataset(n_samples=args.n_samples)
    
    # Split into train, validation, and test sets
    train_adj, temp_adj, train_cliques, temp_cliques = train_test_split(
        adjacency_matrices, clique_counts, test_size=0.3, random_state=42
    )
    val_adj, test_adj, val_cliques, test_cliques = train_test_split(
        temp_adj, temp_cliques, test_size=0.5, random_state=42
    )
    
    print(f"Dataset split - Train: {len(train_adj)}, Val: {len(val_adj)}, Test: {len(test_adj)}")
    
    # Create DataLoaders
    train_dataset = GraphCliqueDataset(train_adj, train_cliques)
    val_dataset = GraphCliqueDataset(val_adj, val_cliques)
    test_dataset = GraphCliqueDataset(test_adj, test_cliques)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Determine which models to train
    models_to_train = []
    
    if args.mlp_only:
        print("Training only the MLP model")
        models_to_train.append(("mlp_model", "MLP Model"))
    elif args.ramsey_gnn_only:
        print("Training only the Ramsey GNN with Clique Attention model")
        models_to_train.append(("ramsey_gnn", "RamseyGNN"))
    else:
        print("Training all models")
        models_to_train.append(("mlp_model", "MLP Model"))
        models_to_train.append(("ramsey_gnn", "RamseyGNN"))
    
    # Create a common output directory for this run
    base_output_dir = args.output_dir
    
    # If training multiple models, create a comparison directory
    if len(models_to_train) > 1:
        print("Multiple models will be trained and compared")
        comparison_dir = create_output_dir(base_output_dir)
        print(f"Comparison results will be saved to: {comparison_dir}")
    
    trained_models = {}
    model_results = {}
    
    # Train and evaluate each model
    for model_dir, model_name in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        # Create output directory for this model
        output_dir = create_output_dir(base_output_dir, model_dir)
        print(f"Results will be saved to: {output_dir}")
        
        # Initialize the model
        if model_name == "MLP Model":
            hidden_dims = [512, 512, 256, 128]
            model = MLPCliquePredictor(n_vertices=17, hidden_dims=hidden_dims).to(device)
        elif model_name == "RamseyGNN":
            model = RamseyGraphGNNWithCliqueAttention(
                n_vertices=17, 
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                clique_attention_context_len=args.clique_attention_context,
                node_attention_context_len=8,
                num_heads=2
            ).to(device)
        
        # Train the model using custom function with TensorBoard
        trained_model, train_losses, val_losses, early_stopped, stopped_reason = train_model_with_tensorboard(
            model, train_loader, val_loader, output_dir, args, model_name
        )
        
        # Evaluate on test set
        print(f"Evaluating {model_name} on test set...")
        mse, mae, predictions, actual = evaluate_model(trained_model, test_loader, device)
        
        # Save model results
        save_training_results(
            trained_model, model_name, output_dir, train_losses, val_losses,
            mse, mae, predictions, actual, args, early_stopped, stopped_reason
        )
        
        # Store model for further analysis
        trained_models[model_name] = trained_model
        
        # Measure prediction time
        batch = test_adj[:10]  # Take a small batch for timing
        batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
        avg_time, std_time = measure_prediction_time(trained_model, batch_tensor, device)
        
        print(f"Average prediction time for {model_name}: {avg_time:.4f} ms ± {std_time:.4f} ms")
        
        # Store model results
        model_results[model_name] = {
            "mse": float(mse),
            "mae": float(mae),
            "train_losses": [float(l) for l in train_losses],
            "val_losses": [float(l) for l in val_losses],
            "final_train_loss": float(train_losses[-1]),
            "final_val_loss": float(val_losses[-1]),
            "best_val_loss": float(min(val_losses)),
            "early_stopped": early_stopped,
            "stopped_reason": stopped_reason,
            "prediction_time_ms": float(avg_time),
            "prediction_time_std_ms": float(std_time)
        }
    
    # If multiple models were trained, compare them
    if len(trained_models) > 1:
        print("\nComparing model performance...")
        
        # Save combined results to the comparison directory
        metrics_file = os.path.join(comparison_dir, "metrics", "model_comparison.json")
        with open(metrics_file, 'w') as f:
            json.dump(model_results, f, indent=2)
            
        print(f"Model comparison metrics saved to: {metrics_file}")
        
        # Plot training curves
        plt.figure(figsize=(12, 6))
        for model_name, results in model_results.items():
            plt.plot(results["train_losses"], linestyle='-', label=f"{model_name} Train")
            plt.plot(results["val_losses"], linestyle='--', label=f"{model_name} Val")
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Comparison')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        comparison_plot_path = os.path.join(comparison_dir, "plots", "training_curves_comparison.png")
        plt.savefig(comparison_plot_path)
        
        print(f"Training curves comparison saved to: {comparison_plot_path}")
        
        # Also save as PDF for publication-quality plots
        comparison_plot_pdf_path = os.path.join(comparison_dir, "plots", "training_curves_comparison.pdf")
        plt.savefig(comparison_plot_pdf_path)
        
        # Analyze performance on specific test cases
        ramsey_results = analyze_performance_on_ramsey_graphs(trained_models)
        
        ramsey_file = os.path.join(comparison_dir, "metrics", "ramsey_graph_analysis.json")
        with open(ramsey_file, 'w') as f:
            json.dump(ramsey_results, f, indent=2)
            
        print(f"Analysis of performance on Ramsey graphs saved to: {ramsey_file}")

if __name__ == "__main__":
    main() 