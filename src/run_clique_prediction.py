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

from clique_prediction_models import (
    CustomCliquePredictor,
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
    parser.add_argument('--custom_only', action='store_true',
                        help='Train only the Custom model')
    parser.add_argument('--ramsey_gnn_only', action='store_true',
                        help='Train only the Ramsey GNN with Clique Attention model')
    
    args = parser.parse_args()
    
    # Check for conflicting model flags
    model_flags = [args.mlp_only, args.custom_only, args.ramsey_gnn_only]
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
    Train a model with tensorboard logging.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        output_dir (str): Output directory
        args (argparse.Namespace): Command line arguments
        model_name (str): Name of the model
        
    Returns:
        tuple: (model, train_losses, val_losses, early_stopped, stopped_reason)
    """
    # Setup tensorboard writer
    tensorboard_dir = os.path.join(output_dir, "tensorboard")
    writer = SummaryWriter(tensorboard_dir)
    
    # Train the model with standard function, but add tensorboard logging
    model, train_losses, val_losses, early_stopped, stopped_reason = train_model(
        model, train_loader, val_loader, epochs=args.epochs,
        lr=args.lr, device=args.device, model_name=model_name,
        patience=args.patience, min_delta=args.min_delta, 
        overfitting_threshold=args.overfitting_threshold,
        output_dir=output_dir
    )
    
    # Log the losses
    for epoch in range(len(train_losses)):
        writer.add_scalar(f'{model_name}/train_loss', train_losses[epoch], epoch)
        writer.add_scalar(f'{model_name}/val_loss', val_losses[epoch], epoch)
    
    # Log early stopping information
    writer.add_text('Training/early_stopped', str(early_stopped), 0)
    writer.add_text('Training/stopped_reason', stopped_reason, 0)
    
    # Save early stopping information to a JSON file
    stopping_info = {
        'early_stopped': early_stopped,
        'stopped_reason': stopped_reason,
        'epochs_completed': len(train_losses),
        'max_epochs': args.epochs
    }
    with open(os.path.join(output_dir, 'metrics', f'{model_name}_stopping_info.json'), 'w') as f:
        json.dump(stopping_info, f, indent=4)
    
    # Close the writer
    writer.close()
    
    return model, train_losses, val_losses, early_stopped, stopped_reason

def main():
    args = parse_args()
    
    # Set device
    if torch.cuda.is_available() and not args.no_gpu:
        device = torch.device('cuda')
        print(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")
    
    # Add device to args for use in other functions
    args.device = device
    
    # Generate dataset
    print(f"Generating dataset...")
    adjacency_matrices, clique_counts = generate_clique_dataset(n_samples=args.n_samples)
    
    # Create dataset and data loaders
    dataset = GraphCliqueDataset(adjacency_matrices, clique_counts)
    
    # Split dataset into train, validation, and test sets (70/15/15)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # Use a fixed random seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Define common training parameters
    n_vertices = 17  # Fixed for Ramsey graphs in this project
    
    # Train and evaluate models based on flags
    if args.mlp_only:
        # Create model-specific output directory
        output_dir = create_output_dir(args.output_dir, "mlp_model")
        print(f"Results will be saved to: {output_dir}")
        
        # Save arguments to file
        with open(os.path.join(output_dir, 'args.json'), 'w') as f:
            # Create a copy of args without the device object
            args_dict = vars(args).copy()
            args_dict['device'] = str(args_dict['device'])  # Convert device to string
            json.dump(args_dict, f, indent=4)
        
        # Train MLP model
        print("\nTraining MLP Model...")
        model = MLPCliquePredictor(n_vertices=n_vertices)
        model, train_losses, val_losses, early_stopped, stopped_reason = train_model_with_tensorboard(
            model, train_loader, val_loader, output_dir, args, "mlp_model"
        )
        
        # Evaluate the model
        print("\nEvaluating MLP Model...")
        mse, mae, predictions, actual = evaluate_model(model, test_loader, device=device)
        
        # Save all results
        save_training_results(model, "MLP_Model", output_dir, train_losses, val_losses, 
                             mse, mae, predictions, actual, args, early_stopped, stopped_reason)
        
        # Print results
        print(f"MLP Model - Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")
        
    elif args.custom_only:
        # Create model-specific output directory
        output_dir = create_output_dir(args.output_dir, "custom_model")
        print(f"Results will be saved to: {output_dir}")
        
        # Save arguments to file
        with open(os.path.join(output_dir, 'args.json'), 'w') as f:
            # Create a copy of args without the device object
            args_dict = vars(args).copy()
            args_dict['device'] = str(args_dict['device'])  # Convert device to string
            json.dump(args_dict, f, indent=4)
        
        # Train Custom model
        print("\nTraining Custom Model...")
        model = CustomCliquePredictor(n_vertices=n_vertices, features_dim=256)
        model, train_losses, val_losses, early_stopped, stopped_reason = train_model_with_tensorboard(
            model, train_loader, val_loader, output_dir, args, "custom_model"
        )
        
        # Evaluate the model
        print("\nEvaluating Custom Model...")
        mse, mae, predictions, actual = evaluate_model(model, test_loader, device=device)
        
        # Save all results
        save_training_results(model, "Custom_Model", output_dir, train_losses, val_losses, 
                             mse, mae, predictions, actual, args, early_stopped, stopped_reason)
        
        # Print results
        print(f"Custom Model - Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")
        
    elif args.ramsey_gnn_only:
        # Create model-specific output directory
        output_dir = create_output_dir(args.output_dir, "ramsey_gnn")
        print(f"Results will be saved to: {output_dir}")
        
        # Save arguments to file
        with open(os.path.join(output_dir, 'args.json'), 'w') as f:
            # Create a copy of args without the device object
            args_dict = vars(args).copy()
            args_dict['device'] = str(args_dict['device'])  # Convert device to string
            json.dump(args_dict, f, indent=4)
        
        # Train Ramsey Graph GNN with Clique Attention model
        print("\nTraining Ramsey Graph GNN with Clique Attention Model...")
        model = RamseyGraphGNNWithCliqueAttention(
            n_vertices=n_vertices, 
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            clique_attention_context_len=args.clique_attention_context
        )
        model, train_losses, val_losses, early_stopped, stopped_reason = train_model_with_tensorboard(
            model, train_loader, val_loader, output_dir, args, "ramsey_gnn"
        )
        
        # Evaluate the model
        print("\nEvaluating Ramsey Graph GNN with Clique Attention Model...")
        mse, mae, predictions, actual = evaluate_model(model, test_loader, device=device)
        
        # Save all results
        save_training_results(model, "Ramsey_GNN", output_dir, train_losses, val_losses, 
                             mse, mae, predictions, actual, args, early_stopped, stopped_reason)
        
        # Print results
        print(f"Ramsey Graph GNN with Clique Attention Model - Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")
        
    else:
        # Create general output directory for comparison
        output_dir = create_output_dir(args.output_dir)
        print(f"Results will be saved to: {output_dir}")
        
        # Save arguments to file
        with open(os.path.join(output_dir, 'args.json'), 'w') as f:
            # Create a copy of args without the device object
            args_dict = vars(args).copy()
            args_dict['device'] = str(args_dict['device'])  # Convert device to string
            json.dump(args_dict, f, indent=4)
        
        # Dictionary to store all results
        all_models = {}
        all_metrics = {}
        
        # Train Custom model
        print("\nTraining Custom Model...")
        custom_model = CustomCliquePredictor(n_vertices=n_vertices, features_dim=256)
        custom_model, custom_train_losses, custom_val_losses, custom_early_stopped, custom_stopped_reason = train_model_with_tensorboard(
            custom_model, train_loader, val_loader, output_dir, args, "custom_model"
        )
        
        # Evaluate the model
        print("\nEvaluating Custom Model...")
        custom_mse, custom_mae, custom_preds, custom_actual = evaluate_model(
            custom_model, test_loader, device=device
        )
        
        # Save results
        save_training_results(custom_model, "Custom_Model", output_dir, custom_train_losses, 
                             custom_val_losses, custom_mse, custom_mae, custom_preds, 
                             custom_actual, args, custom_early_stopped, custom_stopped_reason)
        
        all_models["Custom_Model"] = custom_model
        all_metrics["Custom_Model"] = {"mse": custom_mse, "mae": custom_mae}
        
        # Train MLP model
        print("\nTraining MLP Model...")
        mlp_model = MLPCliquePredictor(n_vertices=n_vertices)
        mlp_model, mlp_train_losses, mlp_val_losses, mlp_early_stopped, mlp_stopped_reason = train_model_with_tensorboard(
            mlp_model, train_loader, val_loader, output_dir, args, "mlp_model"
        )
        
        # Evaluate the model
        print("\nEvaluating MLP Model...")
        mlp_mse, mlp_mae, mlp_preds, mlp_actual = evaluate_model(
            mlp_model, test_loader, device=device
        )
        
        # Save results
        save_training_results(mlp_model, "MLP_Model", output_dir, mlp_train_losses, 
                             mlp_val_losses, mlp_mse, mlp_mae, mlp_preds, 
                             mlp_actual, args, mlp_early_stopped, mlp_stopped_reason)
        
        all_models["MLP_Model"] = mlp_model
        all_metrics["MLP_Model"] = {"mse": mlp_mse, "mae": mlp_mae}
        
        # Train Ramsey Graph GNN with Clique Attention model
        print("\nTraining Ramsey Graph GNN with Clique Attention Model...")
        ramsey_model = RamseyGraphGNNWithCliqueAttention(
            n_vertices=n_vertices, 
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            clique_attention_context_len=args.clique_attention_context
        )
        ramsey_model, ramsey_train_losses, ramsey_val_losses, ramsey_early_stopped, ramsey_stopped_reason = train_model_with_tensorboard(
            ramsey_model, train_loader, val_loader, output_dir, args, "ramsey_gnn"
        )
        
        # Evaluate the model
        print("\nEvaluating Ramsey Graph GNN with Clique Attention Model...")
        ramsey_mse, ramsey_mae, ramsey_preds, ramsey_actual = evaluate_model(
            ramsey_model, test_loader, device=device
        )
        
        # Save results
        save_training_results(ramsey_model, "Ramsey_GNN", output_dir, ramsey_train_losses, 
                             ramsey_val_losses, ramsey_mse, ramsey_mae, ramsey_preds, 
                             ramsey_actual, args, ramsey_early_stopped, ramsey_stopped_reason)
        
        all_models["Ramsey_GNN"] = ramsey_model
        all_metrics["Ramsey_GNN"] = {"mse": ramsey_mse, "mae": ramsey_mae}
        
        # Plot comparative training curves
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(custom_train_losses, label='Custom Model')
        plt.plot(mlp_train_losses, label='MLP Model')
        plt.plot(ramsey_train_losses, label='Ramsey GNN')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training Loss Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(custom_val_losses, label='Custom Model')
        plt.plot(mlp_val_losses, label='MLP Model')
        plt.plot(ramsey_val_losses, label='Ramsey GNN')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Validation Loss Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plots", "training_curves_comparison.pdf"))
        plt.close()
        
        # Plot comparative bar chart of MSE and MAE
        model_names = list(all_metrics.keys())
        mse_values = [all_metrics[model]["mse"] for model in model_names]
        mae_values = [all_metrics[model]["mae"] for model in model_names]
        
        bar_width = 0.35
        indices = np.arange(len(model_names))
        
        plt.figure(figsize=(12, 6))
        plt.bar(indices - bar_width/2, mse_values, bar_width, label='MSE')
        plt.bar(indices + bar_width/2, mae_values, bar_width, label='MAE')
        plt.xlabel('Model')
        plt.ylabel('Error')
        plt.title('Model Performance Comparison')
        plt.xticks(indices, model_names)
        plt.legend()
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(output_dir, "plots", "model_performance_comparison.pdf"))
        plt.close()
        
        # Save comparison results in a single JSON file
        comparison_results = {
            "models": {},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size": args.n_samples,
            "epochs": args.epochs
        }
        
        for model_name in all_metrics:
            comparison_results["models"][model_name] = {
                "mse": float(all_metrics[model_name]["mse"]),
                "mae": float(all_metrics[model_name]["mae"])
            }
        
        with open(os.path.join(output_dir, "metrics", "model_comparison.json"), 'w') as f:
            json.dump(comparison_results, f, indent=4)
        
        # Print final results
        print("\n=== Results ===")
        print(f"Custom Model - Test MSE: {custom_mse:.4f}, Test MAE: {custom_mae:.4f}")
        print(f"MLP Model - Test MSE: {mlp_mse:.4f}, Test MAE: {mlp_mae:.4f}")
        print(f"Ramsey Graph GNN with Clique Attention - Test MSE: {ramsey_mse:.4f}, Test MAE: {ramsey_mae:.4f}")
    
    print(f"Experiment completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 