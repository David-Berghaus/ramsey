import os
import json
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
import pandas as pd
import seaborn as sns

from clique_prediction_models import (
    CustomCliquePredictor,
    MLPCliquePredictor,
    RamseyGraphGNNWithCliqueAttention,
    evaluate_model,
    GraphCliqueDataset
)

def parse_args():
    parser = argparse.ArgumentParser(description='Compare Clique Prediction Models')
    
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Base directory containing model results')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                        help='Directory to save comparison results')
    parser.add_argument('--test_set_path', type=str, default=None,
                        help='Path to a test set to evaluate models (optional)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--no_gpu', action='store_true',
                        help='Disable GPU usage even if available')
    
    return parser.parse_args()

def find_latest_runs(base_dir):
    """
    Find the latest run directories for each model type.
    
    Args:
        base_dir (str): Base directory containing model results
        
    Returns:
        dict: Dictionary mapping model types to their latest run directories
    """
    model_types = ['mlp_model', 'custom_model', 'ramsey_gnn']
    # Map from model_type to possible file prefixes
    model_name_variants = {
        'mlp_model': ['mlp_model', 'MLP_Model'],
        'custom_model': ['custom_model', 'Custom_Model'],
        'ramsey_gnn': ['ramsey_gnn', 'Ramsey_GNN']
    }
    latest_runs = {}
    
    # First, check for model-specific directories
    for model_type in model_types:
        model_dir = os.path.join(base_dir, model_type)
        
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            # Find all run directories
            run_dirs = [d for d in os.listdir(model_dir) 
                        if os.path.isdir(os.path.join(model_dir, d))]
            
            if run_dirs:
                # Sort by timestamp (assuming directory names are timestamps)
                run_dirs.sort(reverse=True)
                
                # Get the latest run
                latest_run = os.path.join(model_dir, run_dirs[0])
                latest_runs[model_type] = latest_run
                print(f"Found latest run for {model_type}: {latest_run}")
    
    # If we didn't find any model-specific directories, check if base_dir is a comparison directory
    if len(latest_runs) < len(model_types) and os.path.isdir(base_dir):
        # Check if this is a comparison directory with all models
        metrics_dir = os.path.join(base_dir, "metrics")
        if os.path.exists(metrics_dir):
            # Get all metrics files
            metrics_files = glob.glob(os.path.join(metrics_dir, '*_metrics.json'))
            
            # Look for model-specific metrics files
            for model_type in model_types:
                if model_type in latest_runs:
                    continue  # Skip if we already found this model
                
                # Check all possible name variants
                for name_variant in model_name_variants[model_type]:
                    metrics_file = os.path.join(metrics_dir, f"{name_variant}_metrics.json")
                    if os.path.exists(metrics_file):
                        latest_runs[model_type] = base_dir
                        print(f"Found {model_type} in comparison directory: {base_dir}")
                        break
    
    # If we still didn't find all models, check if there are comparison directories in the base_dir
    if len(latest_runs) < len(model_types) and os.path.isdir(base_dir):
        # Look for comparison directories
        comparison_dirs = [d for d in os.listdir(base_dir) 
                          if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("comparison_")]
        
        if comparison_dirs:
            # Sort by timestamp (assuming directory names are timestamps)
            comparison_dirs.sort(reverse=True)
            
            # Get the latest comparison directory
            latest_comparison = os.path.join(base_dir, comparison_dirs[0])
            print(f"Found latest comparison directory: {latest_comparison}")
            
            # Check for model metrics in this directory
            metrics_dir = os.path.join(latest_comparison, "metrics")
            if os.path.exists(metrics_dir):
                # Look for model-specific metrics files
                for model_type in model_types:
                    if model_type in latest_runs:
                        continue  # Skip if we already found this model
                    
                    # Check all possible name variants
                    for name_variant in model_name_variants[model_type]:
                        metrics_file = os.path.join(metrics_dir, f"{name_variant}_metrics.json")
                        if os.path.exists(metrics_file):
                            latest_runs[model_type] = latest_comparison
                            print(f"Found {model_type} in comparison directory: {latest_comparison}")
                            break
    
    # Print summary of what we found
    if latest_runs:
        print("\nFound the following models:")
        for model_type, run_dir in latest_runs.items():
            print(f"- {model_type}: {run_dir}")
    else:
        print("\nNo models found in the specified directory.")
    
    return latest_runs

def load_metrics(run_dir, model_type):
    """
    Load metrics from a run directory.
    
    Args:
        run_dir (str): Run directory containing metrics
        model_type (str): Type of model ('mlp_model', 'custom_model', or 'ramsey_gnn')
        
    Returns:
        dict: Dictionary of metrics
    """
    # Map from model_type to possible file prefixes
    model_name_variants = {
        'mlp_model': ['mlp_model', 'MLP_Model'],
        'custom_model': ['custom_model', 'Custom_Model'],
        'ramsey_gnn': ['ramsey_gnn', 'Ramsey_GNN']
    }
    
    metrics_dir = os.path.join(run_dir, 'metrics')
    if not os.path.exists(metrics_dir):
        return None
    
    # Try all possible name variants
    for name_variant in model_name_variants[model_type]:
        metrics_file = os.path.join(metrics_dir, f"{name_variant}_metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                return metrics
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {metrics_file}. Trying to fix...")
                # Try to fix the JSON file by reading it as text and adding missing closing brace
                with open(metrics_file, 'r') as f:
                    content = f.read()
                if not content.strip().endswith('}'):
                    content += '\n}'
                    # Write the fixed content back
                    with open(metrics_file, 'w') as f:
                        f.write(content)
                    # Try to load again
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    return metrics
    
    # If we get here, we didn't find any matching metrics file
    print(f"No metrics file found for {model_type} in {run_dir}")
    return None

def load_training_losses(run_dir, model_type):
    """
    Load training and validation losses from a run directory.
    
    Args:
        run_dir (str): Run directory containing losses
        model_type (str): Type of model ('mlp_model', 'custom_model', or 'ramsey_gnn')
        
    Returns:
        tuple: (train_losses, val_losses)
    """
    # Map from model_type to possible file prefixes
    model_name_variants = {
        'mlp_model': ['mlp_model', 'MLP_Model'],
        'custom_model': ['custom_model', 'Custom_Model'],
        'ramsey_gnn': ['ramsey_gnn', 'Ramsey_GNN']
    }
    
    metrics_dir = os.path.join(run_dir, 'metrics')
    if not os.path.exists(metrics_dir):
        return None, None
    
    # Try all possible name variants
    for name_variant in model_name_variants[model_type]:
        losses_file = os.path.join(metrics_dir, f"{name_variant}_losses.json")
        if os.path.exists(losses_file):
            try:
                with open(losses_file, 'r') as f:
                    losses = json.load(f)
                
                return losses.get('train_losses'), losses.get('val_losses')
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {losses_file}. Trying to fix...")
                # Try to fix the JSON file by reading it as text and adding missing closing brace
                with open(losses_file, 'r') as f:
                    content = f.read()
                if not content.strip().endswith('}'):
                    content += '\n}'
                    # Write the fixed content back
                    with open(losses_file, 'w') as f:
                        f.write(content)
                    # Try to load again
                    with open(losses_file, 'r') as f:
                        losses = json.load(f)
                    return losses.get('train_losses'), losses.get('val_losses')
    
    # If we get here, we didn't find any matching losses file
    print(f"No losses file found for {model_type} in {run_dir}")
    return None, None

def load_model(model_type, run_dir, device):
    """
    Load a model from weights in a run directory.
    
    Args:
        model_type (str): Type of model ('mlp_model', 'custom_model', or 'ramsey_gnn')
        run_dir (str): Run directory containing model weights
        device (torch.device): Device to load model onto
        
    Returns:
        nn.Module: Loaded model
    """
    # Map model type to model class
    model_name_to_class = {
        'mlp_model': MLPCliquePredictor,
        'custom_model': CustomCliquePredictor,
        'ramsey_gnn': RamseyGraphGNNWithCliqueAttention
    }
    
    # Map model type to expected weight filename pattern
    model_name_to_weights_pattern = {
        'mlp_model': 'MLP_Model',
        'custom_model': 'Custom_Model',
        'ramsey_gnn': 'Ramsey_GNN'
    }
    
    # Get model class and weights pattern
    model_class = model_name_to_class.get(model_type)
    weights_pattern = model_name_to_weights_pattern.get(model_type)
    
    if not model_class or not weights_pattern:
        print(f"Unknown model type: {model_type}")
        return None
    
    # Find weights file
    weights_file = os.path.join(run_dir, 'model_weights', f"{weights_pattern}_best.pt")
    
    # Try to find any weights file if the expected one doesn't exist
    if not os.path.exists(weights_file):
        weights_files = glob.glob(os.path.join(run_dir, 'model_weights', '*_best.pt'))
        if weights_files:
            weights_file = weights_files[0]
        else:
            print(f"No weights file found for {model_type} in {run_dir}")
            return None
    
    # Create and load model
    if model_type == 'mlp_model':
        model = model_class()
    elif model_type == 'custom_model':
        model = model_class(features_dim=256)
    elif model_type == 'ramsey_gnn':
        # Load args to get hyperparameters
        args_file = os.path.join(run_dir, 'args.json')
        if os.path.exists(args_file):
            with open(args_file, 'r') as f:
                args = json.load(f)
            hidden_dim = args.get('hidden_dim', 64)
            num_layers = args.get('num_layers', 3)
            clique_attention_context = args.get('clique_attention_context', 20)
            model = model_class(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                clique_attention_context_len=clique_attention_context
            )
        else:
            # Use default values
            model = model_class()
    
    # Load weights
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model

def plot_training_curves(model_losses, output_dir):
    """
    Plot training curves for all models.
    
    Args:
        model_losses (dict): Dictionary mapping model types to (train_losses, val_losses)
        output_dir (str): Directory to save plot
    """
    plt.figure(figsize=(12, 10))
    
    # Plot training losses
    plt.subplot(2, 1, 1)
    for model_type, (train_losses, _) in model_losses.items():
        if train_losses:
            plt.plot(train_losses, label=f"{model_type}")
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot validation losses
    plt.subplot(2, 1, 2)
    for model_type, (_, val_losses) in model_losses.items():
        if val_losses:
            plt.plot(val_losses, label=f"{model_type}")
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves_comparison.pdf'))
    plt.close()

def plot_metrics_comparison(model_metrics, output_dir):
    """
    Plot metrics comparison for all models.
    
    Args:
        model_metrics (dict): Dictionary mapping model types to metrics
        output_dir (str): Directory to save plot
    """
    model_names = list(model_metrics.keys())
    
    # Extract MSE and MAE values
    mse_values = [metrics.get('test_mse', 0) for metrics in model_metrics.values()]
    mae_values = [metrics.get('test_mae', 0) for metrics in model_metrics.values()]
    
    # Create dataframe for seaborn
    data = []
    for model, metrics in model_metrics.items():
        data.append({
            'Model': model,
            'Metric': 'MSE',
            'Value': metrics.get('test_mse', 0)
        })
        data.append({
            'Model': model,
            'Metric': 'MAE',
            'Value': metrics.get('test_mae', 0)
        })
    
    df = pd.DataFrame(data)
    
    # Plot with seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Model', y='Value', hue='Metric')
    plt.title('Model Performance Metrics')
    plt.ylabel('Error')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_metrics_comparison.pdf'))
    plt.close()

def evaluate_on_test_set(models, test_loader, device, output_dir):
    """
    Evaluate all models on the same test set.
    
    Args:
        models (dict): Dictionary mapping model types to model instances
        test_loader (torch.utils.data.DataLoader): Test data loader
        device (torch.device): Device to evaluate on
        output_dir (str): Directory to save results
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    results = {}
    all_predictions = defaultdict(list)
    all_actual = []
    
    # Evaluate each model
    for model_type, model in models.items():
        if model is not None:
            mse, mae, predictions, actual = evaluate_model(model, test_loader, device=device)
            
            results[model_type] = {
                'mse': float(mse),
                'mae': float(mae)
            }
            
            all_predictions[model_type] = predictions
            if not all_actual:  # Only need to save once
                all_actual = actual
    
    # Save results
    with open(os.path.join(output_dir, 'test_set_evaluation.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plot predictions vs actual for original graph clique counts
    plt.figure(figsize=(12, 8))
    
    for i, model_type in enumerate(all_predictions.keys()):
        preds = all_predictions[model_type]
        
        plt.subplot(1, 2, 1)
        plt.scatter(all_actual[:, 0], preds[:, 0], alpha=0.5, label=model_type)
        plt.subplot(1, 2, 2)
        plt.scatter(all_actual[:, 1], preds[:, 1], alpha=0.5, label=model_type)
    
    # Add diagonal line (perfect predictions)
    plt.subplot(1, 2, 1)
    min_val = np.min(all_actual[:, 0])
    max_val = np.max(all_actual[:, 0])
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.xlabel('Actual 4-Clique Count (Original)')
    plt.ylabel('Predicted 4-Clique Count')
    plt.title('Original Graph')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    min_val = np.min(all_actual[:, 1])
    max_val = np.max(all_actual[:, 1])
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.xlabel('Actual 4-Clique Count (Complement)')
    plt.ylabel('Predicted 4-Clique Count')
    plt.title('Complement Graph')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_comparison.pdf'))
    plt.close()
    
    return results

def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    if torch.cuda.is_available() and not args.no_gpu:
        device = torch.device('cuda')
        print(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find latest runs for each model type
    latest_runs = find_latest_runs(args.results_dir)
    
    if not latest_runs:
        print("No model runs found. Please train models first.")
        return
    
    # Load metrics and losses for each model
    model_metrics = {}
    model_losses = {}
    
    for model_type, run_dir in latest_runs.items():
        metrics = load_metrics(run_dir, model_type)
        if metrics:
            model_metrics[model_type] = metrics
        
        train_losses, val_losses = load_training_losses(run_dir, model_type)
        if train_losses and val_losses:
            model_losses[model_type] = (train_losses, val_losses)
    
    # Plot training curves comparison
    if model_losses:
        plot_training_curves(model_losses, args.output_dir)
    
    # Plot metrics comparison
    if model_metrics:
        plot_metrics_comparison(model_metrics, args.output_dir)
    
    # If test set path is provided, evaluate models on it
    if args.test_set_path and os.path.exists(args.test_set_path):
        print(f"Loading test set from {args.test_set_path}")
        
        # Load test set
        test_data = np.load(args.test_set_path, allow_pickle=True)
        adjacency_matrices = test_data['adjacency_matrices']
        clique_counts = test_data['clique_counts']
        
        test_dataset = GraphCliqueDataset(adjacency_matrices, clique_counts)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
        
        # Load models
        models = {}
        for model_type, run_dir in latest_runs.items():
            model = load_model(model_type, run_dir, device)
            if model:
                models[model_type] = model
        
        # Evaluate models on test set
        if models:
            evaluate_on_test_set(models, test_loader, device, args.output_dir)
    else:
        print("No test set provided or file not found. Skipping evaluation on test set.")
        # Load example predictions from run directories to compare
        plot_predictions_from_runs(latest_runs, args.output_dir)
    
    print(f"Comparison results saved to {args.output_dir}")

def plot_predictions_from_runs(run_dirs, output_dir):
    """
    Plot predictions from existing run directories without re-evaluating.
    
    Args:
        run_dirs (dict): Dictionary mapping model types to run directories
        output_dir (str): Directory to save plot
    """
    # Combine existing prediction plots into a single comparison plot
    plt.figure(figsize=(15, 10))
    
    for model_type, run_dir in run_dirs.items():
        # Find prediction plot files
        plot_dir = os.path.join(run_dir, 'plots')
        pred_plots = glob.glob(os.path.join(plot_dir, '*_predictions.pdf'))
        
        if pred_plots:
            # Just mention we found the plots, but we're not actually loading the images
            # We'll just reference them in the comparison
            print(f"Found prediction plot for {model_type}: {pred_plots[0]}")
    
    # Create a summary table of model performance
    plt.figtext(0.5, 0.5, 
                "Comparison of Model Predictions\n\n"
                "See individual model prediction plots in their respective run directories:\n" + 
                "\n".join([f"{model}: {dir}" for model, dir in run_dirs.items()]),
                ha='center', va='center', fontsize=12)
    
    plt.savefig(os.path.join(output_dir, 'predictions_reference.pdf'))
    plt.close()

if __name__ == "__main__":
    main() 