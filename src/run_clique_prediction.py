import argparse
import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import time
from datetime import datetime

from clique_prediction_models import (
    run_clique_prediction_experiment,
    CustomCliquePredictor,
    MLPCliquePredictor,
    generate_clique_dataset,
    train_model,
    evaluate_model,
    visualize_results
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Clique Prediction Models for Ramsey Number Analysis")
    
    parser.add_argument("--n_samples", type=int, default=5000, 
                        help="Number of graph samples to generate")
    parser.add_argument("--n_vertices", type=int, default=17, 
                        help="Number of vertices in each graph")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=20, 
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, 
                        help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="results", 
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--custom_only", action="store_true", 
                        help="Only train the custom model")
    parser.add_argument("--mlp_only", action="store_true", 
                        help="Only train the MLP model")
    parser.add_argument("--measure_time", action="store_true", 
                        help="Measure prediction time")
    parser.add_argument("--features_dim", type=int, default=256, 
                        help="Feature dimension for custom model")
    
    return parser.parse_args()

def create_output_dir(output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"clique_prediction_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
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
    import networkx as nx
    
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

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    print(f"Results will be saved to: {output_dir}")
    
    # Save arguments
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Run the experiment or custom logic based on arguments
    if args.custom_only and args.mlp_only:
        print("Error: Cannot specify both --custom_only and --mlp_only")
        return
    
    if args.custom_only or args.mlp_only:
        print("Generating dataset...")
        adjacency_matrices, max_clique_sizes = generate_clique_dataset(
            n_vertices=args.n_vertices,
            n_samples=args.n_samples,
            seed=args.seed
        )
        
        # Create model
        if args.custom_only:
            print("\nTraining Custom Model...")
            model = CustomCliquePredictor(n_vertices=args.n_vertices, features_dim=args.features_dim)
            model_name = "custom_model"
        else:
            print("\nTraining MLP Model...")
            model = MLPCliquePredictor(n_vertices=args.n_vertices)
            model_name = "mlp_model"
        
        # Create datasets (simplified version without proper validation for brevity)
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, TensorDataset
        
        X_train, X_test, y_train, y_test = train_test_split(
            adjacency_matrices, max_clique_sizes, test_size=0.2, random_state=args.seed
        )
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        # Train model
        start_time = time.time()
        trained_model, train_losses, val_losses = train_model(
            model, train_loader, test_loader,  # Using test as validation for simplicity
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            model_name=os.path.join(output_dir, model_name)
        )
        training_time = time.time() - start_time
        
        # Evaluate model
        mse, mae, predictions, actual = evaluate_model(
            trained_model, test_loader, device=device
        )
        
        # Visualize results
        visualize_results(predictions, actual, model_name=os.path.join(output_dir, model_name))
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title(f'{model_name} Training Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_training_curve.png"))
        plt.close()
        
        # Measure prediction time if requested
        if args.measure_time:
            batch = X_test_tensor[:100].to(device)
            avg_time, std_time = measure_prediction_time(trained_model, batch, device, n_runs=100)
            print(f"\nAverage prediction time: {avg_time:.2f} ms ± {std_time:.2f}")
        
        # Save results
        results = {
            model_name: {
                "MSE": float(mse),
                "MAE": float(mae),
                "Training Time (seconds)": training_time,
                "Epochs": args.epochs,
                "Learning Rate": args.lr,
                "Batch Size": args.batch_size,
                "Features Dim": args.features_dim if args.custom_only else "N/A"
            }
        }
        
        # Analyze on special Ramsey graphs
        ramsey_results = analyze_performance_on_ramsey_graphs(
            {model_name: trained_model}, 
            n_vertices=args.n_vertices
        )
        results[model_name]["Ramsey Graphs Analysis"] = ramsey_results[model_name]
        
        # Save results to JSON
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
        print("\n=== Results ===")
        print(f"Model: {model_name}")
        print(f"Test MSE: {mse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Training Time: {training_time:.2f} seconds")
    
    else:
        # Run the full experiment comparing both models
        results = run_clique_prediction_experiment(
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=device
        )
        
        # Save results to file
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
    
    print(f"\nAll results saved to {output_dir}")

if __name__ == "__main__":
    main() 