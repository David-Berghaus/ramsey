import argparse
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import os
from scipy.special import comb

from clique_prediction_models import CustomCliquePredictor, MLPCliquePredictor
from env import obs_space_to_graph, flattened_off_diagonal_to_adjacency_matrix
from score import get_score_and_cliques, get_cliques_and_count
from model import NodeMeanPoolCliqueAttentionFeatureExtractor

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze specific Ramsey graph examples")
    
    parser.add_argument("--custom_model_path", type=str, required=True,
                        help="Path to the trained custom model checkpoint")
    parser.add_argument("--mlp_model_path", type=str, required=True,
                        help="Path to the trained MLP model checkpoint")
    parser.add_argument("--output_dir", type=str, default="ramsey_analysis",
                        help="Directory to save analysis results")
    parser.add_argument("--n_vertices", type=int, default=17,
                        help="Number of vertices in the graphs")
    
    return parser.parse_args()

def create_special_graphs(n_vertices=17):
    """
    Create special graph examples for Ramsey number analysis.
    
    Returns:
        list: List of (graph_name, flattened_adjacency_matrix) tuples
    """
    n_entries = n_vertices * (n_vertices - 1) // 2
    graphs = []
    
    # 1. Graph with a 4-clique
    g1 = np.zeros(n_entries)
    edge_idx = 0
    for i in range(n_vertices):
        for j in range(i+1, n_vertices):
            if i < 4 and j < 4:  # Connect vertices 0,1,2,3 to form a 4-clique
                g1[edge_idx] = 1
            edge_idx += 1
    
    graphs.append(("Graph with 4-clique", g1))
    
    # 2. Graph with a 4-independent set (4-clique in complement)
    g2 = np.ones(n_entries)
    edge_idx = 0
    for i in range(n_vertices):
        for j in range(i+1, n_vertices):
            if i < 4 and j < 4:  # Disconnect vertices 0,1,2,3 to form a 4-independent set
                g2[edge_idx] = 0
            edge_idx += 1
    
    graphs.append(("Graph with 4-independent set", g2))
    
    # 3. Graph with multiple 4-cliques
    g3 = np.zeros(n_entries)
    edge_idx = 0
    for i in range(n_vertices):
        for j in range(i+1, n_vertices):
            # Create multiple disjoint 4-cliques
            if (i < 4 and j < 4) or (4 <= i < 8 and 4 <= j < 8):
                g3[edge_idx] = 1
            edge_idx += 1
    
    graphs.append(("Graph with multiple 4-cliques", g3))
    
    # 4. Balanced graph - edge probability around 0.5
    # This is a typical example for Ramsey analysis
    np.random.seed(43)
    g4 = (np.random.rand(n_entries) > 0.5).astype(int)
    graphs.append(("Balanced graph (p=0.5)", g4))
    
    # 5. Higher density graph - fewer independent sets
    np.random.seed(44)
    g5 = (np.random.rand(n_entries) > 0.3).astype(int)
    graphs.append(("Dense graph (p=0.7)", g5))
    
    # 6. Lower density graph - fewer cliques
    np.random.seed(45)
    g6 = (np.random.rand(n_entries) > 0.7).astype(int)
    graphs.append(("Sparse graph (p=0.3)", g6))
    
    return graphs

def load_models(custom_model_path, mlp_model_path, n_vertices=17):
    """
    Load the trained models.
    
    Args:
        custom_model_path (str): Path to the custom model checkpoint
        mlp_model_path (str): Path to the MLP model checkpoint
        n_vertices (int): Number of vertices in the graphs
        
    Returns:
        tuple: (custom_model, mlp_model)
    """
    device = torch.device("cpu")  # Force CPU for compatibility
    
    # Load custom model
    custom_model = CustomCliquePredictor(n_vertices=n_vertices)
    custom_model.device = device  # Override the device
    
    # Create dummy observation space
    from gymnasium import spaces
    observation_space = spaces.MultiBinary(custom_model.n_entries)
    
    # Initialize feature extractor before loading the state dict
    custom_model.feature_extractor = NodeMeanPoolCliqueAttentionFeatureExtractor(
        observation_space=observation_space,
        n=custom_model.n,
        r=custom_model.r,
        b=custom_model.b,
        not_connected_punishment=custom_model.not_connected_punishment,
        features_dim=custom_model.features_dim,
        num_heads=custom_model.num_heads,
        node_attention_context_len=custom_model.node_attention_context_len,
        clique_attention_context_len=custom_model.clique_attention_context_len
    ).to(device)
    
    custom_model.load_state_dict(torch.load(custom_model_path, map_location=device))
    custom_model = custom_model.to(device)
    custom_model.eval()
    
    # Load MLP model
    mlp_model = MLPCliquePredictor(n_vertices=n_vertices)
    mlp_model.device = device  # Override the device
    mlp_model.load_state_dict(torch.load(mlp_model_path, map_location=device))
    mlp_model = mlp_model.to(device)
    mlp_model.eval()
    
    return custom_model, mlp_model

def visualize_graph(G, title, max_clique=None, node_colors=None, ax=None):
    """
    Visualize a graph with highlighting of the maximum clique.
    
    Args:
        G (networkx.Graph): Graph to visualize
        title (str): Plot title
        max_clique (list): Nodes in the maximum clique to highlight
        node_colors (list): Optional list of node colors
        ax (matplotlib.Axes): Optional axes to plot on
        
    Returns:
        matplotlib.Axes: The plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set default node colors if not provided
    if node_colors is None:
        node_colors = ['skyblue' for _ in range(len(G.nodes()))]
    
    # Highlight the maximum clique if provided
    if max_clique is not None:
        for i in max_clique:
            node_colors[i] = 'red'
    
    # Position nodes in a circle
    pos = nx.circular_layout(G)
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
            node_size=700, font_weight='bold', ax=ax)
    
    ax.set_title(title)
    return ax

def analyze_graph(graph_name, adjacency_matrix, models, n_vertices, output_dir):
    """
    Analyze a specific graph with the trained models.
    
    Args:
        graph_name (str): Name of the graph
        adjacency_matrix (np.array): Flattened adjacency matrix
        models (tuple): (custom_model, mlp_model)
        n_vertices (int): Number of vertices
        output_dir (str): Directory to save results
        
    Returns:
        dict: Analysis results
    """
    custom_model, mlp_model = models
    device = next(mlp_model.parameters()).device
    
    # Parameters for Ramsey analysis
    r = 4  # We're looking for 4-cliques
    b = 4  # And 4-independent sets (4-cliques in complement)
    
    # Convert to graph
    G = obs_space_to_graph(adjacency_matrix, n_vertices)
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
    
    # Highlight one clique of size 4 if available for visualization
    max_clique_original = next((c for c in cliques_r if len(c) >= r), [])
    max_clique_complement = next((c for c in cliques_b if len(c) >= b), [])
    
    # Prepare input for models
    input_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Get predictions - for now, only use MLP model due to device issues with custom model
    with torch.no_grad():
        # custom_pred = custom_model(input_tensor).cpu().numpy()[0]
        mlp_pred = mlp_model(input_tensor).cpu().numpy()[0]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Original graph with highlighted clique (if available)
    visualize_graph(G, f"Original Graph\nActual 4-Clique Count: {count_r}", 
                   max_clique=max_clique_original[:r] if len(max_clique_original) >= r else None, 
                   ax=axes[0, 0])
    
    # Complement graph with highlighted clique (if available)
    visualize_graph(G_complement, f"Complement Graph\nActual 4-Clique Count: {count_b}", 
                   max_clique=max_clique_complement[:b] if len(max_clique_complement) >= b else None, 
                   ax=axes[0, 1])
    
    # Plot predictions
    axes[1, 0].bar(['Actual', 'MLP Model'], 
                  [count_r, mlp_pred[0]])
    axes[1, 0].set_title('Original Graph 4-Clique Count Predictions')
    axes[1, 0].set_ylim(0, max(count_r, mlp_pred[0]) + 1)
    
    axes[1, 1].bar(['Actual', 'MLP Model'], 
                  [count_b, mlp_pred[1]])
    axes[1, 1].set_title('Complement Graph 4-Clique Count Predictions')
    axes[1, 1].set_ylim(0, max(count_b, mlp_pred[1]) + 1)
    
    plt.suptitle(f"Analysis of {graph_name}", fontsize=16)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{graph_name.replace(' ', '_')}.png"))
    plt.close()
    
    # Return results
    results = {
        "graph_name": graph_name,
        "actual": {
            "original": count_r,
            "complement": count_b
        },
        "mlp_model": {
            "original": float(mlp_pred[0]),
            "complement": float(mlp_pred[1]),
            "error_original": float(abs(mlp_pred[0] - count_r)),
            "error_complement": float(abs(mlp_pred[1] - count_b))
        }
    }
    
    return results

def main():
    args = parse_args()
    
    # Load models
    custom_model, mlp_model = load_models(
        args.custom_model_path, 
        args.mlp_model_path,
        args.n_vertices
    )
    models = (custom_model, mlp_model)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create special graphs
    graphs = create_special_graphs(args.n_vertices)
    
    # Analyze each graph
    results = []
    print(f"Analyzing {len(graphs)} specific graph examples...")
    
    for graph_name, adjacency_matrix in graphs:
        print(f"Analyzing {graph_name}...")
        result = analyze_graph(
            graph_name, 
            adjacency_matrix, 
            models, 
            args.n_vertices, 
            args.output_dir
        )
        results.append(result)
    
    # Plot overall performance comparison
    mlp_errors_original = [r["mlp_model"]["error_original"] for r in results]
    mlp_errors_complement = [r["mlp_model"]["error_complement"] for r in results]
    
    graph_names = [r["graph_name"] for r in results]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    x = np.arange(len(graph_names))
    width = 0.35
    
    axes[0].bar(x - width/2, mlp_errors_original, width, label='MLP Model')
    axes[0].set_ylabel('Absolute Error')
    axes[0].set_title('Original Graph 4-Clique Count Prediction Error')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(graph_names, rotation=45, ha='right')
    axes[0].legend()
    
    axes[1].bar(x - width/2, mlp_errors_complement, width, label='MLP Model')
    axes[1].set_ylabel('Absolute Error')
    axes[1].set_title('Complement Graph 4-Clique Count Prediction Error')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(graph_names, rotation=45, ha='right')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "model_comparison.png"))
    plt.close()
    
    # Print summary
    print("\nAnalysis Summary:")
    for result in results:
        print(f"\n{result['graph_name']}:")
        print(f"  Actual Clique Sizes: Original={result['actual']['original']}, Complement={result['actual']['complement']}")
        print(f"  MLP Model: Original={result['mlp_model']['original']:.2f}, Complement={result['mlp_model']['complement']:.2f}")
    
    print(f"\nAll results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 