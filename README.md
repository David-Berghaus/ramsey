# Ramsey Number Analysis with Graph Clique Prediction

This project implements two machine learning models for predicting the maximum clique sizes in graphs and their complements. These predictions are useful for exploring properties of Ramsey numbers, specifically R(4,4).

## Overview

The Ramsey number R(4,4) relates to finding graphs that don't contain either a 4-clique or a 4-independent set (equivalent to a 4-clique in the complement). This implementation provides:

1. A custom architecture adapted from an existing reinforcement learning implementation
2. A simple MLP baseline model for comparison

Both models take adjacency matrices of graphs with 17 vertices as input and output two values:
- Maximum clique size in the original graph
- Maximum clique size in the graph's complement

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ramsey

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Comparison Experiment

To run the full experiment comparing both models:

```bash
python src/run_clique_prediction.py
```

This will:
1. Generate a dataset of random graphs
2. Train both the custom and MLP models
3. Evaluate their performance
4. Visualize the results
5. Save everything to a results directory

### Training Individual Models

To train only the custom model:

```bash
python src/run_clique_prediction.py --custom_only
```

To train only the MLP model:

```bash
python src/run_clique_prediction.py --mlp_only
```

### Analyzing Specific Graph Examples

Once you have trained models, you can analyze their performance on specific Ramsey graph examples:

```bash
python src/analyze_ramsey_examples.py --custom_model_path results/custom_model_best.pt --mlp_model_path results/mlp_model_best.pt
```

This will:
1. Generate several special graph examples (including graphs with 4-cliques, 4-independent sets, etc.)
2. Analyze how well each model predicts the maximum clique sizes
3. Create visualizations of the graphs with highlighted cliques
4. Compare model performance across different graph types
5. Save results to the specified output directory

### Additional Options

```
  --n_samples N_SAMPLES   Number of graph samples to generate
  --n_vertices N_VERTICES Number of vertices in each graph
  --batch_size BATCH_SIZE Training batch size
  --epochs EPOCHS         Number of training epochs
  --lr LR                 Learning rate
  --output_dir OUTPUT_DIR Directory to save results
  --seed SEED             Random seed
  --measure_time          Measure prediction time
  --features_dim FEATURES_DIM
                        Feature dimension for custom model
```

## Example

```bash
# Generate a larger dataset and train for more epochs
python src/run_clique_prediction.py --n_samples 10000 --epochs 50 --measure_time

# Then analyze specific examples with the trained models
python src/analyze_ramsey_examples.py --custom_model_path results/clique_prediction_YYYYMMDD_HHMMSS/custom_model_best.pt --mlp_model_path results/clique_prediction_YYYYMMDD_HHMMSS/mlp_model_best.pt --output_dir ramsey_analysis
```

## Implementation Details

### Custom Model

The custom model leverages a graph-based architecture with:
- Node embeddings
- Clique attention mechanisms
- Graph convolution operations

This architecture was originally designed for reinforcement learning but has been adapted for supervised regression.

### MLP Baseline

The MLP model is a simple feedforward neural network that:
- Takes a flattened adjacency matrix as input
- Uses multiple hidden layers with ReLU activations
- Outputs two values for the maximum clique sizes

## Results

The training produces several visualizations and metrics:
- Prediction vs. actual plots for both models
- Training curves showing convergence
- Performance metrics (MSE, MAE)
- Analysis on special Ramsey graphs

All results are saved to the specified output directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.