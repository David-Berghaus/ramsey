# Ramsey Number Analysis with Graph Clique Prediction

This project implements three machine learning models for predicting the maximum clique sizes in graphs and their complements. These predictions are useful for exploring properties of Ramsey numbers, specifically R(4,4).

## Overview

The Ramsey number R(4,4) relates to finding graphs that don't contain either a 4-clique or a 4-independent set (equivalent to a 4-clique in the complement). This implementation provides:

1. A custom architecture adapted from an existing reinforcement learning implementation
2. A simple MLP baseline model for comparison
3. An optimized Ramsey Graph GNN with Clique Attention model that combines traditional GNN layers with clique attention mechanisms

All models take adjacency matrices of graphs with 17 vertices as input and output two values:
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

To run the full experiment comparing all models:

```bash
python src/run_clique_prediction.py
```

This will:
1. Generate a dataset of random graphs
2. Train all three models (Custom, MLP, and Ramsey GNN with Clique Attention)
3. Evaluate their performance
4. Visualize the results
5. Save everything to a structured results directory

### Training Individual Models

To train only the custom model:

```bash
python src/run_clique_prediction.py --custom_only
```

To train only the MLP model:

```bash
python src/run_clique_prediction.py --mlp_only
```

To train only the Ramsey GNN with Clique Attention model:

```bash
python src/run_clique_prediction.py --ramsey_gnn_only
```

### Comparing Trained Models

After training multiple models, you can compare their performance using:

```bash
python src/compare_models.py
```

This will:
1. Find the latest training runs for each model type
2. Load their metrics, training curves, and predictions
3. Create comparative visualizations
4. Save everything to a comparison results directory

Additional options for comparison:

```
  --results_dir RESULTS_DIR   Base directory containing model results
  --output_dir OUTPUT_DIR     Directory to save comparison results
  --test_set_path TEST_SET    Path to a test set for evaluation (optional)
  --batch_size BATCH_SIZE     Batch size for evaluation
  --no_gpu                    Disable GPU usage even if available
```

### Additional Training Options

```
  --n_samples N_SAMPLES       Number of graph samples to generate
  --batch_size BATCH_SIZE     Batch size for training
  --hidden_dim HIDDEN_DIM     Hidden dimension for neural network layers
  --num_layers NUM_LAYERS     Number of layers in the GNN
  --clique_attention_context CONTEXT
                             Context length for clique attention mechanism
  --epochs EPOCHS             Number of training epochs
  --lr LR                     Learning rate
  --no_gpu                    Disable GPU usage even if available
  --output_dir OUTPUT_DIR     Directory to save results
```

## Example

```bash
# Generate a larger dataset and train for more epochs
python src/run_clique_prediction.py --n_samples 500 --epochs 20 --hidden_dim 64

# Train only the Ramsey GNN model with custom parameters
python src/run_clique_prediction.py --ramsey_gnn_only --n_samples 200 --epochs 10 --hidden_dim 32 --clique_attention_context 10

# Compare all previously trained models
python src/compare_models.py --output_dir comparison_results
```

## Implementation Details

### Custom Model

The custom model leverages a graph-based architecture with:
- Node embeddings
- Attention mechanisms
- Simple graph operations

This architecture was originally designed for reinforcement learning but has been adapted for supervised regression.

### MLP Baseline

The MLP model is a simple feedforward neural network that:
- Takes a flattened adjacency matrix as input
- Uses multiple hidden layers with ReLU activations
- Outputs two values for the maximum clique sizes

### Ramsey Graph GNN with Clique Attention

This optimized model combines traditional GNN architecture with clique-specific attention mechanisms:

- **Node Feature Extraction**: Efficiently extracts important graph features
- **Graph Neural Network Layers**: Propagates information through the graph structure using message passing
- **Clique Attention**: Applies attention mechanisms to cliques found in the graph and its complement
- **Hybrid Architecture**: Combines both GNN embeddings and clique attention embeddings for final predictions

## Results Storage

Each training run now stores results in a structured directory format:

```
results/
│
├── mlp_model/                  # Model-specific directories
│   └── YYYYMMDD_HHMMSS/        # Timestamp of training run
│       ├── model_weights/      # Saved model weights
│       │   └── MLP_Model_best.pt
│       ├── plots/              # Visualizations
│       │   ├── MLP_Model_training_curve.png
│       │   └── MLP_Model_predictions.png
│       ├── metrics/            # Performance metrics
│       │   ├── MLP_Model_metrics.json
│       │   └── MLP_Model_losses.json
│       └── tensorboard/        # Tensorboard logs
│
├── custom_model/
│   └── ...
│
├── ramsey_gnn/
│   └── ...
│
└── comparison_YYYYMMDD_HHMMSS/ # Comparative run results
    ├── plots/
    │   ├── training_curves_comparison.png
    │   ├── model_performance_comparison.png
    │   └── ...
    └── metrics/
        └── model_comparison.json
```

This organization makes it easy to:
- Track multiple runs of the same model
- Compare different models
- Visualize training progress
- Analyze model performance

## TensorBoard Integration

The project now includes TensorBoard support for monitoring training:

```bash
# After training, start TensorBoard
tensorboard --logdir results/model_type/timestamp/tensorboard
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.