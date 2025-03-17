# Ramsey Number Analysis with Graph Clique Prediction

This project implements machine learning models for predicting the maximum clique sizes in graphs and their complements, as well as a reinforcement learning approach to explore the properties of Ramsey numbers, specifically R(4,4).

## Overview

The Ramsey number R(4,4) relates to finding graphs that don't contain either a 4-clique or a 4-independent set (equivalent to a 4-clique in the complement). This implementation provides:

1. A reinforcement learning approach using GNN with clique attention for edge scoring
2. A simple MLP baseline model for supervised learning
3. An optimized Ramsey Graph GNN with Clique Attention model for supervised learning

All supervised models take adjacency matrices of graphs with 17 vertices as input and output two values:
- Maximum clique size in the original graph
- Maximum clique size in the graph's complement

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ramsey

# Install dependencies
pip install -r requirements.txt

# Activate the Ramsey environment (if using conda)
conda activate ramsey
```

## Usage

### Running the Reinforcement Learning Model

The reinforcement learning model can be run to explore the Ramsey graph space using the `main.py` script:

```bash
python src/main.py [OPTIONS]
```

Available options:

```
Basic parameters:
  --model_id MODEL_ID       Identifier for this model run (default: 0)
  --algorithm {PPO,A2C}     RL algorithm to use (default: PPO)
  --lr LR                   Learning rate (default: 1e-4)

Environment parameters:
  --num_envs NUM_ENVS       Number of parallel environments (default: 256)
  --steps_per_iter STEPS    Number of steps per training iteration (default: 1)

Model parameters:
  --features_dim DIM        Hidden dimension for neural network layers (default: 64)
  --num_layers LAYERS       Number of layers in the GNN (default: 3)
  --clique_attention_context LEN 
                            Context length for clique attention mechanism (default: 64)
  --node_attention_context LEN
                            Context length for node attention mechanism (default: 8)
  --num_heads HEADS         Number of attention heads (default: 2)

Execution options:
  --num_threads THREADS     Number of PyTorch threads (default: 1)
  --save_interval INTERVAL  Save model every N iterations (default: 1000)
  --base_dir DIR            Base directory for data storage (default: 'data/')
  --model_path PATH         Path to an existing model to continue training
```

Examples:

```bash
# Train with default parameters
python src/main.py

# Use a different algorithm and learning rate
python src/main.py --algorithm A2C --lr 5e-5

# Customize model architecture
python src/main.py --features_dim 128 --num_layers 4 --clique_attention_context 32

# Continue training from an existing model
python src/main.py --model_path data/17/PPO/0.0001/previous_model.zip
```

To stop training, press Ctrl+C, and the model will save a final checkpoint before exiting.

### Running the Clique Comparison Experiment

To run the full experiment comparing the supervised models:

```bash
python src/run_clique_prediction.py
```

This will:
1. Generate a dataset of random graphs
2. Train both models (MLP and Ramsey GNN with Clique Attention)
3. Evaluate their performance
4. Visualize the results
5. Save everything to a structured results directory

### Training Individual Models

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
  --epochs EPOCHS             Maximum number of training epochs
  --lr LR                     Learning rate
  --patience PATIENCE         Number of epochs to wait for improvement before early stopping
  --min_delta MIN_DELTA       Minimum change in validation loss to be considered as improvement
  --overfitting_threshold THRESHOLD
                             Number of consecutive epochs with improving train loss but worsening val loss
  --no_gpu                    Disable GPU usage even if available
  --output_dir OUTPUT_DIR     Directory to save results
```

### Adaptive Termination

The training process for supervised models includes an adaptive termination mechanism that can automatically stop training when:

1. **Early Stopping**: Training stops if there's no improvement in validation loss for a specified number of epochs (controlled by `--patience`).
2. **Overfitting Detection**: Training stops if the model shows signs of overfitting for several consecutive epochs (controlled by `--overfitting_threshold`).
3. **Convergence Detection**: Training stops when the validation loss stabilizes, indicating the model has converged.

This eliminates the need to guess the optimal number of epochs for each model. The training will continue until one of these conditions is met or until the maximum number of epochs is reached.

Example usage with adaptive termination parameters:

```bash
# Train with custom adaptive termination settings
python src/run_clique_prediction.py --patience 10 --min_delta 0.0005 --overfitting_threshold 5
```

## Examples

```bash
# Train the RL model with more parallel environments
python src/main.py --num_envs 512 --features_dim 128

# Generate a larger dataset and train supervised models for more epochs
python src/run_clique_prediction.py --n_samples 500 --epochs 20 --hidden_dim 64

# Train only the Ramsey GNN model with custom parameters
python src/run_clique_prediction.py --ramsey_gnn_only --n_samples 200 --epochs 10 --hidden_dim 32 --clique_attention_context 10

# Compare all previously trained models
python src/compare_models.py --output_dir comparison_results
```

## Implementation Details

### Reinforcement Learning Approach

The RL model uses a GNN with clique attention to:
- Score possible edge flips in the graph
- Learn to avoid creating 4-cliques or 4-independent sets
- Explore the space of Ramsey graphs efficiently

This approach is implemented in the `RamseyGNNFeatureExtractor` and `RamseyGraphGNNEdgeScorer` classes.

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

For RL models, results are stored in the data directory:

```
data/
│
└── 17/                         # For 17-vertex graphs
    └── PPO/                    # Algorithm type
        └── 0.0001/             # Learning rate
            └── YYYYMMDD_HHMMSS/   # Timestamp of run
                ├── log/           # Tensorboard logs
                └── model_X_Y.zip  # Saved models where X is model_id and Y is iteration
```

This organization makes it easy to:
- Track multiple runs of the same model
- Compare different models
- Visualize training progress
- Analyze model performance

## TensorBoard Integration

The project includes TensorBoard support for monitoring training:

```bash
# For supervised models:
tensorboard --logdir results/model_type/timestamp/tensorboard

# For RL models:
tensorboard --logdir data/17/algorithm/learning_rate/timestamp/log
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.