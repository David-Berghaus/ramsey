# Ramsey Graph Clique Count Regression

This repository contains code for an ablation study that uses a simplified architecture inspired by the Ramsey graph generation model but trained as a regression model to predict clique counts.

## Overview

The regression model takes a graph represented as an adjacency matrix and predicts two values:
1. **r** - The count of cliques of size r in the original graph
2. **b** - The count of cliques of size b in the complementary graph

The model is inspired by the neural network architecture that was originally developed for the reinforcement learning task but has been simplified to avoid compatibility issues while keeping the core architectural ideas.

## Implementation Details

The implementation:
1. Generates random graph samples and computes their clique counts
2. Creates a dataset with adjacency matrices as inputs and clique counts as targets
3. Uses a simplified graph neural network architecture with similar principles to the original model
4. Trains using MSE loss to predict clique counts accurately

### Architecture

The simplified architecture includes:
1. A node-level encoder that computes node features based on local graph structure
2. A graph-level encoder that aggregates node features through pooling
3. A regression head that predicts the clique counts (r, b) from the graph representation

This preserves the key ideas of the original architecture while being more tractable for the regression task.

## Files

- `src/clique_regression.py`: Contains the implementation of the regression model, dataset, and training/evaluation functions
- `src/run_regression.py`: Script to run the training and evaluation from the command line
- `README_regression.md`: This file, explaining the regression model implementation

## Usage

### Training a New Model

To train a regression model with default parameters:

```bash
python src/run_regression.py
```

This will:
1. Generate 5000 random graph samples with n=17 nodes
2. Train a model to predict the counts of r=4 cliques in the graph and b=4 cliques in the complement
3. Save the model and training curves in the data/regression directory
4. Evaluate the model on 100 random samples

### Customizing Training

You can customize the training with various parameters:

```bash
python src/run_regression.py --n 15 --r 3 --b 3 --batch_size 64 --num_epochs 100 --learning_rate 1e-3 --num_samples 10000
```

### Evaluating a Pre-trained Model

To evaluate a pre-trained model:

```bash
python src/run_regression.py --eval_only --model_path data/regression/17_4_4/DD_MM_YYYY__HH_MM_SS/model_final.pt
```

## Parameters

- `--n`: Number of nodes in the graph (default: 17)
- `--r`: Size of cliques to count in the graph (default: 4)
- `--b`: Size of cliques to count in the complementary graph (default: 4)
- `--batch_size`: Batch size for training (default: 32)
- `--num_epochs`: Number of epochs for training (default: 50)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--num_samples`: Number of random graph samples to generate (default: 5000)
- `--eval_only`: Only evaluate a trained model (requires --model_path)
- `--model_path`: Path to a trained model for evaluation
- `--eval_samples`: Number of samples for evaluation (default: 100)

## Ablation Study Notes

This implementation serves as an ablation study to:
1. Assess how well a neural network can learn to predict clique counts directly from graph structure
2. Compare this supervised learning approach with the reinforcement learning approach
3. Evaluate whether the learned representations capture meaningful graph properties

By comparing the performance of this regression model with the original RL approach, we can better understand which architectural components contribute most to learning graph properties related to Ramsey numbers. 