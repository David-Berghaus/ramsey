"""
Test script to verify the visualization functionality.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from clique_prediction_models import visualize_results

def test_visualization():
    """Test the visualization function with some synthetic data."""
    # Create a plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Create some synthetic data
    np.random.seed(42)
    n_samples = 100
    
    # Case 1: Perfect diagonal predictions
    actual_1 = np.random.randint(0, 10, size=(n_samples, 2))
    predictions_1 = actual_1 + np.random.normal(0, 0.5, size=(n_samples, 2))
    
    # Case 2: Random predictions
    actual_2 = np.random.randint(0, 10, size=(n_samples, 2))
    predictions_2 = np.random.randint(0, 10, size=(n_samples, 2))
    
    # Case 3: All the same value
    actual_3 = np.ones((n_samples, 2)) * 5
    predictions_3 = np.ones((n_samples, 2)) * 5.5
    
    # Test cases
    test_cases = [
        (predictions_1, actual_1, "Test_Perfect"),
        (predictions_2, actual_2, "Test_Random"),
        (predictions_3, actual_3, "Test_Constant")
    ]
    
    # Run tests
    for predictions, actual, name in test_cases:
        print(f"\nTesting {name}:")
        print(f"Predictions shape: {predictions.shape}, Actual shape: {actual.shape}")
        
        # Test with numpy arrays
        print("Testing with numpy arrays...")
        fig = visualize_results(predictions, actual, f"{name}_numpy")
        if fig:
            save_path = os.path.join('plots', f"{name}_numpy.png")
            fig.savefig(save_path)
            print(f"Plot saved to: {save_path}")
        
        # Test with torch tensors
        print("Testing with torch tensors...")
        predictions_tensor = torch.tensor(predictions)
        actual_tensor = torch.tensor(actual)
        fig = visualize_results(predictions_tensor, actual_tensor, f"{name}_tensor")
        if fig:
            save_path = os.path.join('plots', f"{name}_tensor.png")
            fig.savefig(save_path)
            print(f"Plot saved to: {save_path}")
    
    print("\nAll visualization tests completed.")

if __name__ == "__main__":
    test_visualization() 