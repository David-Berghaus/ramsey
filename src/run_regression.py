import argparse
import torch
import os
from clique_regression import train_regression_model, evaluate_model, CliqueRegressionModel

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate a regression model for predicting clique counts')
    parser.add_argument('--n', type=int, default=17, help='Number of nodes in the graph')
    parser.add_argument('--r', type=int, default=4, help='Size of cliques to count in the graph')
    parser.add_argument('--b', type=int, default=4, help='Size of cliques to count in the complementary graph')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of random graph samples to generate')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate a trained model (requires --model_path)')
    parser.add_argument('--model_path', type=str, help='Path to a trained model for evaluation')
    parser.add_argument('--eval_samples', type=int, default=100, help='Number of samples for evaluation')
    
    args = parser.parse_args()
    
    # Check if only evaluation is required
    if args.eval_only:
        if not args.model_path:
            print("Error: --model_path is required when using --eval_only")
            return
        
        # Load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CliqueRegressionModel(n=args.n, r=args.r, b=args.b)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        
        # Evaluate the model
        mse, (actual_rs, pred_rs, actual_bs, pred_bs) = evaluate_model(
            model,
            n=args.n,
            r=args.r,
            b=args.b,
            num_samples=args.eval_samples
        )
        
        print(f"Model evaluation MSE: {mse}")
    else:
        # Train and evaluate the model
        trained_model, train_losses, val_losses = train_regression_model(
            n=args.n, 
            r=args.r, 
            b=args.b,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            num_samples=args.num_samples
        )
        
        # Evaluate the model
        mse, (actual_rs, pred_rs, actual_bs, pred_bs) = evaluate_model(
            trained_model,
            n=args.n,
            r=args.r,
            b=args.b,
            num_samples=args.eval_samples
        )
        
        print(f"Final MSE: {mse}")

if __name__ == "__main__":
    main() 