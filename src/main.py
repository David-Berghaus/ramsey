from datetime import datetime
from stable_baselines3 import A2C, PPO, TD3, SAC, DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
import torch
import os
import numpy as np
import logging
import argparse

from env import AdjacencyMatrixFlippingEnv
from model import RamseyGNNFeatureExtractor
from callbacks import TensorboardCallback  # Import the custom callback

def make_env(model_id, seed, base_path, env_id):
    def _init():
        env = AdjacencyMatrixFlippingEnv(
            n=17,
            r=4,
            b=4,
            not_connected_punishment=-1000,
            num_local_searches_before_reset=1000,
            dir=base_path,
            model_id=model_id,
            logger=None,  # Temporarily disable logging
            env_id=env_id,
        )
        env = Monitor(env)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return env
    return _init

def get_model(algorithm, model_path, env, lr=1e-4, policy="MlpPolicy", policy_kwargs=None, tensorboard_log=None):
    if algorithm == "PPO":
        if model_path and os.path.exists(model_path):
            print("Loading model from", model_path)
            model = PPO.load(
                model_path, 
                env=env, 
                learning_rate=lr, 
                verbose=1, 
                policy_kwargs=policy_kwargs, 
                tensorboard_log=tensorboard_log,
                # max_grad_norm=0.1,
                n_steps=1,
            )
        else:
            model = PPO(
                policy, 
                env, 
                learning_rate=lr, 
                verbose=1, 
                policy_kwargs=policy_kwargs, 
                tensorboard_log=tensorboard_log,
                # max_grad_norm=0.1,
                n_steps=1,
            )
    elif algorithm == "A2C":
        if model_path and os.path.exists(model_path):
            print("Loading model from", model_path)
            model = A2C.load(
                model_path,
                env=env,
                learning_rate=lr,
                verbose=1,
                policy_kwargs=policy_kwargs,
                tensorboard_log=tensorboard_log
            )
        else:
            model = A2C(
                policy, 
                env, 
                learning_rate=lr, 
                verbose=1, 
                policy_kwargs=policy_kwargs, 
                tensorboard_log=tensorboard_log
            )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return model


def train_model(model_id, lr=1e-4, policy="MlpPolicy", algorithm="PPO",
                  torch_num_threads=1, iteration_training_steps=1,
                  model_path=None, num_envs=256, features_dim=64, 
                  num_layers=3, clique_attention_context_len=64,
                  node_attention_context_len=8, num_heads=2,
                  save_interval=1000, base_dir="data/"):
    """
    Train the Ramsey RL model with the specified parameters.
    
    Args:
        model_id (int): Identifier for the model
        lr (float): Learning rate
        policy (str): Policy type to use
        algorithm (str): RL algorithm to use (PPO or A2C)
        torch_num_threads (int): Number of threads for PyTorch
        iteration_training_steps (int): Number of steps per training iteration
        model_path (str): Path to load an existing model (None for new model)
        num_envs (int): Number of parallel environments to use
        features_dim (int): Hidden dimension for the feature extractor
        num_layers (int): Number of layers in the GNN
        clique_attention_context_len (int): Context length for clique attention
        node_attention_context_len (int): Context length for node attention
        num_heads (int): Number of attention heads
        save_interval (int): Interval for saving model checkpoints
        base_dir (str): Base directory for data storage
    """
    time_stamp = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    base_path = os.path.join(base_dir, "17", algorithm, str(lr), time_stamp)
    log_path = os.path.join(base_path, "log")
    os.makedirs(log_path, exist_ok=True)

    print(f"Training with algorithm: {algorithm}, learning rate: {lr}")
    print(f"Using {num_envs} parallel environments")
    print(f"Model files will be saved to: {base_path}")

    seed = model_id + int(datetime.now().timestamp())

    # Create vectorized environments with unique env_ids
    env_fns = [make_env(model_id, seed + i, base_path, env_id=i) for i in range(num_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)  # Monitors rewards and other metrics

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[], vf=[32, 32]),  # Empty policy network since our model does all the work
        features_extractor_class=RamseyGNNFeatureExtractor,
        features_extractor_kwargs=dict(
            n_vertices=17, r=4, b=4,
            hidden_dim=features_dim,
            num_layers=num_layers,
            clique_attention_context_len=clique_attention_context_len,
            node_attention_context_len=node_attention_context_len,
            num_heads=num_heads,
        )
    )

    # Initialize the model with TensorBoard logging
    model = get_model(
        algorithm, 
        model_path, 
        env, 
        lr=lr, 
        policy=policy, 
        policy_kwargs=policy_kwargs, 
        tensorboard_log=log_path
    )
    torch.set_num_threads(torch_num_threads)
    
    # get model feature extractor
    feature_extr: RamseyGNNFeatureExtractor = model.policy.features_extractor

    # convert all parameters to trainable
    for name, param in feature_extr.named_parameters():
        param.requires_grad = True

    # Initialize the custom callback
    callback = TensorboardCallback()

    print(f"Starting training with {iteration_training_steps} steps per iteration")
    print(f"Models will be saved every {save_interval} iterations")

    iteration_count = 0
    try:
        while True:
            model.learn(
                total_timesteps=iteration_training_steps, 
                reset_num_timesteps=False, 
                callback=callback,
            )
            if iteration_count % save_interval == 0:
                save_path = os.path.join(base_path, f"model_{model_id}_{iteration_count}.zip")
                model.save(save_path)
                print(f"Model saved to {save_path}")
            iteration_count += 1
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final model...")
        final_save_path = os.path.join(base_path, f"model_{model_id}_final.zip")
        model.save(final_save_path)
        print(f"Final model saved to {final_save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train Ramsey RL model')
    
    # Basic parameters
    parser.add_argument('--model_id', type=int, default=0, 
                        help='Identifier for this model run')
    parser.add_argument('--algorithm', type=str, choices=['PPO', 'A2C'], default='PPO',
                        help='RL algorithm to use')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    # Environment parameters
    parser.add_argument('--num_envs', type=int, default=256,
                        help='Number of parallel environments')
    parser.add_argument('--steps_per_iter', type=int, default=1,
                        help='Number of steps per training iteration')
    
    # Model parameters
    parser.add_argument('--features_dim', type=int, default=64,
                        help='Hidden dimension for neural network layers')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of layers in the GNN')
    parser.add_argument('--clique_attention_context', type=int, default=64, 
                        help='Context length for clique attention mechanism')
    parser.add_argument('--node_attention_context', type=int, default=8,
                        help='Context length for node attention mechanism')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='Number of attention heads')
    
    # Execution options
    parser.add_argument('--num_threads', type=int, default=1,
                        help='Number of PyTorch threads')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Save model every N iterations')
    parser.add_argument('--base_dir', type=str, default='data/',
                        help='Base directory for data storage')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to an existing model to continue training')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Train the model with the specified parameters
    train_model(
        model_id=args.model_id,
        lr=args.lr,
        algorithm=args.algorithm,
        torch_num_threads=args.num_threads,
        iteration_training_steps=args.steps_per_iter,
        model_path=args.model_path,
        num_envs=args.num_envs,
        features_dim=args.features_dim,
        num_layers=args.num_layers,
        clique_attention_context_len=args.clique_attention_context,
        node_attention_context_len=args.node_attention_context,
        num_heads=args.num_heads,
        save_interval=args.save_interval,
        base_dir=args.base_dir
    )

if __name__ == "__main__":
    main()