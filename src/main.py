from datetime import datetime
from stable_baselines3 import A2C, PPO, TD3, SAC, DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
import torch
import os
import numpy as np
import logging

from env import AdjacencyMatrixFlippingEnv
from model import NodeMeanPoolCliqueAttentionFeatureExtractor
from callbacks import TensorboardCallback  # Import the custom callback

def make_env(model_id, seed, base_path, env_id):
    def _init():
        env = AdjacencyMatrixFlippingEnv(
            n=17,
            r=4,
            b=4,
            not_connected_punishment=-100,
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

def get_model(algorithm, model_path, env, lr=1e-5, policy="MlpPolicy", policy_kwargs=None, tensorboard_log=None):
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
                # max_grad_norm=0.05,
                n_steps=1,
            )
    elif algorithm == "A2C":
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


def train_model(model_id, lr=5e-5, policy="MlpPolicy", algorithm="PPO",
                  torch_num_threads=1, iteration_training_steps=1,
                  model_path=None, num_envs=2):
    base_dir = "data/"
    time_stamp = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    base_path = os.path.join(base_dir, "17", algorithm, time_stamp)
    log_path = os.path.join(base_path, "log")
    os.makedirs(log_path, exist_ok=True)

    seed = model_id + int(datetime.now().timestamp())

    # Create vectorized environments with unique env_ids
    env_fns = [make_env(model_id, seed + i, base_path, env_id=i) for i in range(num_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)  # Monitors rewards and other metrics

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[32, 32], vf=[32, 32]),
        features_extractor_class=NodeMeanPoolCliqueAttentionFeatureExtractor,
        features_extractor_kwargs=dict(
            n=17, r=4, b=4,
            not_connected_punishment=-10000,
            features_dim=64,
            num_heads=1,
            node_attention_context_len=8, 
            clique_attention_context_len=64,
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
    feature_extr: NodeMeanPoolCliqueAttentionFeatureExtractor = model.policy.features_extractor

    # convert all parameters to trainable
    for name, param in feature_extr.named_parameters():
        param.requires_grad = True

    # Initialize the custom callback
    callback = TensorboardCallback()

    iteration_count = 0
    while True:
        model.learn(
            total_timesteps=iteration_training_steps, 
            reset_num_timesteps=False, 
            callback=callback,
        )
        if iteration_count%1000 == 0:
            model.save(os.path.join(base_path, f"model_{model_id}_{iteration_count}.zip"))
        iteration_count += 1


if __name__ == "__main__":
    train_model(model_id=0)