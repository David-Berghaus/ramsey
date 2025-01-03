from datetime import datetime, timedelta
from stable_baselines3 import A2C, PPO, TD3, SAC, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from multiprocessing import Pool
import torch
import os
import numpy as np

from env import AdjacencyMatrixFlippingEnv


def train_model(model_id, n=35, lr=5e-5, policy="MlpPolicy", algorithm="PPO", torch_num_threads=8, iteration_training_steps=100000, model_path=None):
    base_dir = "data/"
    time_stamp = datetime.now().strftime("%d_%m_%Y__%H_%M_%S") # Unique timestamp for each model
    base_path = base_dir + str(n) + "/" + "/" + algorithm + "/" + time_stamp + "/"
    os.makedirs(base_path, exist_ok=True)
    log_path = base_path + "log/"
    os.makedirs(log_path, exist_ok=True)
    new_logger = configure(log_path, ["tensorboard", "csv"])

    # Ensure different random seeds
    seed = model_id + int(datetime.now().timestamp())
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create the environment and pass the seed if possible
    E = AdjacencyMatrixFlippingEnv(n, dir=base_path, model_id=model_id, logger=new_logger)
    E = Monitor(E)

    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256])) # Updated format

    model = get_model(algorithm, model_path, E, lr=lr, policy=policy, policy_kwargs=policy_kwargs)
    
    model.set_logger(new_logger)
    torch.set_num_threads(torch_num_threads)

    iteration_count = 0
    while True:
        model.learn(iteration_training_steps)
        model.save(base_path + f"model_{model_id}_{iteration_count}.zip")
        iteration_count += 1

def get_model(algorithm, model_path, E, lr=5e-4, policy="MlpPolicy", policy_kwargs=None):
    if algorithm == "PPO":
        if model_path is not None and os.path.exists(model_path):
            print("Loading model from", model_path)
            model = PPO.load(model_path, env=E, learning_rate=lr, verbose=1, policy_kwargs=policy_kwargs)
        else:
            model = PPO(policy, E, learning_rate=lr, verbose=1, policy_kwargs=policy_kwargs)
    elif algorithm == "A2C":
        model = A2C(policy, E, learning_rate=lr, verbose=1, policy_kwargs=policy_kwargs)
    elif algorithm == "SAC":
        model = SAC(policy, E, learning_rate=lr, verbose=1)
    elif algorithm == "DQN":
        model = DQN(policy, E, learning_rate=lr, verbose=1)
    return model

if __name__ == "__main__":
    # N = 1  # Number of models to train in parallel
    # with Pool(N) as p:
    #     p.map(train_model, range(N))
    train_model(0)