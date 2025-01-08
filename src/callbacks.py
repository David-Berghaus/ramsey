from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to TensorBoard by averaging
    across all parallel environments.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        
        # Initialize lists to collect metrics from all envs
        scores = []
        best_scores = []
        rewards = []

        # Iterate over each env's info dictionary
        for info in infos:
            if 'score' in info:
                scores.append(info['score'])
            if 'best_score' in info:
                best_scores.append(info['best_score'])
            if 'reward' in info:
                rewards.append(info['reward'])
        
        # Compute averages if metrics are available
        if scores:
            avg_score = np.mean(scores)
            score_std = np.std(scores)
            self.logger.record('custom_env/average_score', avg_score)
            self.logger.record('custom_env/score_std', score_std)
        
        if best_scores:
            avg_best_score = np.mean(best_scores)
            best_score_std = np.std(best_scores)
            self.logger.record('custom_env/average_best_score', avg_best_score)
            self.logger.record('custom_env/best_score_std', best_score_std)
        
        if rewards:
            avg_reward = np.mean(rewards)
            reward_std = np.std(rewards)
            self.logger.record('custom_env/average_reward', avg_reward)
            self.logger.record('custom_env/reward_std', reward_std)
        
        # Dump the recorded averages to TensorBoard
        self.logger.dump(self.num_timesteps)
        
        return True