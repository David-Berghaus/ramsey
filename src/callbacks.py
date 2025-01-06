from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to TensorBoard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'score' in info:
                self.logger.record('custom_env/score', info['score'])
            if 'best_score' in info:
                self.logger.record('custom_env/best_score', info['best_score'])
            if 'reward' in info:
                self.logger.record('custom_env/reward', info['reward'])
        return True