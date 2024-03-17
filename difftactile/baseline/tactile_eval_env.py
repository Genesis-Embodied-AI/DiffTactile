""" 
    tactile_env.py
    Implemented by Elgce August, 2023
    TactileEnv class inherited from gym.Env
    Wrap implemented Contact_models like Surface_follow
    For use of stable-baselines3
"""
from difftactile.baseline.tactile_env import TactileEnv

##################################################################
# implement TactileEvalEnv, 
# inherit from TactileEnv 
# for eval usage in rl algos
##################################################################
class TactileEvalEnv(TactileEnv):
    def __init__(self, use_state, use_tactile, obs_tactile, contact, dt, total_steps, sub_steps, obj, time_limit, n_actions, n_observations, task_name=None):
        super().__init__(use_state, use_tactile, obs_tactile, contact, dt, total_steps, sub_steps, obj, time_limit, n_actions, n_observations, task_name)
    
    def compute_rewards(self, obs):
        """_summary_
        Args:
            obs (array from self.get_observations): tactile image
        Returns:
            rewards : taichi env's -loss
        """
        initial_loss = self.contact_model.loss[None]
        self.contact_model.compute_loss(self.time_step)
        delta_loss = self.contact_model.loss[None] - initial_loss
        return -delta_loss

    def get_metric_time(self):
        return self.contact_model.angle[self.contact_model.total_steps - 2]
    