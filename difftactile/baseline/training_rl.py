""" 
    multi_env.py
    Implemented August, 2023
    Use implemented class TactileEnv to train ppo & sac policy
    For use of stable-baselines3
"""

# pypi
import os
import argparse
import numpy as np
import taichi as ti

# sb3-related
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# self-defined
from difftactile.baseline.tactile_env import TactileEnv
from difftactile.baseline.tactile_eval_env import TactileEvalEnv
from register_tasks import ContactTask, get_tasks

##################################################################
# NOTE: if you only want to use some of available gpu, set it here
# os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
##################################################################

##################################################################
# A class to use env wrapper to train taichi tasks
# num_envs: number of envs for multi-processing training
# num_eval_envs: number of envs to evaluate trained policy
# training_iterations: number to iterations to train the policy
# resume: whether or not to 
# num_total_step: num of total_step, maximum step length
##################################################################
class RL_Trainer():
    def __init__(self, num_envs, num_eval_envs, task: ContactTask, training_iterations, resume, off_screen, algo, obs, loss):
        self.off_screen = off_screen
        if self.off_screen:
            os.environ["PYTHON_PLATFORM"] = 'egl'
        self.num_envs = num_envs
        self.num_eval_envs = num_eval_envs
        self.task = task
        self.training_iterations = training_iterations
        self.resume = resume
        self.eval_rwd = 0
        use_state = True

        if obs == "state":
            n_observation = self.task.state_n_observations
            obs_tactile = False
        elif obs == "tactile":
            n_observation = self.task.tactile_n_observations
            obs_tactile = True

        if loss == "state":
            use_tactile = False
        else:
            use_tactile = True

        if algo == "PPO":
            self.algo = PPO
        elif algo == "SAC":
            self.algo = SAC
        else:
            raise NotImplementedError("Error: Algorithm type not implemented")
        self.algo_name = algo
        self.env = TactileEvalEnv(
            use_state       =   use_state, 
            use_tactile     =   use_tactile, 
            obs_tactile     =   obs_tactile, 
            contact         =   task.contact_model, 
            dt              =   task.dt, 
            total_steps     =   task.total_steps,
            sub_steps       =   task.sub_steps, 
            obj             =   task.obj, 
            time_limit      =   task.total_steps,
            n_actions       =   task.n_actions, 
            n_observations  =   n_observation, 
            task_name       =   task.task_name)
        self.env_kwargs = {
            "use_state"     :   use_state, 
            "use_tactile"   :   use_tactile, 
            "obs_tactile"   :   obs_tactile, 
            "contact"       :   task.contact_model, 
            "dt"            :   task.dt, 
            "total_steps"   :   task.total_steps, 
            "sub_steps"     :   task.sub_steps, 
            "obj"           :   task.obj,
            "time_limit"    :   task.total_steps, 
            "n_actions"     :   task.n_actions, 
            "n_observations":   n_observation, 
            "task_name"     :   task.task_name}
        self.vec_env = make_vec_env(TactileEnv, n_envs=self.num_envs, env_kwargs=self.env_kwargs)
        self.eval_env = make_vec_env(TactileEvalEnv, n_envs=self.num_eval_envs, env_kwargs=self.env_kwargs)
        self.checkpoint_path = f"RL_checkpoints/{self.task.task_name}/rl/obs_force_loss_force_{str(self.algo_name)}/"                      
    
    # use multiprocessing ppo env to train for self.training_iterations
    def train(self):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        if not self.resume:
            self.model = self.algo("MlpPolicy", self.vec_env, verbose=1, tensorboard_log=self.checkpoint_path)
        else:
            self.load_model()
        self.eval_callback = EvalCallback(eval_env=self.eval_env, best_model_save_path=self.checkpoint_path,
                                          log_path=self.checkpoint_path, eval_freq=self.task.total_steps * 10)
        self.model.learn(total_timesteps=self.training_iterations * self.task.total_steps, \
            callback=self.eval_callback)
        print("================= Training process has finished! =================")
    
    def load_model(self):
        try:
            self.model = self.algo.load(self.checkpoint_path + "best_model")
        except FileNotFoundError as e:
            print(f"Model file does not exist!: {e}")

##################################################################

##################################################################
# example application of implemented PPO algorithm
# Currently use contact_grasping_elastic as an example
##################################################################
if __name__ == "__main__":
    ti.init(arch=ti.cuda, device_memory_GB = 12)
    # parse arguments for usage of sb-3 based ppo & sac training
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_envs", type=int, default=4,    help="num of envs to train policy")
    parser.add_argument("--num_eval_envs",  type=int, default=1,    help="num of envs to eval policy")
    parser.add_argument("--max_iteration",  type=int, default=200,  help="max iterations for training")
    parser.add_argument("--resume",         action="store_true",    help="whether or not resume best_model")
    parser.add_argument("--off_screen",     action="store_true",    help="whether or not use off screen mode")
    parser.add_argument("--task",           default="box_open",     help="task to train")
    parser.add_argument("--algorithm",      default="PPO",          help="algorithm for training, SAC or PPO") # 1 means PPO, 2 means SAC
    parser.add_argument("--obs",            default="state",        help="whether or not to use tactile in observations, state or tactile")
    parser.add_argument("--loss",           default="state",        help="whether or not to use tactile in loss, state or tactile")
    args = parser.parse_args()
            
    # parse args and run given trainings or evaluations
    # choose task
    task = get_tasks(args.task)
    if task == None:
        raise NotImplementedError
    
    # choose train or eval
    runner = RL_Trainer(
            task                =   task, 
            algo                =   args.algorithm,
            obs                 =   args.obs,
            loss                =   args.loss,
            resume              =   args.resume, 
            off_screen          =   args.off_screen, 
            num_envs            =   args.num_train_envs, 
            num_eval_envs       =   args.num_eval_envs, 
            training_iterations =   args.max_iteration
            )
    runner.train()
##################################################################