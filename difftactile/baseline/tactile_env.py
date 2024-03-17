""" 
    tactile_env.py
    Implemented by Elgce August, 2023
    TactileEnv class inherited from gym.Env
    Wrap implemented Contact_models like Surface_follow
    For use of stable-baselines3
"""
import os
import random
import math
import numpy as np
import taichi as ti
import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
from stable_baselines3.common.env_checker import check_env

##################################################################
# implement TactileEnv, 
# a wrapper of implemented taichi contact classes
# for use of stable_baselines based DRL training
##################################################################
class TactileEnv(gym.Env):
    """_TactileEnv_
    A class implemented for RL training of current tasks
    init with implemented contact_model
        required ti.kernel of contact_model:
        (since it's all Python scope here, all these should not be ti.func)
            a. (Python) prepare_env: including any code between __init__ and ts loop
            b. (Python) apply_action: input action, apply such action in taichi env, end with self.memory_to_cache
            c. (Python) calculate_force: do whatever needed to calculate force of mpm and fem objects
            d. (Python) compute_loss: run all funcs to get rwd in taichi env
        required fields and vectors of contact_model:
            a. ()
            b. ()
    PPO related parameters:
        a. obs: use the image of tactile info currently
        b. rewards:
    NOTE: this version, we use CNN (obs: images of tactile) as input
    """
    def __init__(self, use_state, use_tactile, obs_tactile, contact, dt, total_steps, sub_steps, obj, time_limit, n_actions, n_observations, task_name=None):
        super().__init__()
        self.n_actions = n_actions
        self.n_observations = n_observations
        self.action_space = spaces.Box(low=-0.015, high=0.015, shape=(n_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(n_observations,), dtype=np.float32)
        self.contact_model = contact(use_state=use_state, use_tactile=use_tactile, dt=dt, total_steps=total_steps, sub_steps=sub_steps, obj=obj)
        self.contact_model.prepare_env()
        self.time_step = 0
        self.time_limit = time_limit
        self.total_rewards = 0
        self.task_name = task_name
        self.iters = []
        self.rewards = []
        downsample_num = 2 * 136 * 2
        if self.task_name == "cable_manip":
            n_particles = self.contact_model.rope_object.n_particles
        elif self.task_name == "box_open":
            n_particles = self.contact_model.mpm_object.n_particles
        elif self.task_name == "repose_obj":
            n_particles = self.contact_model.mpm_object.n_particles
        else:
            n_particles = 8000
        if n_particles < downsample_num:
            self.index = np.array(range(n_particles))
        else:
            self.index = random.sample(range(n_particles), downsample_num)
        self.model = None
        self.max_rewards = 0
        self.actions = []
        self.obs_tactile = obs_tactile
        self.reset()
        
    
    def step(self, action):
        """_summary_
        Args:
            action (tuple with 2 elements): actions got by ppo network, (d_pos, d_ori)
        Returns:
            obs: observation got in taichi env after apply action
            rewards: rewards of current observation
            dones: if the env if done, now only check the time_steps of env, may check more later
            infos: []
        """
        self.time_step += 1
        self.contact_model.apply_action(action, self.time_step)        
        self.contact_model.calculate_force(self.time_step)
        obs = self.get_observations()
        rewards = self.compute_rewards(obs=obs)
        dones = self.check_termination()
        infos = {}
        # NOTE: add truncated condition: if obs contains nan, means action is wrong
        obs_nan = np.isnan(obs).any()
        truncated = dones if not obs_nan else True
        self.actions.append(action)
        if truncated:
            if obs_nan:
                obs = np.zeros_like(obs) # otherwise, will get error for nan X float
                print("truncate at", self.time_step)
        else:
            self.total_rewards+= rewards
        return obs, rewards, dones, truncated, infos
    
    def reset(self, seed=None, options=None):
        """_summary_
        Args:
            seed (_type_, optional): _description_. Defaults to None.
            options (_type_, optional): _description_. Defaults to None.
        Returns:
            obs: observation of initial state given by self.get_observations
            infos: []
        """
        self.contact_model.init()
        self.contact_model.clear_all_grad()
        if self.task_name == "repose_obj":
            self.contact_model.clear_state_loss_grad()
        self.time_step = 0
        self.total_rewards = 0
        infos = {}
        self.contact_model.reset()
        obs = self.get_observations()
        self.actions = []
        return obs, infos
    
    def get_observations(self):
        """_summary_
        Returns:
            obs : we only use the marker points' position of fem_sensor1 here
        """
        if self.task_name == "box_open":
            sensor_matrix = self.contact_model.fem_sensor1.trans_h.to_numpy()
            sensor_matrix = np.matrix(sensor_matrix)
            trang, rot = self.extract_translation_and_euler_angles(sensor_matrix)
            obs = self.contact_model.mpm_object.x_0.to_numpy()[0][self.index].reshape(-1) # 
            obs = np.append(obs, np.array(trang)) # 3
            obs = np.append(obs, np.array(rot)) # 3
            if self.obs_tactile:
                self.contact_model.fem_sensor1.extract_markers(0)
                obs = np.append(obs, self.contact_model.fem_sensor1.predict_markers.to_numpy().reshape(-1))
                obs = np.append(obs, self.contact_model.predict_force1[None].to_numpy())
        elif self.task_name == "surface_follow":
            sensor_matrix = self.contact_model.fem_sensor1.trans_h.to_numpy()
            sensor_matrix = np.matrix(sensor_matrix)
            trang, rot = self.extract_translation_and_euler_angles(sensor_matrix)
            obs = np.array(trang) # 3
            obs = np.append(obs, np.array(rot)) # 3
            if self.obs_tactile:
                self.contact_model.fem_sensor1.extract_markers(0)
                obs = np.append(obs, self.contact_model.fem_sensor1.predict_markers.to_numpy().reshape(-1))
                obs = np.append(obs, self.contact_model.predict_force[None].to_numpy())
        elif self.task_name == "cable_manip":
            sensor_matrix = self.contact_model.gripper.fem_sensor1.trans_h.to_numpy()
            sensor_matrix = np.matrix(sensor_matrix)
            trang, rot = self.extract_translation_and_euler_angles(sensor_matrix)
            obs = self.contact_model.rope_object.pos.to_numpy()[0][self.index].reshape(-1) # 
            obs = np.append(obs, np.array(trang)) # 3
            obs = np.append(obs, np.array(rot)) # 3
            if self.obs_tactile:
                self.contact_model.gripper.fem_sensor1.extract_markers(0)
                obs = np.append(obs, self.contact_model.gripper.fem_sensor1.predict_markers.to_numpy().reshape(-1))
                obs = np.append(obs, self.contact_model.predict_force1[None].to_numpy())
        elif self.task_name == "repose_obj":
            sensor_matrix = self.contact_model.fem_sensor1.trans_h.to_numpy()
            sensor_matrix = np.matrix(sensor_matrix)
            trang, rot = self.extract_translation_and_euler_angles(sensor_matrix)
            obs = self.contact_model.mpm_object.x_0.to_numpy()[0][self.index].reshape(-1) # 
            obs = np.append(obs, np.array(trang)) # 3
            obs = np.append(obs, np.array(rot)) # 3
            if self.obs_tactile:
                self.contact_model.fem_sensor1.extract_markers(0)
                obs = np.append(obs, self.contact_model.fem_sensor1.predict_markers.to_numpy().reshape(-1))
                obs = np.append(obs, self.contact_model.predict_force1[0].to_numpy())
        return obs
    
    def extract_translation_and_euler_angles(self, matrix):
        translation = matrix[:3, 3]
        rotation_matrix = matrix[:3, :3]
        sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = math.atan2(-rotation_matrix[2, 0], sy)
            z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = math.atan2(-rotation_matrix[2, 0], sy)
            z = 0

        x_deg = math.degrees(x)
        y_deg = math.degrees(y)
        z_deg = math.degrees(z)

        return translation, (x_deg, y_deg, z_deg)
    
    def compute_rewards(self, obs):
        """_summary_
        Args:
            obs (array from self.get_observations): tactile image
        Returns:
            rewards : rewards of present obs, same as taichi env
        """
        # calculate detailed rewards --> the same as taichi env is OK
        initial_loss = self.contact_model.loss[None]
        self.contact_model.compute_loss(self.time_step)
        delta_loss = self.contact_model.loss[None] - initial_loss
        if self.task_name == "surface_follow":
            scale = 100000.0
        elif self.task_name == "cable_manip":
            scale = 1000.0
        elif self.task_name == "box_open":
            scale = 1.0
        elif self.task_name == "repose_obj":
            scale = 100.0
        else:
            raise NotImplementedError
        rewards = scale * 2000 - delta_loss
        return rewards
    
    def check_termination(self):
        if self.time_step >= self.time_limit:
            return True
        # TODO: if need more terminate condition, add here
        return False
    
    def render(self, gui1, gui2, gui3):
        # TODO: if needed, render env image here
        self.contact_model.render(gui1, gui2, gui3)
    
    def close(self):
        pass
    
    def render_rewards(self):
        """_summary_
            according to self.rewards & self.iters, render rewards curve
        """
        plt.figure(figsize=(20, 12))
        plt.plot(self.iters, self.rewards, marker='o')
        plt.title("mean rewards for " + self.task_name) # NOTE: change titles to what you need
        plt.xlabel("training iter")
        plt.ylabel("total reward")
        plt.savefig("checkpoints/" + self.task_name + "/" +"iter_reward.png")
##################################################################
