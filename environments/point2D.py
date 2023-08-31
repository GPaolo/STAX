# Made by: Giuseppe PAOLO
# Email: giuseppe.paolo@huawei.com
# Date: 1/30/23

from core.controllers import FFNeuralController
import numpy as np

try: import gym_dummy
except: print("Gym dummy not installed")

def input_formatter(t, obs):
  """
  This function formats the data to give as input to the controller
  :param t:
  :param obs:
  :return: None
  """
  return obs

def output_formatter(action):
  """
  This function formats the output of the controller to extract the action for the env
  :param action:
  :return:
  """
  return action

def traj_to_obs(traj):
  """
  Get observations from the trajectory coming from the Walker2D environment
  :param traj:
  :return:
  """
  return np.array([t[0] for t in traj]) # t[0] selects the observation part

def gt_bd(traj, max_steps, ts=1):
  """
  This function extract the ground truth BD for the dummy environment
  :param traj:
  :param max_steps: Maximum number of steps the traj can have
  :param ts: percentage of the traj len at which the BD is extracted. In the range [0, 1]. Default: 1
  :return:
  """
  if ts == 1:
    index = -1 # If the trajectory is shorted, consider it as having continued withouth changes
  else:
    index = int(max_steps * ts)
    if index >= len(traj): index = -1 # If the trajectory is shorted, consider it as having continued withouth changes
  obs = traj[index][0]
  return obs

environment = {
  'gym_name': 'Walker2D-v0',
  'controller': {
    'controller': FFNeuralController,
    'input_formatter': input_formatter,
    'output_formatter': output_formatter,
    'input_size': 2,
    'output_size': 2,
    'hidden_layers': 2,
    'name': 'dummy',
  },
  'traj_to_obs': traj_to_obs,
  'gt_bd': gt_bd,
  'max_steps': 50,
  'grid':{
    'min_coord':[-1,-1],
    'max_coord':[1, 1],
    'bins':50
  }
}