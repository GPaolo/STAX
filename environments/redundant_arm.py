# Made by: Giuseppe PAOLO
# Email: giuseppe.paolo@huawei.com
# Date: 1/30/23

from core.controllers import FFNeuralController
import numpy as np

try:
  import gym_redarm
except:
  print("Gym redundant arm not installed")


def traj_to_obs(traj):
  """
  Get the observations taken from the traj
  :param traj: list containing gym [obs, reward, done, info]
  :return:
  """
  return np.array([t[3]['End effector pose'] for t in traj[1:]])


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


def gt_bd(traj, max_steps, ts=1):
  """
  	Computes the behavior descriptor from a trajectory.
  	:param traj: gym env list of observations
  	:param info: gym env list of info
    :param max_steps: Maximum number of steps the traj can have
    :param ts: percentage of the traj len at which the BD is extracted. In the range [0, 1]. Default: 1
    :return:
  	"""
  if ts == 1:
    index = -1  # If the trajectory is shorted, consider it as having continued withouth changes
  else:
    index = int(max_steps * ts)
    if index >= len(traj): index = -1  # If the trajectory is shorted, consider it as having continued withouth changes
  obs = traj[index][3]['End effector pose']
  return obs


environment = {
  'gym_name': 'RedundantArm-v0',
  'controller': {
    'controller': FFNeuralController,
    'input_formatter': input_formatter,
    'output_formatter': output_formatter,
    'input_size': 20,
    'output_size': 20,
    'hidden_layers': 2,
    'name': 'neural'
  },
  'traj_to_obs': traj_to_obs,
  'gt_bd': gt_bd,
  'max_steps': 100,
  'grid': {
    'min_coord': [0, 0],
    'max_coord': [1, 1],
    'bins': 50
  }
}
