# Made by: Giuseppe PAOLO
# Email: giuseppe.paolo@huawei.com
# Date: 1/30/23

import numpy as np
from core.controllers import FFNeuralController

try:
  import ant_maze
except:
  print("Ant maze not installed")


def input_formatter(t, obs):
  """
  This function formats the data to give as input to the controller
  :param t:
  :param obs:
  :return:
  """
  return obs  # In the first 2 positions there is the (x,y) pose of the robot


def output_formatter(action):
  """
  This function formats the output of the controller to extract the action for the env
  :param action:
  :return:
  """
  return action


def traj_to_obs(traj):
  """
  Extracts the observation part from the whole trajectory
  Use the x-y position of the robot as observation
  :param traj:
  :return:
  """
  return np.array([(np.array(t[3]['bc'])-5.)/70. for t in traj[1:]])


def gt_bd(traj, max_steps, ts=1):
  """
  The ground truth BD that we use to measure exploration
  """
  if ts == 1:
    index = -1 # If the trajectory is shorted, consider it as having continued withouth changes
  else:
    index = int(max_steps * ts)
    if index >= len(traj): index = -1 # If the trajectory is shorted, consider it as having continued withouth changes
  obs = traj[index][0][:2]
  return obs


environment = {
    'gym_name': 'AntObstacles-v0',
    'controller': {
        'controller': FFNeuralController,
        'input_formatter': input_formatter,
        'output_formatter': output_formatter,
        'input_size': 29,
        'output_size': 8,
        'hidden_layers': 3,
        'hidden_layer_size': 10,
        'name': 'neural'
    },
    'traj_to_obs': traj_to_obs,
    'gt_bd': gt_bd,
    'max_steps': 1000,
    'grid': {
        'min_coord': [-.5, -.5],
        'max_coord': [.5, .5],
        'bins': 50
    }
}
