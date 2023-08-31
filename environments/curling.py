# Made by: Giuseppe PAOLO
# Email: giuseppe.paolo@huawei.com
# Date: 1/30/23
from core.controllers import FFNeuralController
import numpy as np

try: import gym_billiard
except: print('Gym billiard not installed')

def input_formatter(t, obs):
  """
  This function formats the data to give as input to the controller
  :param t:
  :param obs:
  :return: The time
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
  Use the xy position of the ball as observations for the bd
  :param traj:
  :return:
  """
  return np.array([t[0][:2] for t in traj]) # t[0] selects the observation part

def gt_bd(traj, max_steps, ts=1):
  if ts == 1:
    index = -1 # If the trajectory is shorted, consider it as having continued withouth changes
  else:
    index = int(max_steps * ts)
    if index >= len(traj): index = -1 # If the trajectory is shorted, consider it as having continued withouth changes
  obs = traj[index][0][:2]
  return obs


environment = {
  'gym_name': 'Curling-v0',
  'controller': {
    'controller': FFNeuralController,
    'input_formatter': input_formatter,
    'output_formatter': output_formatter,
    'input_size': 6,
    'output_size': 2,
    'hidden_layers': 3,
    'name': 'neural',
  },
  'traj_to_obs': traj_to_obs,
  'gt_bd': gt_bd,
  'max_steps': 500,
  'grid':{
    'min_coord':[-1.35,-1.35],
    'max_coord':[1.35, 1.35],
    'bins':50
  }
}
