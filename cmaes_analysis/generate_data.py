# Created by Giuseppe Paolo 
# Date: 17/11/2020

import numpy as np
from core.controllers.neural_controller import FFNeuralController
import gym
import matplotlib.pyplot as plt
import pickle as pkl
from environments.environments import registered_envs
import torch
import torch.nn as nn
import torch.optim as optim


class SIG(nn.Module):
  def forward(self, tensor):
    return (torch.sigmoid(tensor) * 2) - 1

class FFNN(nn.Module):
  def __init__(self, input_size=2, output_size=2, learning_rate=0.001, **kwargs):
    """
    Constructor
    :param device: Device on which run the computation
    :param learning_rate:
    :param lr_scale: Learning rate scale for the lr scheduler
    :param encoding_shape: Shape of the feature space
    """
    super(FFNN, self).__init__()
    self.l1 = nn.Linear(input_size, 5, bias=False)
    self.l2 = nn.Linear(5, 5, bias=False)
    self.l3 = nn.Linear(5, output_size, bias=False)
    self.bias1 = nn.Parameter(torch.tensor(0., requires_grad=True))
    self.bias2 = nn.Parameter(torch.tensor(0., requires_grad=True))
    self.bias3 = nn.Parameter(torch.tensor(0., requires_grad=True))
    self.sigmoid = SIG()


    self.optimizer = optim.Adam(self.parameters(), learning_rate)
    self.loss = nn.MSELoss()

  def forward(self, x):
    return self.net(x)

  def save(self, area):
    state_dict = self.state_dict()
    genome = np.concatenate([np.reshape(state_dict['l1.weight'].numpy().transpose(), 10), np.array([state_dict['bias1'].numpy()]),
                             np.reshape(state_dict['l2.weight'].numpy().transpose(), 25), np.array([state_dict['bias2'].numpy()]),
                             np.reshape(state_dict['l3.weight'].numpy().transpose(), 10), np.array([state_dict['bias3'].numpy()])])
    with open('./data/genome_{}.pkl'.format(area), 'wb') as f:
      pkl.dump(genome, f)

  def train_step(self, x, target):
    x = self.sigmoid(self.l1(x) + self.bias1)
    x = self.sigmoid(self.l2(x) + self.bias2)
    y = self.sigmoid(self.l3(x) + self.bias3)

    loss = self.loss(y, target)

    self.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss

def collect_data(area, show=False):
  env = gym.make(registered_envs['Walker2D']['gym_name'])
  env.reset()

  if show:
    plt.figure()
    view = env.render(mode='rgb_array')
    plt.imshow(view)
    plt.draw()
    plt.pause(1)

  if area == 0:
    action = np.array([1, 1])
  elif area == 1:
    action = np.array([-1, .95])
  elif area == 2:
    action = np.array([0, -1])
  elif area == 3:
    action = np.array([-1, 0])
  else:
    raise ValueError('Given area index {}.'.format(area))

  cum_r = 0
  done = False
  count = 0
  observations = []
  actions = []
  stop_count = 0
  old_act = action.copy()

  while not done :
    obs, r, done, i = env.step(action)
    observations.append(obs)
    actions.append(action)

    if r != 0 and stop_count <= 2:
      action = np.zeros(2)
      stop_count += 1
    if stop_count > 2:
      action = old_act

    count += 1
    cum_r += r
    if show:
      view = env.render(mode='rgb_array')
      plt.imshow(view)
      plt.draw()
      plt.pause(.01)

  print("Tot steps: {} - Tot r: {} - R area: {}".format(count, cum_r, i))

  data = np.array([actions, observations])
  with open('data/data_area_{}.pkl'.format(area), 'wb') as f:
    pkl.dump(data, f)

def train_nn(area, epochs=20000):
  controller = FFNN()
  with open('./data/data_area_{}.pkl'.format(area), 'rb') as f:
    data = pkl.load(f)

  x = torch.Tensor(data[1])
  target = torch.Tensor(data[0])

  for i in range(epochs):
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    loss = controller.train_step(x[idx], target[idx])
    if i%100 == 0:
      print("Loss at {}: {}".format(i, loss))

  controller.save(area)

def test_saved_controller(area, show=False):

  with open('./data/genome_{}.pkl'.format(area), 'rb') as f:
    genome = pkl.load(f)

  controller = FFNeuralController(input_size=2, output_size=2)
  controller.load_genome(genome)
  if show:
    plt.figure()

  env = gym.make(registered_envs['Walker2D']['gym_name'])
  obs = env.reset()
  if show:
    view = env.render(mode='rgb_array')
    plt.imshow(view)
    plt.draw()
    plt.pause(1)
  done = False
  count = 0
  cum_r = 0

  while not done :
    action = controller.evaluate(obs)
    obs, r, done, i = env.step(action)
    count += 1
    cum_r += r
    if show:
      view = env.render(mode='rgb_array')
      plt.imshow(view)
      plt.draw()
      plt.pause(.01)

  print("Tot steps: {} - Tot r: {} - R area: {}".format(count, cum_r, i))
  assert i['rew_area'] == area

for area in range(0, 4):
  print("Area: {}".format(area))
  print('Collecting training data...')
  collect_data(area, show=True)
  print("Training net...")
  train_nn(area)
  print("Testing...")
  test_saved_controller(area)
  print()
print("Done")