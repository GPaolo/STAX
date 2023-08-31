# Created by Giuseppe Paolo 
# Date: 21/09/2021

import numpy as np

class FixedSampling(object):
  """
  This scheduler samples between exploration and exploitation with a fixed prob
  """
  def __init__(self, **kwargs):
    self.choice = None
    self.sampling_prob = [0.5, 0.5]

  def action(self, **kwargs):
    # If no selection has been done yet, only choose exploration
    if self.choice is None:
      self.choice = 'explore'
    else:
      self.choice = np.random.choice(['explore', 'exploit'], p=self.sampling_prob)
    return self.choice

  def update(self, **kwargs):
    return

class TimeSampling(FixedSampling):
  """
  This scheduler samples between exploration and exploitation with a prob that changes with the amount of time passed
  """
  def __init__(self, total_budget):
    super(TimeSampling, self).__init__()
    self.total_budget = total_budget
    self.sampling_prob = [1., 0]

  def update(self, evolver, **kwargs):
    """
    Updates the selection probabilities according to the used budget
    :param evolver:
    :param kwargs:
    :return:
    """
    exploration_prob = (self.total_budget - evolver.evaluated_points)/self.total_budget
    self.sampling_prob = [exploration_prob, 1.-exploration_prob]
    return