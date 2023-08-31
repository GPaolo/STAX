# Created by Giuseppe Paolo 
# Date: 16/04/2021

import numpy as np

class DiscountedUCB(object):
  """
  This class implements the discounted UCB MAP
  alpha in [0,1]. alpha = 0 only past Q is considered. alpha = 1 only most recent reward is considered
  If alpha=-1 will use as stepsize calculated as 1/arm_pulls.
  The more an arm is pulled, the more stuff from the past (previous Q) is considered.
  """
  def __init__(self, c, alpha, **kwargs):
    self.arm_pulls = np.zeros(2)
    self.total_pulls = 0

    self.Q = np.zeros(2)
    self.c = c
    self.alpha = alpha
    self.choice = None
    self.emitters_available = False

  def action(self, found_emitters=False, **kwargs):
    """
    Choose what to do by using the values calculated with the UCB.
    :param found_emitters:
    :return:
    """

    # The first time or when no emitters have yet been found, just keep on exploring
    if self.choice is None or not self.emitters_available:
      self.choice = 'explore'

    #If emitters are found, calculate which is best between exploration and exploitation.
    else:
      # Upper confidence bound action selection
      # A=argmax(Q(a)+c*(sqrt(log(t)/N(a))
      A = []
      for i in range(2):
        if self.arm_pulls[i] > 0:
          val = self.Q[i] + self.c * np.sqrt(np.log(self.total_pulls) / self.arm_pulls[i])
        else:
          val = np.inf

        A.append(val)
      best = np.argmax(A)
      print("A: {}".format(A))
      if best == 0:
        self.choice = 'explore'
      else:
        self.choice = 'exploit'

    return self.choice

  def update(self, reward, evolver, **kwargs):
    """
    Update values.
    :param choice:
    :param kwargs:
    :return:
    """
    if len(evolver.emitter_candidate) > 0 or len(evolver.emitters) > 0:
      self.emitters_available = True
    else:
      self.emitters_available = False

    if self.choice == 'explore': idx = 0
    else: idx = 1

    # Update counters
    self.arm_pulls[idx] += 1
    self.total_pulls += 1

    # Update step size
    if self.alpha == -1:
      stepSize = 1 / self.arm_pulls[idx]
    else:
      stepSize = self.alpha

    # Update values with discount factor
    self.Q[idx] += stepSize * (reward - self.Q[idx])
    return