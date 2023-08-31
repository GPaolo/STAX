# Created by Giuseppe Paolo 
# Date: 20/09/2021

from core.evolvers import NSGAII
from core.evolvers import utils
import numpy as np
import itertools

class NSLC(NSGAII):
  """
  This class implements the NSGA-II evolver.
  """
  def __init__(self, parameters, **kwargs):
    super(NSLC, self).__init__(parameters, **kwargs)
    self.update_criteria = ['novelty', 'local_comp']

  def evaluate_performances(self, population, offsprings, pool=None):
    """
    This function evaluates the novelty and the local competitiveness of the population and off springs.
    The calculations of the front and all the NSGA-II stuff related to the pareto front is done after but elsewhere
    :param population:
    :param offsprings:
    :param pool:
    :return:
    """
    # Get BSs
    population_bd = population['bd']
    offsprings_bd = offsprings['bd']

    # Need the archive for the novelty
    if self.archive.size > 0:
      archive_bd = self.archive['bd']
    else:
      archive_bd = []

    reference_set = population_bd + offsprings_bd + archive_bd
    bd_set = population_bd + offsprings_bd

    # Get Rewards
    population_rew = population['reward']
    offsprings_rew = offsprings['reward']
    if self.archive.size > 0:
      archive_rew = self.archive['reward']
    else:
      archive_rew = []
    rewards = population_rew + offsprings_rew + archive_rew
    if np.sum(rewards) > 0:
      print()

    novelties, lc_scores = utils.calculate_nov_and_lc(bd_set, reference_set, rewards, distance_metric=self.params.novelty_distance_metric,
                                                      novelty_neighs=self.params.novelty_neighs, pool=pool)
    # Update population and offsprings
    population['novelty'] = novelties[:population.size]
    offsprings['novelty'] = novelties[population.size:]
    population['local_comp'] = lc_scores[:population.size]
    offsprings['local_comp'] = lc_scores[population.size:]