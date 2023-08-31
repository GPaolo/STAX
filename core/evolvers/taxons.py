# Created by Giuseppe Paolo 
# Date: 17/02/2021

from core.evolvers import BaseEvolver, NSGAII, NoveltySearch
from core.evolvers import utils
import numpy as np


class TaxonsSearch(NoveltySearch):
  """
  This class implements the way TAXONS selects between Novelty and Surprise.
  It works like NS but uses one of the two criteria to update the pop.
  It does so by randomly selecting one of the two
  https://arxiv.org/abs/1909.05508
  """
  def __init__(self, parameters, **kwargs):
    super().__init__(parameters, **kwargs)
    self.update_criteria = ['novelty', 'surprise']

  def choose_criteria(self):
    """
    This function chooses the criteria that is used to update the population and, in case of selection_operator = best
    the criteria for the archive update
    :return:
    """
    self.criteria = str(np.random.choice(self.update_criteria))

  def update_archive(self, population, offsprings, generation):
    """
    Updates the archive according to the strategy and the criteria given.
    :param population:
    :param offsprings:
    :return:
    """
    # Select update criteria. Done here in any case, then the fuctions that need it use it.
    self.choose_criteria()

    # Get list of ordered indexes according to selection strategy
    if self.params.selection_operator == 'random':
      idx = list(range(offsprings.size))
      np.random.shuffle(idx)
    elif self.params.selection_operator == 'best':
      performances = offsprings[self.criteria]
      idx = np.argsort(performances)[::-1]  # Order idx according to performances. (From highest to lowest)
    else:
      raise ValueError(
        'Please specify a valid selection operator for the archive. Given {} - Valid: ["random", "best"]'.format(
          self.params.selection_operator))

    # TODO This part gets slower with time.
    # Add to archive the first lambda offsprings in the idx list
    for i in idx[:self.params._lambda]:
      offsprings[i]['stored'] = generation
      self.archive.store(offsprings[i])

  def update_population(self, population, offsprings, generation):
    """
    This function updates the population according to the given criteria
    :param population:
    :param offsprings:
    :return:
    """
    performances = population[self.criteria] + offsprings[self.criteria]
    idx = np.argsort(performances)[::-1]  # Order idx according to performances.
    parents_off = population.pop + offsprings.pop
    # Update population list by going through it and putting an agent from parents+off at its place
    for new_pop_idx, old_pop_idx in zip(range(population.size), idx[:population.size]):
      population.pop[new_pop_idx] = parents_off[old_pop_idx]


class TaxonsPareto(NSGAII):
  """
  This class implements a multiobjective approach search like NSGA.
  But insted of using novelty and reward uses novelty and surprise from the trained TAXONS's AE
  """

  def __init__(self, parameters):
    super().__init__(parameters)
    self.update_criteria = ['novelty', 'surprise']