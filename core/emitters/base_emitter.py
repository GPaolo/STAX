# Created by Giuseppe Paolo 
# Date: 13/09/2021

import numpy as np
from core.population import Population

class BaseEmitter(object):
  """
  This class implements the base emitter
  """
  def __init__(self, ancestor, mutation_rate, parameters, **kwargs):
    self.ancestor = ancestor
    self._init_mean = self.ancestor['genome']
    self.id = ancestor['id']
    self._mutation_rate = mutation_rate
    self._params = parameters
    self._pop_size = self._params.emitter_population
    self._genome_bounds = self._params.genome_limit

    # List of lists. Each inner list corresponds to the values obtained during a step
    # We init with the ancestor reward so it's easier to calculate the improvement
    self.values = []
    self.archived_values = []
    self.improvement = 0
    self._init_values = None
    self.archived = []  # List containing the number of archived agents at each step
    self.most_novel = self.ancestor
    self.steps = 0

    self._init_pop()
    self.ns_arch_candidates = Population(self._params, init_size=0, name='ns_arch_cand')

  def mutate_genome(self, genome):
    """
    This function mutates the genome
    :param genome:
    :return:
    """
    genome = genome + np.random.normal(0, self._mutation_rate, size=np.shape(genome))
    return genome.clip(self._params.genome_limit[0], self._params.genome_limit[1])

  def estimate_improvement(self):
    """
    This function calculates the improvement given by the last updates wrt the parent
    If negative improvement, set it to 0.
    If there have been no updates yet, return the ancestor parent as reward
    Called at the end of the emitter evaluation cycle
    :return:
    """
    if self._init_values is None: # Only needed at the fist time
      self._init_values = self.values[:3]
    self.improvement = np.max([np.mean(self.values[-3:]) - np.mean(self._init_values), 0])

    # If local improvement update init_values to have improvement calculated only on the last expl step
    if self._params.local_improvement:
      self._init_values = self.values[-3:]

  def _init_pop(self):
    """
    This function initializes the emitter pop around the parent
    :return:
    """
    raise NotImplementedError

  def should_stop(self):
    """
    Checks internal stopping criteria
    :return:
    """
    raise NotImplementedError

  def update_pop(self, offsprings):
    """
    This function updates the current population of the emitter
    :param offsprings:
    :return:
    """
    raise NotImplementedError

  def generate_new_solutions(self, generation):
    """
    This function generates new solutions to be evaluated
    :return:
    """
    raise NotImplementedError