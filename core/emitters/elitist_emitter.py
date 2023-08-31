# Created by Giuseppe Paolo 
# Date: 13/09/2021

from core.emitters import BaseEmitter
from core.population import Population
import numpy as np

class ElitistEmitter(BaseEmitter):
  """
  This class implements the elitist emitter
  """
  @property
  def _update_parameter(self):
    raise NotImplementedError

  def _init_pop(self):
    """
    This function initializes the emitter pop around the parent
    :return:
    """
    self.pop = Population(self._params, self._pop_size)
    for agent in self.pop:
      agent['genome'] = self.mutate_genome(self._init_mean)

  def should_stop(self):
    """
    Checks internal stopping criteria
    :return:
    """
    return False

  def update_pop(self, offsprings):
    """
    This function chooses the agents between the pop and the off with highest reward to create the new pop
    :param offsprings:
    :return:
    """
    performances = self.pop[self._update_parameter] + offsprings[self._update_parameter]
    idx = np.argsort(performances)[::-1]  # Order idx according to performances.
    parents_off = self.pop.pop + offsprings.pop
    # Update population list by going through it and putting an agent from parents+off at its place
    for new_pop_idx, old_pop_idx in zip(range(self.pop.size), idx[:self.pop.size]):
      self.pop.pop[new_pop_idx] = parents_off[old_pop_idx]

  def generate_new_solutions(self, generation):
    """
    This function generates new solutions to be evaluated
    :return:
    """
    offsprings = Population(self._params, init_size=2*self.pop.size, name='offsprings')
    off_genomes = []
    off_ancestors = []
    off_parents = []
    for agent in self.pop:  # Generate 2 offsprings from each parent
      off_genomes.append(self.mutate_genome(agent['genome']))
      off_genomes.append(self.mutate_genome(agent['genome']))
      off_ancestors.append(agent['ancestor'] if agent['ancestor'] is not None else agent['id'])
      off_ancestors.append(agent['ancestor'] if agent['ancestor'] is not None else agent['id'])
      off_parents.append(agent['id'])
      off_parents.append(agent['id'])

    off_ids = self.pop.agent_id + np.array(range(len(offsprings)))

    offsprings['genome'] = off_genomes
    offsprings['id'] = off_ids
    offsprings['ancestor'] = off_ancestors
    offsprings['parent'] = off_parents
    offsprings['born'] = [generation] * offsprings.size
    offsprings['novelty'] = [None] * offsprings.size
    self.pop.agent_id = max(off_ids) + 1 # This saves the maximum ID reached till now
    return offsprings


class FitnessEmitter(ElitistEmitter):
  """
  This class implements the elitist Fitness emitter that selects the agents based on their reward
  """
  @property
  def _update_parameter(self):
    return 'reward'


class NoveltyEmitter(ElitistEmitter):
  """
  This class implements the elitist Fitness emitter that selects the agents based on their reward
  """
  @property
  def _update_parameter(self):
    return 'novelty'