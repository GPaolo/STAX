# Created by Giuseppe Paolo 
# Date: 13/09/2021


from core.emitters import BaseEmitter
from core.population import Population
from environments import registered_envs
import numpy as np
from cmaes import CMA
import copy

class CMAESEmitter(BaseEmitter):
  """
  This class implements the CMA-ES based emitter.
  NB: Given that this is an estimation of distribution algo, it only uses a pop structure to handle the solutions,
  rather and a pop and a off structure.
  """
  def _init_pop(self):
    self.pop = Population(self._params, init_size=self._params.emitter_population, name='cma_es')

    # Instantiated only to extract genome size
    controller = registered_envs[self._params.env_name]['controller']['controller'](**registered_envs[self._params.env_name]['controller'])
    self.genome_size = controller.genome_size

    bounds = self._genome_bounds * np.ones((self.genome_size, len(self._genome_bounds)))
    self._cmaes = CMA(mean=self._init_mean.copy(),
                      sigma=self._mutation_rate,
                      bounds=bounds,
                      seed=self._params.seed,
                      population_size=self._params.emitter_population)

  def should_stop(self):
    """
    Checks internal stopping criteria
    :return:
    """
    return self._cmaes.should_stop()

  def update_pop(self, offsprings):
    """
    This function updates the CMA-ES distribution.
    :param offsprings:
    :return:
    """
    solutions = [(genome, -value) for genome, value in zip(self.pop['genome'], self.pop['reward'])]
    self._cmaes.tell(solutions)

  def generate_new_solutions(self, generation):
    """
    This function generates new solutions to be evaluated
    Given how CMA-ES works, these agents are directly stored into the pop rather than creating a new pop of offsprings
    :return:
    """
    self.pop['genome'] = [self._cmaes.ask() for i in range(self.pop.size)]
    self.pop['parent'] = [self.id] * self.pop.size
    self.pop['born'] = [generation] * self.pop.size
    self.pop['evaluated'] = [None] * self.pop.size
    ancestor = self.ancestor['ancestor'] if self.ancestor is not None else self.id
    self.pop['ancestor'] = [ancestor] * self.pop.size

    self.pop['id'] = self.pop.agent_id + np.array(range(self.pop.size))
    self.pop.agent_id = max(self.pop['id']) + 1 # This saves the maximum ID reached till now
    return self.pop
