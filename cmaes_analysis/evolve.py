# Created by Giuseppe Paolo 
# Date: 17/11/2020

import pickle as pkl
import numpy as np
from parameters import params, ROOT_DIR
from environments.environments import *
from core.evaluator import Evaluator
from core.behavior_descriptors.behavior_descriptors import BehaviorDescriptor
from cmaes import CMA
from core.population import Population
import os

class FIT(object):
  """
  This class implements the fitness emitter
  """
  def __init__(self, init_mean, mutation_rate, parameters):
    self.init_mean = init_mean
    self._mutation_rate = mutation_rate
    self._params = parameters
    self._pop_size = self._params.emitter_population
    self.pop = self._init_pop()

  def mutate_genome(self, genome):
    """
    This function mutates the genome
    :param genome:
    :return:
    """
    genome = genome + np.random.normal(0, self._mutation_rate, size=np.shape(genome))
    return genome.clip(self._params.genome_limit[0], self._params.genome_limit[1])

  def _init_pop(self):
    """
    This function initializes the emitter pop around the parent
    :return:
    """
    pop = Population(self._params, self._pop_size)
    for agent in pop:
      agent['genome'] = self.mutate_genome(self.init_mean)
    return pop

  def generate_off(self, generation):
    """
    This function generates the offsprings of the emitter
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
    self.pop.agent_id = max(off_ids) + 1 # This saves the maximum ID reached till now
    return offsprings

  def update_pop(self, offsprings):
    """
    This function chooses the agents between the pop and the off with highest reward to create the new pop
    :param offsprings:
    :return:
    """
    performances = self.pop['reward'] + offsprings['reward']
    idx = np.argsort(performances)[::-1]  # Order idx according to performances.
    parents_off = self.pop.pop + offsprings.pop
    # Update population list by going through it and putting an agent from parents+off at its place
    for new_pop_idx, old_pop_idx in zip(range(self.pop.size), idx[:self.pop.size]):
      self.pop.pop[new_pop_idx] = parents_off[old_pop_idx]

  def should_stop(self):
    """
    Checks internal stopping criteria
    :return:
    """
    return False

class Evolver(object):
  def __init__(self, area, params, type='cmaes'):
    """This class is used to do tests on Fit and CMAES evolvers"""
    self.params = params
    self.params.env_name = 'Walker2D'
    self.params.exp_type = 'NS'
    self.area = area
    self.type = type

    with open(os.path.join(ROOT_DIR, 'analysis/cmaes_analysis/data/genome_{}.pkl'.format(self.area)), 'rb') as f:
      self.init_genome = pkl.load(f)

    self.bd_extractor = BehaviorDescriptor(self.params)
    self.evaluator = Evaluator(self.params)

    self.controller = registered_envs[self.params.env_name]['controller']['controller'](
      input_size=registered_envs[self.params.env_name]['controller']['input_size'],
      output_size=registered_envs[self.params.env_name]['controller']['output_size'])

    genome_size = self.controller.genome_size
    bounds = np.array([-5, 5]) * np.ones((genome_size, len(self.params.genome_limit)))

    self.pop = Population(self.params, self.params.emitter_population)

    if type == 'cmaes':
      self.cmaes = CMA(mean=self.init_genome,
                       sigma=0.01,
                       bounds=bounds,
                       seed=self.params.seed,
                       population_size=self.params.emitter_population
                      )
    elif self.type == 'fit':
      self.fit = FIT(init_mean=self.init_genome,
                     mutation_rate=0.05,
                     parameters=self.params)

  def update_cmaes_pop(self, generation):
    self.pop.empty()
    for idx in range(self.params.emitter_population):
      self.pop.add()
      self.pop[idx]['genome'] = self.cmaes.ask()
      self.pop[idx]['born'] = generation

  def evaluate_cames_performances(self):
    """
    This function evaluates performances of the population. It's what calls the tell function from the optimizer
    The novelty is evaluated according to the given distance metric
    :param population:
    :param offsprings:
    :param pool: Multiprocessing pool
    :return:
    """
    solutions = [(genome, -value) for genome, value in zip(self.pop['genome'], self.pop['reward'])]
    self.cmaes.tell(solutions)

  def evaluate(self, population):
    for i in range(population.size):
      if population[i]['evaluated'] is None:
        population[i] = self.evaluator(population[i], self.bd_extractor)

  def main(self, budget):
    generation = 0
    eval_points = 0
    init_pop = False

    save_path = os.path.join(ROOT_DIR, 'analysis/cmaes_analysis/data/{}/{}'.format(self.type, self.area))
    if not os.path.exists(save_path):
      os.makedirs(save_path)

    while eval_points < budget:

      if self.type == 'cmaes':
        self.update_cmaes_pop(generation)
        self.evaluate(self.pop)
        self.pop['evaluated'] = list(range(eval_points, eval_points + self.pop.size))
        eval_points += self.pop.size
        self.evaluate_cames_performances()
        self.pop.save(save_path, 'gen_{}'.format(generation))

        print("Pop at gen {}. Values: {}".format(generation, self.pop['reward']))
        generation += 1

      if self.type == 'fit':
        if not init_pop:
          # Evaluate initial pop
          self.evaluate(self.fit.pop)
          self.fit.pop['evaluated'] = list(range(eval_points, eval_points + self.fit.pop.size))
          eval_points += self.fit.pop.size
          init_pop = True
        else:
          offs = self.fit.generate_off(generation)
          self.evaluate(offs)
          offs['evaluated'] = list(range(eval_points, eval_points + offs.size))
          eval_points += offs.size
          self.fit.update_pop(offs)

        self.fit.pop.save(save_path, 'gen_{}'.format(generation))
        print("Pop at gen {}. Values: {}".format(generation, self.fit.pop['reward']))
        generation += 1






if __name__ == "__main__":
  area = 0
  evals = 10000

  for area in range(0, 4):
    evolver = Evolver(area=area, params=params, type='cmaes')
    evolver.main(evals)
    evolver = Evolver(area=area, params=params, type='fit')
    evolver.main(evals)



