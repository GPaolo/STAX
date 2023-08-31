# Created by Giuseppe Paolo
# Date: 29/07/2020

import numpy as np
from parameters import params
from environments.environments import registered_envs
from core.evolvers import utils

sigma = params.mutation_parameters['sigma']
mu = params.mutation_parameters['mu']
mutation_operator = np.random.normal

controller = registered_envs[params.env_name]['controller']['controller'](**registered_envs[params.env_name]['controller'])
genome_size = 300#controller.genome_size
genome_limit = params.genome_limit
genome = np.clip(np.random.normal(0, 1, size=genome_size), genome_limit[0], genome_limit[1])

def mutate_genome(genome):
  """
  This function mutates the genome by using the mutation operator.
  NB: The genome is clipped in the range [-1, 1]
  :param genome:
  :return:
  """
  mutation = [mutation_operator(mu, sigma) if np.random.random() < 1 else 0 for k in range(len(genome))]
  genome = np.clip(genome + mutation,
                   params.genome_limit[0], params.genome_limit[1])
  return genome

mutated = []
for i in range(100):
  mutated.append(mutate_genome(genome))

dist = utils.calculate_distances([genome], mutated).flatten()
print("Mean dist: {} - NB of STD: {}".format(np.mean(dist), np.mean(dist)/sigma))
