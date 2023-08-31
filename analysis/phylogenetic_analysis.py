# Created by Giuseppe Paolo 
# Date: 09/10/2020

import numpy as np
from core.population import Population
from parameters import Params
import sys, os
import argparse
import gzip
import pickle as pkl

sys.setrecursionlimit(10000)

class PhylAnal(object):
  """
  This class is used to perform the phylogenetic analysis of the solutions
  """
  def __init__(self, path, max_gen=-1):
    self.path = path
    self.params = Params()
    self.params.load(os.path.join(self.path, '_params.json'))
    self.pop = Population(self.params)
    self.generation = 0
    if max_gen == -1:
      self.max_gen = np.inf
    else:
      self.max_gen = max_gen

  def build_phyl_tree(self):
    """
    This function is the one that builds the tree
    :return:
    """
    indivs_by_id = {}
    indivs_by_generation = {}

    while not self.generation == self.max_gen:
      self.generation += 1
      print("Working on generation: {}".format(self.generation))
      pop_filename = "population_gen_{}.pkl".format(self.generation)
      if not os.path.exists(os.path.join(self.path, pop_filename)):
        print('No population at generation {}. Stopping'.format(self.generation))
        break

      self.pop.empty()
      self.pop.load(os.path.join(self.path, pop_filename))
      indiv_gen = []
      for agent in self.pop:
        # Agent was already discovered
        if agent['id'] in indivs_by_id:
          indiv_gen.append(indivs_by_id[agent['id']])
          indivs_by_id[agent['id']]['lastgen'] = self.generation  # update the last generation seen attribute
          indivs_by_id[agent['id']]['age'] = self.generation - indivs_by_id[agent['id']]['born']
          indivs_by_id[agent['id']]['stored'] = agent['stored']
          indivs_by_id[agent['id']]['emitter'] = agent['emitter']

        # Agent is just discovered
        else:
          if agent['parent'] is not None:
            parent = indivs_by_id[agent['parent']]

          indiv = agent.copy()
          indiv['off'] = []
          indiv['lastgen'] = self.generation
          indiv['age'] = self.generation - indiv['born']

          if indiv['parent'] is not None:
            indiv['bd_path'] = parent['bd_path'] + np.linalg.norm(indiv['bd'] - parent['bd'])
            indiv['genome_path'] = parent['genome_path'] + np.linalg.norm(indiv['genome'] - parent['genome'])
            parent['off'].append(indiv['id'])
          else:
            indiv['bd_path'] = 0
            indiv['genome_path'] = 0

          indivs_by_id[indiv['id']] = indiv
          indiv_gen.append(indiv)

      indivs_by_generation[self.generation] = indiv_gen

    for agent in indivs_by_id:
      del indivs_by_id[agent]['genome'] # No need anymore of the genome

    return indivs_by_id, indivs_by_generation

  def find_lineages(self, tree_id):
    """
    This function build two lineages dicts, one for every agent, one only for the rewarding
    :return:
    """
    lineages = {}
    rew_lineages = {}
    for agent_id in tree_id:
      agent = tree_id[agent_id]

      parent = agent['parent']
      lineage = [agent_id]
      rew_lineage = []
      if agent['reward'] > 0: rew_lineage = [agent_id]

      while True:
        if parent is None:
          break
        lineage.append(parent)
        if len(rew_lineage) > 0: rew_lineage.append(parent)
        agent = tree_id[parent]
        parent = agent['parent']

      if agent['id'] not in lineages:
        lineages[agent['id']] = [lineage]
      else:
        lineages[agent['id']].append(lineage)

      if len(rew_lineage) > 0:
        if agent['id'] not in rew_lineages:
          rew_lineages[agent['id']] = [rew_lineage]
        else:
          rew_lineages[agent['id']].append(rew_lineage)

    # Now remove uncomplete lineages
    lineages = self.find_unique_lins(lineages)
    rew_lineages = self.find_unique_lins(rew_lineages)
    return lineages, rew_lineages

  def find_unique_lins(self, lineages):
    """
    This function removes lineages that are not complete, so to have only unique lins
    :param lineages:
    :return:
    """
    for agent in lineages:
      lins = lineages[agent]
      lins.sort(key=len)
      i = 0
      while i < len(lins):
        for j in range(i+1, len(lins)):
          if all(elem in lins[j] for elem in lins[i]):
            del lins[i]
            i = -1
            break
        i += 1
      lineages[agent] = lins
    return lineages

if __name__ == "__main__":
  parser = argparse.ArgumentParser('Perform phylogenetic analysis')
  parser.add_argument('-p', '--path', help='Path of the experiment', type=str)
  parser.add_argument('-g', '--gen', help='Max generations to consider. If -1 it goes till the max generation available', default=-1, type=int)

  args = parser.parse_args(['-p', '/home/giuseppe/src/cmans/experiment_data/Walker2D_CMA-NS/2020_10_13_14:16_557647', '-g', '-1'])
  analyzer = PhylAnal(path=args.path, max_gen=args.gen)
  indivs_by_id, indivs_by_gen = analyzer.build_phyl_tree()
  lineages, rew_lineages = analyzer.find_lineages(indivs_by_id)

  analyzed_path = os.path.join(args.path, 'analyzed_data')
  if not os.path.exists(analyzed_path):
    os.mkdir(analyzed_path)

  with gzip.open(os.path.join(analyzed_path, 'phyl_tree.pkl'), 'wb') as f:
    pkl.dump({'id': indivs_by_id, 'gen':indivs_by_gen, 'lin':lineages, 'rew_lin':rew_lineages}, f)
  print("Done")








