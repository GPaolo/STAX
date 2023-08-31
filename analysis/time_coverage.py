# Created by Giuseppe Paolo 
# Date: 30/06/2021

import os
import numpy as np
import parameters
import multiprocessing as mp
from core import Evaluator
from analysis import utils
from environments.environments import registered_envs
import pickle as pkl
import argparse
from analysis.gt_bd import *
from core.population import Archive, Population

env_evaluator = None
main_pool = None

class EvalTimeCVG(object):
  """
  This class is used to evaluate the time coverage of the whole run along the generations
  """
  def __init__(self, exp_path, num_gen_eval, ts=50, parallel=False):
    """
    :param exp_path:
    :param num_gen_eval: How many generations to evaluate
    """
    self.params = parameters.Params()
    self.params.load(os.path.join(exp_path, '_params.json'))
    self.exp_path = exp_path
    self.num_ts = ts
    self.time_steps = np.linspace(0, 1, num=self.num_ts)
    self.parallel = parallel

    self.env = registered_envs[self.params.env_name]
    self.gt_bd = self.env['gt_bd']
    self.traj_to_obs = self.env['traj_to_obs']
    self.grid_params = self.env['grid']

    if not os.path.exists(os.path.join(self.exp_path, 'analyzed_data')):
      os.mkdir(os.path.join(self.exp_path, 'analyzed_data'))

    self.generations_to_eval = self.find_generations(num_gen_eval)

    if self.parallel:
      global main_pool
      main_pool = mp.Pool(initializer=self.init_process)

    self.env_evaluator = Evaluator(self.params)

  def bd_extractor(self, traj, *args, **kwargs):
    """
    This function works as bd_extractor. It uses the gt_bd function to extract the gt_bd at each of the given time_steps
    This info is then saved in the field 'bd' of the archive_pop
    :param traj:
    :return:
    """
    return [self.gt_bd(traj[1:], self.env['max_steps'], ts=t) for t in self.time_steps]

  def init_process(self):
    """
    This function is used to initialize the pool so each process has its own instance of the evaluator
    :return:
    """
    global env_evaluator
    env_evaluator = Evaluator(self.params)

  def _feed_eval(self, agent):
    """
    This function feeds the agent to the evaluator and returns the updated agent
    :param agent:
    :return:
    """
    global env_evaluator
    agent = env_evaluator(agent, self.bd_extractor) # TODO pass the bd that returns the whole traj
    return agent

  def evaluate_in_env(self, agents):
    """
    This function evaluates the population in the environment by passing it to the parallel evaluators.
    :return:
    """
    global main_pool
    if self.parallel:
      agents = main_pool.map(self._feed_eval, agents) # As long as the ID is fine, the order of the element in the list does not matter
    else:
      for i in range(agents.size):
        if self.params.verbose: print(".", end = '') # The end prevents the newline
        agents[i] = self.env_evaluator(agents[i], self.bd_extractor)
    return agents

  def find_generations(self, num_gen_eval):
    gen = 1
    while os.path.exists(os.path.join(self.exp_path, 'population_gen_{}.pkl'.format(gen))):
      gen += 1
    final_gen = gen - 1
    return np.linspace(1, final_gen, num_gen_eval, dtype=int)

  def eval_final_archive(self):
    """
    This function loads and evaluates the final archive
    :return: a dict containing the id and traj from the last archive
    """
    archive = Archive(self.params)
    archive.load(os.path.join(self.exp_path, 'archive_final.pkl'))

    # Reconvert archive to Population, this way it's easier to evaluate with the evaluator
    archive_pop = Population(self.params, init_size=archive.size)
    archive_pop['genome'] = archive['genome']
    archive_pop['id'] = archive['id']

    archive_pop = self.evaluate_in_env(archive_pop)
    return {'id': archive_pop['id'], 'traj': archive_pop['bd']}

  def calculate_time_cvg(self, final_arch):
    """
    This function calculates the coverage along time of the archive for each of the given generations
    :param final_arch: archive data returned by the eval_final_archive function
    :return: a dict in the form {gen: [cvg along time]}
    """
    grid = utils.CvgGrid(self.params, self.grid_params)
    cvg_at_gen = {}

    for gen in self.generations_to_eval:
      archive = utils.load_data(self.exp_path,
                                 generation=gen,
                                 params=self.params,
                                 container_type='archive',
                                 info=['id'])[gen]
      trajs = np.array([final_arch['traj'][final_arch['id'].index(id)] for id in archive['id']])
      cvg_at_gen[gen] = []

      for t in range(self.num_ts):
        [grid.store({'evaluated': gen, 'gt_bd': traj}) for traj in trajs[:, t]]
        cvg_at_gen[gen].append(utils.calculate_coverage(grid.grid))
        grid.reset_grid()
    return cvg_at_gen


def main(path, args):
  time_cvg = EvalTimeCVG(path, num_gen_eval=args.gens, ts=args.time_steps, parallel=args.multiprocessing)

  print('Evaluating archive...')
  final_arch = time_cvg.eval_final_archive()

  print('Calculating time coverage...')
  cvg = time_cvg.calculate_time_cvg(final_arch)

  print('Saving...')
  with open(os.path.join(time_cvg.exp_path, 'analyzed_data', 'time_cvg.pkl'), 'wb') as f:
    pkl.dump(cvg, f)

  print('Done.')

if __name__ == "__main__":
  parser = argparse.ArgumentParser('Run script for time coverage evaluation')
  parser.add_argument('-p', '--path', help='Path of experiment')
  parser.add_argument('-mp', '--multiprocessing', help='Multiprocessing', action='store_true')
  parser.add_argument('-g', '--gens', help='Number of generations to evaluate', type=int, default=None)
  parser.add_argument('-ts', '--time_steps', help='Time steps to evaluate', type=int, default=50)
  parser.add_argument('--multi', help='Flag to give in case multiple runs have to be evaluated', action='store_true')

  args = parser.parse_args()
  #['-p', '/home/giuseppe/src/cmans/experiment_data/CollectBall_SIGN_std', '-g', '4', '--multi', '-mp'])

  if not args.multi:
    paths = [args.path]
  else:
    if not os.path.exists(args.path):
      raise ValueError('Path does not exist: {}'.format(args.path))
    raw_paths = [x for x in os.walk(args.path)]
    raw_paths = raw_paths[0]
    paths = [os.path.join(raw_paths[0], p) for p in raw_paths[1]]

  # Use this only if multiple paths but env cannot be evaluated in multi.
  if args.multi and not args.multiprocessing:
    pool = mp.Pool()
    pool.starmap(main, zip(paths, [args]*len(paths)))
  else:
    for path in paths:
      print('Working on: {}'.format(path))
      main(path, args)
