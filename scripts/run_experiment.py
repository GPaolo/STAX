# Created by Giuseppe Paolo 
# Date: 28/07/2020

import sys, os
import setuptools
from parameters import Params
import numpy as np
import random
import traceback
from progress.bar import Bar
import argparse
import multiprocessing as mp
from parameters import params
from core.searcher import Searcher
import json
from analysis.logger import Logger

import datetime
from environments import registered_envs

experiments = ['NS', 'LOGSIGN', 'TIME-LOGSIGN',
               'SIGN', 'TIME-SIGN',
               'CMA-ES', 'CMA-NS',
               'NSGA-II', 'SERENE', 'NSLC',
               'ME', 'CMA-ME',
               'RND', 'NT',
               'TAXONS', 'TAXO-S', 'TAXO-N', 'TAXONS-P',
               'STAX', 'STAX-P', 'STAX-NT'
               ]

#######
# Solution coming from:
# https://stackoverflow.com/questions/47776486/python-struct-error-i-format-requires-2147483648-number-2147483647
#######
import functools
import struct

def patch_mp_connection_bpo_17560():
    """Apply PR-10305 / bpo-17560 connection send/receive max size update

    See the original issue at https://bugs.python.org/issue17560 and
    https://github.com/python/cpython/pull/10305 for the pull request.

    This only supports Python versions 3.3 - 3.7, this function
    does nothing for Python versions outside of that range.

    """
    patchname = "Multiprocessing connection patch for bpo-17560"
    if not (3, 3) < sys.version_info < (3, 8):
        print(
            patchname + " not applied, not an applicable Python version: %s",
            sys.version
        )
        return

    from multiprocessing.connection import Connection

    orig_send_bytes = Connection._send_bytes
    orig_recv_bytes = Connection._recv_bytes
    if (
        orig_send_bytes.__code__.co_filename == __file__
        and orig_recv_bytes.__code__.co_filename == __file__
    ):
        print(patchname + " already applied, skipping")
        return

    @functools.wraps(orig_send_bytes)
    def send_bytes(self, buf):
        n = len(buf)
        if n > 0x7fffffff:
            pre_header = struct.pack("!i", -1)
            header = struct.pack("!Q", n)
            self._send(pre_header)
            self._send(header)
            self._send(buf)
        else:
            orig_send_bytes(self, buf)

    @functools.wraps(orig_recv_bytes)
    def recv_bytes(self, maxsize=None):
        buf = self._recv(4)
        size, = struct.unpack("!i", buf.getvalue())
        if size == -1:
            buf = self._recv(8)
            size, = struct.unpack("!Q", buf.getvalue())
        if maxsize is not None and size > maxsize:
            return None
        return self._recv(size)

    Connection._send_bytes = send_bytes
    Connection._recv_bytes = recv_bytes

    print(patchname + " applied")
#######

if __name__ == "__main__":
  patch_mp_connection_bpo_17560()

  # To check why these options are here:
  # 1. https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
  # 2. https://pytorch.org/docs/stable/multiprocessing.html#sharing-strategies
  mp.set_start_method('spawn')
  #  mp.set_sharing_strategy('file_system') # Fundamental otherwise crashes complaining that too many files are open

  parser = argparse.ArgumentParser('Run evolutionary script')
  parser.add_argument('-env', '--environment', help='Environment to use', choices=list(registered_envs.keys()))
  parser.add_argument('-exp', '--experiment', help='Experiment type. Defines the behavior descriptor',
                      choices=experiments),
  parser.add_argument('-ver', '--version', help='Version of the experiment')
  parser.add_argument('-sp', '--save_path', help='Path where to save the experiment')
  parser.add_argument('-mp', '--multiprocesses', help='How many parallel workers need to use', type=int)
  parser.add_argument('-p', '--pop_size', help='Size of the population', type=int)
  parser.add_argument('-eb', '--evaluation_budget', help='Number of evaluations to perform', type=int)
  parser.add_argument('-v', '--verbose', help='Verbose', action='store_true')
  parser.add_argument('-li', help='Local improvement', action='store_true')
  parser.add_argument('--restart_gen', help='Generation at which to restart. It will load from the savepath', type=int)
  parser.add_argument('--reset_ae', help='Resets AE before every training. If used with NT the AE is a new random AE everytime.', action='store_true')
  parser.add_argument('--samples', help='How many samples to use to calculate the BD for TAXONS based approachs', type=int)
  parser.add_argument('--sched', help='Choose which scheduler to use on SERENE based algos', default='alt', choices=['mab', 'alt'])
  parser.add_argument('-em', '--emitter', help='Choose which emitter to use', default='fitness', choices=['novelty', 'fitness', 'cma-es'])
  parser.add_argument('-sched', '--scheduler', help='Choose scheduler to use', default='alternating', choices=['alternating', 'fixed_sampling', 'time_sampling', 'mab'])
  parser.add_argument('-sm_exp',
                      help='Activates the flag for the STAX_single/multi experiment in which the AE is trained on all the frames but the BD is obtained only on the last frame',
                      action='store_true')

  args = parser.parse_args()

  if args.environment is not None: params.env_name = args.environment
  if args.experiment is not None: params.exp_type = args.experiment
  if args.save_path is not None: params.save_path = os.path.join(args.save_path, params.save_dir)
  if args.multiprocesses is not None: params.multiprocesses = args.multiprocesses
  if args.pop_size is not None: params.pop_size = args.pop_size
  if args.evaluation_budget is not None: params.evaluation_budget = args.evaluation_budget
  if args.version is not None: params.version = args.version
  if args.verbose is True: params.verbose = args.verbose
  if args.li is True: params.local_improvement = False
  if args.reset_ae is True: params.reset_ae = True
  if args.samples is not None: params.samples_per_traj = args.samples
  if args.sched == 'mab': params.mab = True
  params.emitter_type = args.emitter
  params.single_multi_exp = args.sm_exp
  params.scheduler_type = args.scheduler

  print("SAVE PATH: {}".format(params.save_path))
  params.save()

  if params.seed is not None:
    np.random.seed(params.seed)

  bar = Bar('Evals:', max=params.evaluation_budget, suffix='[%(index)d/%(max)d] - Avg time per eval: %(avg).3fs - Elapsed: %(elapsed_td)s')

  searcher = Searcher(params)

  if args.restart_gen is not None:
    print("Restarting:")
    print("\t Restarting from generation {}".format(args.restart_gen))
    print("\t Loading from: {}".format(args.save_path))
    searcher.load_generation(args.restart_gen, args.save_path)
    print("\t Loading done.")

  gen_times = []
  evaluated_points = 0
  previous_count = evaluated_points
  while evaluated_points < params.evaluation_budget:
    if params.verbose:
      print("Generation: {}".format(searcher.generation))

    try:
      gen_time, evaluated_points = searcher.chunk_step()
      if evaluated_points % 100000 == 0:
        print("Evaluated: {}".format(evaluated_points))
      gen_times.append(gen_time)

    except KeyboardInterrupt:
      print('User interruption. Saving.')
      searcher.population.save(params.save_path, 'gen_{}'.format(searcher.generation))
      searcher.evolver.archive.save(params.save_path, 'gen_{}'.format(searcher.generation))
      searcher.offsprings.save(params.save_path, 'gen_{}'.format(searcher.generation))
      if hasattr(searcher.evolver, 'rew_archive'):
        searcher.evolver.rew_archive.save(params.save_path, 'gen_{}'.format(searcher.generation))
      bar.finish()
      total_time = np.sum(gen_times)
      Logger.data['Time'] = str(datetime.timedelta(seconds=total_time))
      Logger.data['Evaluated points'] = searcher.evolver.evaluated_points
      Logger.data['Generations'] = searcher.generation + 1
      Logger.data['End'] = 'User interrupt'
      searcher.close()
      with open(os.path.join(params.save_path, 'recap.json'), 'w') as fp:
        json.dump(Logger.data, fp)
      break

    except Exception as e:
      print('Exception occurred.')
      print(traceback.print_exc())
      total_time = np.sum(gen_times)
      Logger.data['Time'] = str(datetime.timedelta(seconds=total_time))
      Logger.data['Evaluated points'] = searcher.evolver.evaluated_points
      Logger.data['Generations'] = searcher.generation + 1
      Logger.data['End'] = traceback.print_exc()
      with open(os.path.join(params.save_path, 'recap.json'), 'w') as fp:
        json.dump(Logger.data, fp)
      searcher.close()
      bar.finish()
      sys.exit()

    bar.next(n=evaluated_points - previous_count)
    previous_count = evaluated_points

  searcher.population.save(params.save_path, 'final')
  searcher.evolver.archive.save(params.save_path, 'final')
  searcher.offsprings.save(params.save_path, 'final')
  if hasattr(searcher.evolver, 'rew_archive'):
    searcher.evolver.rew_archive.save(params.save_path, 'final')
  if searcher.ae is not None:
    searcher.ae.save(os.path.join(params.save_path, 'ae'), 'ae_final')

  total_time = np.sum(gen_times)
  print("Total time: {}".format(str(datetime.timedelta(seconds=total_time))))
  Logger.data['Time'] = str(datetime.timedelta(seconds=total_time))
  Logger.data['Evaluated points'] = evaluated_points
  Logger.data['Generations'] = searcher.generation + 1
  Logger.data['End'] = 'Finished'
  Logger.data['Emitters'] = searcher.evolver.emitters_data

  with open(os.path.join(params.save_path, 'recap.json'), 'w') as fp:
    json.dump(Logger.data, fp)

  searcher.close()
  print('Done.')














































