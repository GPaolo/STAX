# Created by Giuseppe Paolo 
# Date: 27/07/2020

import os
from datetime import datetime
import json
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
print('Root directory {}'.format(ROOT_DIR))

class Params(object):
  """
  This is the parameter file
  """
  def __init__(self):
    self.seed = datetime.now().microsecond  # Random seed is set as system clock
    self.genome_limit = [-5, 5]
    self.pop_size = 10
    self.generations = 10
    self.evaluation_budget = 500
    self.chunk_size = 1000

    self.signature_order = 2
    # AE Stuff
    # ----------------
    self.bd_size = 10
    self.max_training_epochs = 100
    self.validation_interval = 1
    self.archive_sampling = False
    self.batch_size = 128
    self.train_on_reward = True
    self.reset_ae = False # This param randomized the AE everytime before training. If used with NT the AE is randomly chosen everytime
    self.samples_per_traj = 1
    # ----------------

    self.single_multi_exp = False

    self.multiprocesses = 0
    self.novelty_neighs = 15
    self.mutation_parameters = {'mu': 0., 'sigma': 0.05, 'mut_prob': 1}
    self.offsprings_per_parent = 2
    self.selection_operator = 'random'  # random or best
    self.novelty_distance_metric = 'euclidean'
    self._lambda = 5  # Number of agents added to archive

    self.archive_stored_info = ['genome', 'bd', 'id', 'reward', 'parent', 'born',
                                'stored', 'evaluated', 'ancestor', 'rew_area', 'gt_bd', 'view', 'surprise']

    self.agent_template = {'genome': None,
                           'reward': None,
                           'bd': None,
                           'novelty': None,
                           'parent': None,
                           'id': None,
                           'born': 0,
                           'stored': None,
                           'emitter': False,
                           'evaluated': None,
                           'ancestor': None,
                           'rew_area': None,
                           'gt_bd': None,
                           'view': None,
                           'surprise': None,
                           'local_comp': None,
                          }

    self.verbose = True
    self._env_name = 'Point2D'
    self.behavior_descriptor = 'last_obs'
    self._exp_type = 'NS'
    self._version = 'std'
    self.emitter_type = 'fitness' # ['fitness', 'novelty', 'cma-es']

    self.scheduler_type = 'alternating' # ['alternating', 'fixed_sampling', 'time_sampling', 'mab']
    # Emitter stuff
    # ---------------
    self.emitter_population = 6
    self.local_improvement = False
    self.max_emitters = -1 # If -1 no limit is used
    self.max_emitter_steps = -1 # If -1, no limit is used
    self.stagnation = 'custom'
    # ---------------

    self.date_time = datetime.now().strftime("%Y_%m_%d_%H:%M_%f")
    self.save_dir = os.path.join('{}_{}_{}'.format(self.env_name, self.exp_type, self.version), self.date_time)
    self.save_path = os.path.join(ROOT_DIR, 'experiment_data', self.save_dir)

  @property
  def ae(self):
    """True if a method uses the AE"""
    if self._exp_type in ['TAXONS', 'NT', 'TAXO-N', 'TAXO-S', 'TAXONS-P', 'STAX', 'STAX-P', 'STAX-NT']:
      return True
    else:
      return False

  # If env name and exp type are changed, also the save directory needs to be changed
  @property
  def env_name(self):
    return self._env_name

  @env_name.setter
  def env_name(self, value):
    self._env_name = value
    date_time = datetime.now().strftime("%Y_%m_%d_%H:%M_%f")
    self.save_dir = os.path.join('{}_{}_{}'.format(self.env_name, self.exp_type, self.version), date_time)
    self.save_path = os.path.join(ROOT_DIR, 'experiment_data', self.save_dir)

  @property
  def exp_type(self):
    return self._exp_type

  @exp_type.setter
  def exp_type(self, value):
    self._exp_type = value
    date_time = datetime.now().strftime("%Y_%m_%d_%H:%M_%f")
    self.save_dir = os.path.join('{}_{}_{}'.format(self.env_name, self.exp_type, self.version), date_time)
    self.save_path = os.path.join(ROOT_DIR, 'experiment_data', self.save_dir)

  @property
  def version(self):
    return self._version

  @version.setter
  def version(self, value):
    self._version = value
    date_time = datetime.now().strftime("%Y_%m_%d_%H:%M_%f")
    self.save_dir = os.path.join('{}_{}_{}'.format(self.env_name, self.exp_type, self.version), date_time)
    self.save_path = os.path.join(ROOT_DIR, 'experiment_data', self.save_dir)

  # --------------------------------------------
  def _get_dict(self):
    params_dict = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
    return params_dict

  def save(self):
    os.makedirs(self.save_path, exist_ok=True)
    with open(os.path.join(self.save_path, '_params.json'), 'w') as f:
      json.dump(self._get_dict(), f, indent=4)

  def load(self, load_path):
    print("Loading parameters...")
    assert os.path.exists(load_path), 'Specified parameter file does not exists in {}.'.format(load_path)
    with open(load_path) as f:
      data = json.load(f)
    for key in data:
      setattr(self, key, data[key])
      # assert self.__dict__[key] == data[key], 'Could not set {} parameter.'.format(key)
    print('Done')
  # --------------------------------------------

params = Params()