# Created by Giuseppe Paolo 
# Date: 27/07/2020

#Here I make the class that creates everything. I pass the parameters as init arguments, this one creates the param class, and the pop, arch, opt alg

import os
from core.population import Population
from core.evolvers import *
from core.schedulers import *
from core.behavior_descriptors.behavior_descriptors import BehaviorDescriptor
from core.auto_encoders import ConvAE as AE
from core import Evaluator
import multiprocessing as mp
from timeit import default_timer as timer
import gc
from analysis.logger import Logger
import torch
import numpy as np

evaluator = None
main_pool = None # Using pool as global prevents the creation of new environments at every generation


class Searcher(object):
  """
  This class creates the instance of the NS algorithm and everything related
  """
  def __init__(self, parameters):
    self.parameters = parameters
    self.bd_extractor = BehaviorDescriptor(self.parameters)
    self.ae = None

    self.generation = 1
    # This two are used to select how many chunks steps have to pass before training the AE.
    # Every time the AE is trained, the chunk counter is reset to 0 and the training_interval increased of 1.
    # The AE is then trained everytime the chunk_cunter reached the training_interval.
    self.training_interval = 1
    self.chunk_counter = 0

    if self.parameters.multiprocesses:
      global main_pool
      main_pool = mp.Pool(initializer=self.init_process, processes=self.parameters.multiprocesses)

    self.evaluator = Evaluator(self.parameters)

    self.population = Population(self.parameters, init_size=self.parameters.pop_size)
    self.init_pop = True
    self.offsprings = None

    # Instantiate evolver
    self.init_evolver()

    # Instantiate scheduler
    self.init_scheduler()

    if self.ae is not None:
      if not os.path.exists(os.path.join(self.parameters.save_path, 'ae')):
        os.mkdir(os.path.join(self.parameters.save_path, 'ae'))

  def init_scheduler(self):
    """
    This function initializes the scheduler.
    This happens only if emitter based, otherwise the scheduler always gives exploration
    :return:
    """
    if self.evolver.emitter_based:
      if self.parameters.scheduler_type == 'alternating':
        self.scheduler = Alternating()
      elif self.parameters.scheduler_type == 'fixed_sampling':
        self.scheduler = FixedSampling()
      elif self.parameters.scheduler_type == 'time_sampling':
        self.scheduler = TimeSampling(total_budget=self.parameters.evaluation_budget)
      elif self.parameters.scheduler_type == 'mab':
        self.scheduler = DiscountedUCB(c=1, alpha=-1)
      else:
        raise ValueError("Scheduler {} not implemented.".format(self.parameters.scheduler_type))
    else:
      self.scheduler = Exploration()

  def init_evolver(self):
    """
    This function instantiates the evolver
    :return:
    """
    if self.parameters.exp_type == 'NS':
      self.evolver = NoveltySearch(self.parameters)

    elif self.parameters.exp_type == 'NSLC':
      self.evolver = NSLC(self.parameters)

    elif self.parameters.exp_type == 'LOGSIGN':
      self.evolver = NoveltySearch(self.parameters)

    elif self.parameters.exp_type == 'SIGN':
      self.evolver = NoveltySearch(self.parameters)

    elif self.parameters.exp_type == 'TIME-LOGSIGN':
      self.evolver = NoveltySearch(self.parameters)

    elif self.parameters.exp_type == 'TIME-SIGN':
      self.evolver = NoveltySearch(self.parameters)

    elif self.parameters.exp_type == 'CMA-ES':
      self.evolver = CMAES(self.parameters)
      # Generate CMA-ES initial population
      del self.population
      self.population = Population(self.parameters, init_size=self.parameters.emitter_population)
      for agent in self.population:
        agent['genome'] = self.evolver.optimizer.ask()

    elif self.parameters.exp_type == 'NSGA-II':
      self.evolver = NSGAII(self.parameters)

    elif self.parameters.exp_type == 'SERENE':
      self.evolver = SERENE(self.parameters)

    elif self.parameters.exp_type == 'ME':
      self.evolver = MAPElites(self.parameters)

    elif self.parameters.exp_type == 'CMA-ME':
      self.evolver = CMAME(self.parameters)

    elif self.parameters.exp_type == 'RND':
      self.evolver = RandomSearch(self.parameters)

    elif self.parameters.exp_type == 'TAXO-N': # This version only uses novelty
      self.evolver = NoveltySearch(self.parameters)
      self.ae = AE(encoding_shape=self.parameters.bd_size)
      self.train_ae = True

    elif self.parameters.exp_type == 'TAXO-S': # This version only uses surprise
      self.evolver = SurpriseSearch(self.parameters)
      self.ae = AE(encoding_shape=self.parameters.bd_size)
      self.train_ae = True

    elif self.parameters.exp_type == 'TAXONS': # This version choses randomly between novelty and surprise at each gen
      self.evolver = TaxonsSearch(self.parameters)
      self.ae = AE(encoding_shape=self.parameters.bd_size)
      self.train_ae = True

    elif self.parameters.exp_type == 'TAXONS-P': # This version choses randomly between novelty and surprise at each gen
      self.evolver = TaxonsPareto(self.parameters)
      self.ae = AE(encoding_shape=self.parameters.bd_size)
      self.train_ae = True

    elif self.parameters.exp_type == 'NT': # This version only uses novelty but the net is not trained
      self.evolver = NoveltySearch(self.parameters)
      self.ae = AE(encoding_shape=self.parameters.bd_size)
      self.train_ae = False

    elif self.parameters.exp_type == 'STAX':
      self.evolver = SereneTaxons(self.parameters)
      self.ae = AE(encoding_shape=self.parameters.bd_size)
      self.train_ae = True

    elif self.parameters.exp_type == 'STAX-NT': # This version only uses the novelty of the non trained AE
      self.evolver = STAX_NT(self.parameters)
      self.ae = AE(encoding_shape=self.parameters.bd_size)
      self.train_ae = False

    elif self.parameters.exp_type == 'STAX-P':
      self.evolver = STAXPareto(self.parameters)
      self.ae = AE(encoding_shape=self.parameters.bd_size)
      self.train_ae = True

    else:
      print("Experiment type {} not implemented.".format(self.parameters.exp_type))
      raise ValueError

  def init_process(self):
    """
    This function is used to initialize the pool so each process has its own instance of the evaluator
    :return:
    """
    global evaluator
    evaluator = Evaluator(self.parameters)

  def _feed_eval(self, agent):
    """
    This function feeds the agent to the evaluator and returns the updated agent
    :param agent:
    :return:
    """
    global evaluator
    if agent['evaluated'] == None: # Agents are evaluated only once
      agent = evaluator(agent, self.bd_extractor.__call__)
    return agent

  def evaluate_in_env(self, pop, pool=None):
    """
    This function evaluates the population in the environment by passing it to the parallel evaluators.
    :return:
    """
    if self.parameters.verbose: print('Evaluating {} in environment.'.format(pop.name))
    if pool is not None:
      pop.pop = pool.map(self._feed_eval, pop.pop) # As long as the ID is fine, the order of the element in the list does not matter
    else:
      for i in range(pop.size):
        if self.parameters.verbose: print(".", end = '') # The end prevents the newline
        if pop[i]['evaluated'] is None: # Agents are evaluated only once
          pop[i] = self.evaluator(pop[i], self.bd_extractor)
      if self.parameters.verbose: print()

  def extract_bd(self, pop):
    """
    This function uses the autoencoder to extract the BD from the views
    :param pop:
    :return:
    """
    if len(pop) > 0:
      # Flag for experiment on comparing STAX_single and STAX_multi performances
      # If activated the AE is trained on multiple samples but the BD is only from the final frame
      if self.parameters.single_multi_exp:
        if isinstance(pop, dict):
          views = np.array(pop['view'])[-1:]
        else:
          views = np.stack(pop['view'])[:, -1]
      else:
        if isinstance(pop, dict):
          views = np.array(pop['view'])
        else:
          views = np.concatenate(pop['view'])

      # This one is needed when updating the emitters ancestor.
      # In this case in fact we don't pass a pop but just an agent, so we need to add the batch dim usually given by the pop
      if views.ndim == 3: views = np.expand_dims(views, axis=0)

      batch_idx = [0, self.parameters.batch_size]
      surprises = []
      bds = []
      # Elaborate batches
      while batch_idx[0] < len(views):
        batch = views[batch_idx[0]:min(batch_idx[1], len(views))]

        with torch.no_grad():
          batch = torch.Tensor(batch).permute((0, 3, 1, 2)).contiguous() # NWHC -> NCWH
          ae_output = self.ae(batch.to(self.ae.device))
          bds.append(np.atleast_2d(ae_output['feat'].cpu().detach().numpy()))
          surprises.append(ae_output['error'].cpu().detach().numpy())

        batch_idx[0] = batch_idx[1]
        batch_idx[1] += self.parameters.batch_size

      # Merge the batches dimensions
      bds = np.concatenate(bds)
      surprises = np.concatenate(surprises)

      # UPDATE DATA
      # Flag for experiment on comparing STAX_single and STAX_multi performances
      # If activated the AE is trained on multiple samples but the BD is only from the final frame
      if self.parameters.single_multi_exp:
        if isinstance(pop, dict):
          pop['bd'] = bds[0]
          pop['surprise'] = surprises[0]
        else:
          pop['bd'] = bds
          pop['surprise'] = surprises
      else:
        if isinstance(pop, dict):  # If this is a single agent
          pop['bd'] = np.reshape(bds, (self.parameters.samples_per_traj * self.parameters.bd_size), order='C')
          pop['surprise'] = np.sum(surprises)
        else:
          pop['bd'] = np.reshape(bds, (pop.size, self.parameters.samples_per_traj*self.parameters.bd_size), order='C') # [pop*samples, bd] -> [pop, samples*bd]
          pop['surprise'] = np.sum(np.reshape(surprises, (pop.size, self.parameters.samples_per_traj), order='C'), axis=1)  # Sum along the samples dim

  def _main_search(self, budget_chunk):
    """
    This function performs the main search e.g. NS/NSGA/CMA-ES
    :return:
    """
    # The cost of the exploration process that will be used in the scheduler is given by the increase in initialized emitters
    # The reason behaind this is that if there are too many init emitters, it is time to perform exploitation
    # cause a possible new reward area has been found.
    # The cost is transformed into a reward by passing it in a tanh and using its negative as reward.
    old_emitters = len(self.evolver.emitter_candidate) + len(self.evolver.emitters)

    # Evaluate population in the environment only the first time
    if self.init_pop:
      self.evaluate_in_env(self.population, pool=main_pool)
      self.population['evaluated'] = list(range(self.evolver.evaluated_points, self.evolver.evaluated_points + self.population.size))
      self.evolver.evaluated_points += self.population.size
      self.evolver.evaluation_budget -= self.population.size
      budget_chunk -= self.population.size
      self.init_pop = False

      # Get BD of initial population
      if self.ae is not None:
        self.extract_bd(self.population)

      if not self.evolver.emitter_based:
        for area in self.population['rew_area']:
          if area is not None:
            name = 'rew_area_{}'.format(area)
            if name not in Logger.data:
              Logger.data[name] = 0
            Logger.data[name] += 1

    while budget_chunk > 0 and self.evolver.evaluation_budget > 0:
      self.offsprings = self.evolver.generate_offspring(self.population, pool=None, generation=self.generation)  # Generate offsprings

      # Evaluate offsprings in the env
      self.evaluate_in_env(self.offsprings, pool=main_pool)
      self.offsprings['evaluated'] = list(range(self.evolver.evaluated_points, self.evolver.evaluated_points + self.offsprings.size))
      self.evolver.evaluated_points += self.offsprings.size
      self.evolver.evaluation_budget -= self.offsprings.size
      budget_chunk -= self.offsprings.size

      # Calculate BD for offsprings
      if self.ae is not None:
        self.extract_bd(self.offsprings)

      # We init the emitters after the bd has been calculated so the BD is there already
      self.evolver.init_emitters(self.population, self.offsprings)

      # Evaluate performances of pop and off and update archive
      self.evolver.evaluate_performances(self.population, self.offsprings, pool=main_pool)  # Calculate novelty/fitness/curiosity etc
      # Only update archive using NS stuff. No archive candidates from emitters
      self.evolver.update_archive(self.population, self.offsprings, generation=self.generation)

      # Save pop, archive and off
      if self.generation % 1 == 0:
        self.population.save(self.parameters.save_path, 'gen_{}'.format(self.generation))
        self.evolver.archive.save(self.parameters.save_path, 'gen_{}'.format(self.generation))
        self.offsprings.save(self.parameters.save_path, 'gen_{}'.format(self.generation))

      # Log reward only if not emitter based
      if not self.evolver.emitter_based:
        for area in self.offsprings['rew_area']:
          if area is not None:
            name = 'rew_area_{}'.format(area)
            if name not in Logger.data:
              Logger.data[name] = 0
            Logger.data[name] += 1

      # Last thing we do is to update the population
      self.generation += 1
      self.evolver.update_population(self.population, self.offsprings, generation=self.generation)

    reward = -np.tanh(len(self.evolver.emitter_candidate) + len(self.evolver.emitters) - old_emitters)
    return reward

  def _emitter_search(self, budget_chunk):
    """
    This function performs the reward search through the emitters
    :return:
    """
    reward = -1 # Reward is squshed between -1 and 0
    if self.evolver.emitter_based and (len(self.evolver.emitters) > 0 or len(self.evolver.emitter_candidate) > 0):
      old_emitters = len(self.evolver.emitters) + len(self.evolver.emitter_candidate)
      self.evolver.emitter_step(self.evaluate_in_env,
                                self.generation,
                                ns_pop=self.population,
                                ns_off=self.offsprings,
                                budget_chunk=budget_chunk,
                                pool=None,
                                extract_bd=self.extract_bd)
      self.evolver.rew_archive.save(self.parameters.save_path, 'gen_{}'.format(self.generation))
      # The reward is calculated as the change in candidate emitters. If many emitters have been added to the active emitters list,
      # it means that more exploitation needs to be performed
      reward = np.tanh(len(self.evolver.emitter_candidate) + len(self.evolver.emitters) - old_emitters)

      # Update the performaces due to possible changes in the pop and archive given by the emitters
      self.evolver.evaluate_performances(self.population, self.offsprings, pool=None)
      # Update main archive with the archive candidates from the emitters
      self.evolver.elaborate_archive_candidates(self.generation)

    return reward

  def _train_ae(self):
    """
    This function is used to train the AE.
    :return:
    """
    print("Training of AE...")
    training_data, validation_data = self.get_training_data()

    training_data_idx = np.array(range(len(training_data)))
    validation_data_idx = np.array(range(len(validation_data)))

    prev_valid_error = np.inf
    valid_increase = 0 # Counter for how many times the validation error increased
    training_epoch = 0

    while training_epoch <= self.parameters.max_training_epochs:
      training_epoch += 1

      # Training cycle
      # -------------------------------------------------------------
      batch_idx = [0, self.parameters.batch_size]
      np.random.shuffle(training_data_idx) # These are reshuffled at every epoch
      training_total_loss = 0
      training_rec_loss = 0
      training_kl_div = 0
      training_step = 0

      # Yes it's idx_batch[0]< . I tested and it gets all elements, so no need to repeat stuff out of the while
      while batch_idx[0] < len(training_data):
        batch = training_data[training_data_idx[batch_idx[0]:min(batch_idx[1], len(training_data))]]
        batch = torch.Tensor(batch).permute((0, 3, 1, 2)) # NHWC -> NCHW

        loss = self.ae.training_step(batch.to(self.ae.device))
        batch_idx[0] = batch_idx[1]
        batch_idx[1] += self.parameters.batch_size

        training_total_loss += loss['total loss']
        training_rec_loss += loss['rec loss']
        if loss['kl div'] is not None:
          training_kl_div += loss['kl div']
        training_step += 1

      # if self.parameters.verbose:
      print("Training Loss: {} - Rec Loss: {} - KL div: {}".format(training_total_loss / training_step,
                                                                     training_rec_loss/training_step,
                                                                     training_kl_div/training_step))
      # -------------------------------------------------------------

      # Validation
      # -------------------------------------------------------------
      if training_epoch % self.parameters.validation_interval == 0 and training_epoch > 1:
        batch_idx = [0, self.parameters.batch_size]
        errors = [] # For the valid we calculate as error the mean over all the rec errors. We store them in a list cause the data is batched
        # No need to shuffle the idx cause no learning happens
        # We still batch it so to help with memory constraints
        while batch_idx[0] < len(validation_data):
          batch = validation_data[validation_data_idx[batch_idx[0]:min(batch_idx[1], len(validation_data))]]

          with torch.no_grad():
            batch = torch.Tensor(batch).permute((0, 3, 1, 2))  # NHWC -> NCHW
            ae_output = self.ae.forward(batch.to(self.ae.device))

          batch_idx[0] = batch_idx[1]
          batch_idx[1] += self.parameters.batch_size
          errors.append(ae_output['error'].cpu().numpy())

        validation_error = np.mean(np.concatenate(errors))
        if self.parameters.verbose: print('Validation error: {}'.format(validation_error))

        # Check validation error increase
        if validation_error > prev_valid_error:
          if valid_increase < 3:
            valid_increase += 1
            if self.parameters.verbose: print("Valid error consecutive increases: {}".format(valid_increase))
          else:
            if self.parameters.verbose: print("\nValidation error increased. Stopping.")
            gc.collect()
            break
        else:
          valid_increase = 0
          prev_valid_error = validation_error
      # -------------------------------------------------------------

    print("Final training error {}".format(training_total_loss / training_step))
    print("Final validation error {}".format(validation_error))

    self.ae.save(os.path.join(self.parameters.save_path, 'ae'), 'ae_gen_{}'.format(self.generation))
    return 0

  def get_training_data(self):
    """
    This function collects the training data for the AE
    No weighting or sampling happens for now. All the data from the archive are collected
    :return:
    """
    valid_percentage = .2  # 20% of the data is used for validation
    if not self.parameters.archive_sampling:
      data = self.evolver.archive['view'] + self.population['view'] + self.offsprings['view']
      if hasattr(self.evolver, 'rew_archive') and self.parameters.train_on_reward:
        data = data + self.evolver.rew_archive['view']
    else:
      data = self.population['view'] + self.offsprings['view']

      if self.evolver.archive.size > len(data):
        arch_views = np.array(self.evolver.archive['view'])
        arch_samples = np.random.choice(range(len(arch_views)),
                                        size=len(data),
                                        replace=True,)
        data = data + list(arch_views[arch_samples])
      else:
        data = data + self.evolver.archive['view']
    data = np.concatenate(data)

    # Shuffle and split data
    indexes = list(range(len(data)))
    np.random.shuffle(indexes)
    validation_data = data[indexes[:int(len(data) * valid_percentage)]]
    training_data = data[indexes[int(len(data) * valid_percentage):]]
    del data

    return training_data, validation_data

  def chunk_step(self):
    """
    This function performs all the calculations needed for one generation.
    Generates offsprings, evaluates them and the parents in the environment, calculates the performance metrics,
    updates archive and population and finally saves offsprings, population and archive.
    :return: time taken for running the generation
    """
    global main_pool
    start_time = timer()

    print("\nRemaining budget: {}".format(self.evolver.evaluation_budget))

    # -------------------
    # Perform step
    # -------------------
    choice = self.scheduler.action()
    # print("CHOICE: {} - Q: {}".format(choice, self.scheduler.Q))
    # print()

    if choice == 'explore':
      self.chunk_counter += 1 # This is updated only when performing the exploration step. Cause the AE works with exploration data. so it makes sense
      # Explore
      budget_chunk = self.parameters.chunk_size
      if self.evolver.evaluation_budget > 0:
        if self.parameters.verbose: print("MAIN")
        reward = self._main_search(budget_chunk) # TODO implement reward calculation
        if self.parameters.verbose: print("Exploration Rew: {}".format(reward))

    elif choice == 'exploit':
      # Emitters part
      budget_chunk = self.parameters.chunk_size
      # Starts only if a reward has been found.
      if self.evolver.evaluation_budget > 0:
        if self.parameters.verbose: print("EMITTERS: {}".format(len(self.evolver.emitters)))
        reward = self._emitter_search(budget_chunk) # TODO implement reward calculation
        if self.parameters.verbose: print("Exploitation Rew: {}".format(reward))

    else:
      raise ValueError('Wrong scheduler choice given. Given: {} - Available: [explore, exploit]'.format(choice))

    self.scheduler.update(evolver=self.evolver, reward=reward)
    # -------------------


    # -------------------
    # Update AE and BDS
    # -------------------
    if self.ae is not None and self.chunk_counter == self.training_interval:
      if self.parameters.reset_ae:
        self.ae = AE(encoding_shape=self.parameters.bd_size)

      if self.train_ae:
        self._train_ae()

      # Now update the BDs of pop, off and arch
      self.extract_bd(self.population)
      self.extract_bd(self.offsprings)
      self.extract_bd(self.evolver.archive)
      if hasattr(self.evolver, 'rew_archive') and self.evolver.rew_archive.size > 0:
        self.extract_bd(self.evolver.rew_archive)

      # Update emitter related buffers
      if self.evolver.emitter_based:
        # We do not update the self.archive_candidates of dead emitters cause it's empty given that is elaborate and
        # emptied at the end of the emitter step.
        # But we update the ns_cand_buffer for each living emitter.
        # At the same time we update the BD of the ancestor and the most novel.
        # No need to update the pop of the emitters cause their novelty is not needed and the new offspring will have
        # the BD calculated with the new AE
        for em in self.evolver.emitter_candidate:
          self.extract_bd(self.evolver.emitter_candidate[em].ns_arch_candidates)
          self.extract_bd(self.evolver.emitter_candidate[em].ancestor)
          self.extract_bd(self.evolver.emitter_candidate[em].most_novel)
        for em in self.evolver.emitters:
          self.extract_bd(self.evolver.emitters[em].ns_arch_candidates)
          self.extract_bd(self.evolver.emitters[em].ancestor)
          self.extract_bd(self.evolver.emitters[em].most_novel)

      # Update counters
      self.chunk_counter = 0
      self.training_interval += 1
    # -------------------

    return timer() - start_time, self.evolver.evaluated_points

  def load_generation(self, generation, path):
    """
    This function loads the population, the offsprings and the archive at a given generation, so it can restart the
    search from there.
    :param generation:
    :param path: experiment path
    :return:
    """
    self.generation = generation

    self.population.load(os.path.join(path, 'population_gen_{}.pkl'.format(self.generation)))
    self.offsprings.load(os.path.join(path, 'offsprings_gen_{}.pkl'.format(self.generation)))
    self.evolver.archive.load(os.path.join(path, 'archive_gen_{}.pkl'.format(self.generation)))

  def close(self):
    """
    This function closes the pool and deletes everything.
    :return:
    """
    if self.parameters.multiprocesses:
      global main_pool
      main_pool.close()
      main_pool.join()
    gc.collect()
