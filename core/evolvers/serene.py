# Created by giuseppe
# Date: 05/10/20

from core.evolvers import EmitterEvolver
from core.evolvers import utils
from core.emitters import FitnessEmitter, NoveltyEmitter, CMAESEmitter
from core.population import Population, Archive
import numpy as np
import copy
from analysis.logger import Logger

class SERENE(EmitterEvolver):
  """
  This class implements the FIT-NS evolver. It performs NS till a reward is found, then launches fitness based emitters
  to search the reward area.
  """
  def create_emitter(self, parent_id, ns_pop, ns_off):
    """
    This function creates the emitter
    :param parent_id:
    :param ns_pop:
    :param ns_off:
    :return:
    """
    if self.params.emitter_type == 'fitness':
      Emitter = FitnessEmitter
    elif self.params.emitter_type == 'novelty':
      Emitter = NoveltyEmitter
    elif self.params.emitter_type == 'cma-es':
      Emitter = CMAESEmitter
    else:
      raise ValueError('Wrong emitter type given: {}'.format(self.params.emitter_type))

    return Emitter(ancestor=self.rewarding[parent_id].copy(),
                   mutation_rate=self.calculate_init_sigma(ns_pop, ns_off, self.rewarding[parent_id]),
                   parameters=self.params)

  def candidate_emitter_eval(self, evaluate_in_env, budget_chunk, ns_reference_set, generation, extract_bd=None, pool=None):
    """
    This function does a small evaluation for the cadidate emitters to calculate their initial improvement.
    Estract_bd has to be passed whenever we use an algo that uses an AE to obtain the bd.
    Example:
      STAX has to pass it otherwise we cannot calculate the novelty needed for the possible novelty emitter
      SERENE does not have to pass it cause the BD is already calculated

    :return:
    """

    candidates = self.candidates_by_novelty(pool=pool)

    for candidate in candidates:
      # Bootstrap candidates improvements
      if budget_chunk <= self.params.chunk_size/3 or self.evaluation_budget <= 0:
        break

      # Initial population evaluation
      evaluate_in_env(self.emitter_candidate[candidate].pop, pool=pool)
      self.emitter_candidate[candidate].pop['evaluated'] = list(range(self.evaluated_points,
                                                                      self.evaluated_points + self.emitter_candidate[candidate].pop.size))
      self.emitter_candidate[candidate].values.append(self.emitter_candidate[candidate].pop['reward'])

      if extract_bd is not None:
        extract_bd(self.emitter_candidate[candidate].pop)
      novelties = utils.calculate_novelties(self.emitter_candidate[candidate].pop['bd'],
                                            ns_reference_set,
                                            distance_metric=self.params.novelty_distance_metric,
                                            novelty_neighs=self.params.novelty_neighs)
      self.emitter_candidate[candidate].pop['novelty'] = novelties

      # Update counters
      self.evaluated_points += self.params.emitter_population
      self.evaluation_budget -= self.params.emitter_population
      budget_chunk -= self.params.emitter_population
      rew_area = 'rew_area_{}'.format(self.emitter_candidate[candidate].ancestor['rew_area'])
      if rew_area not in Logger.data:
        Logger.data[rew_area] = 0
      Logger.data[rew_area] += self.params.emitter_population

      for i in range(5): # Evaluate emitter on 6 generations
        offsprings = self.emitter_candidate[candidate].generate_new_solutions(generation)
        evaluate_in_env(offsprings, pool=pool)
        offsprings['evaluated'] = list(range(self.evaluated_points, self.evaluated_points + offsprings.size))

        if extract_bd is not None:
          extract_bd(offsprings)
        novelties = utils.calculate_novelties(offsprings['bd'],
                                              ns_reference_set,
                                              distance_metric=self.params.novelty_distance_metric,
                                              novelty_neighs=self.params.novelty_neighs)
        offsprings['novelty'] = novelties

        self.emitter_candidate[candidate].update_pop(offsprings)
        self.emitter_candidate[candidate].values.append(self.emitter_candidate[candidate].pop['reward'])
        self.update_reward_archive(generation, self.emitter_candidate, candidate)

        # Update counters
        # step_count += 1
        self.emitter_candidate[candidate].steps += 1
        self.evaluated_points += offsprings.size
        self.evaluation_budget -= offsprings.size
        budget_chunk -= offsprings.size
        Logger.data[rew_area] += offsprings.size

      self.emitter_candidate[candidate].estimate_improvement()

      # Add to emitters list
      if self.emitter_candidate[candidate].improvement > 0:
        self.emitters[candidate] = copy.deepcopy(self.emitter_candidate[candidate])
      del self.emitter_candidate[candidate]
    return budget_chunk

  def emitter_step(self, evaluate_in_env, generation, ns_pop, ns_off, budget_chunk, pool=None, **kwargs):
    """
    This function performs the steps for the CMA-ES emitters
    :param **kwargs:
    :param evaluate_in_env: Function used to evaluate the agents in the environment
    :param generation: Generation at which the process is
    :param ns_pop: novelty search population
    :param ns_off: novelty search offsprings
    :param budget_chunk: budget chunk to allocate to search
    :param pool: Multiprocessing pool
    :return:
    """
    ns_reference_set = self.get_novelty_ref_set(ns_pop, ns_off)

    budget_chunk = self.candidate_emitter_eval(evaluate_in_env=evaluate_in_env,
                                               budget_chunk=budget_chunk,
                                               ns_reference_set=ns_reference_set,
                                               generation=generation,
                                               extract_bd=None,
                                               pool=pool)

    while self.emitters and budget_chunk > 0 and self.evaluation_budget > 0: # Till we have emitters or computation budget
      emitter_idx = self.choose_emitter()

      # Calculate parent novelty
      self.emitters[emitter_idx].ancestor['novelty'] = utils.calculate_novelties([self.emitters[emitter_idx].ancestor['bd']],
                                                                                 ns_reference_set,
                                                                                 distance_metric=self.params.novelty_distance_metric,
                                                                                 novelty_neighs=self.params.novelty_neighs,
                                                                                 pool=pool)[0]

      print("Emitter: {} - Improv: {}".format(emitter_idx, self.emitters[emitter_idx].improvement))
      rew_area = 'rew_area_{}'.format(self.emitters[emitter_idx].ancestor['rew_area'])

      # The emitter evaluation cycle breaks every X steps to choose a new emitter
      # ---------------------------------------------------
      while budget_chunk > 0 and self.evaluation_budget > 0:
        offsprings = self.emitters[emitter_idx].generate_new_solutions(generation)
        evaluate_in_env(offsprings, pool=pool)

        offsprings['evaluated'] = list(range(self.evaluated_points, self.evaluated_points + offsprings.size))
        novelties = utils.calculate_novelties(offsprings['bd'],
                                              ns_reference_set,
                                              distance_metric=self.params.novelty_distance_metric,
                                              novelty_neighs=self.params.novelty_neighs)
        offsprings['novelty'] = novelties

        self.emitters[emitter_idx].update_pop(offsprings)
        self.emitters[emitter_idx].values.append(self.emitters[emitter_idx].pop['reward'])
        self.update_reward_archive(generation, self.emitters, emitter_idx)

        # Now calculate novelties and update most novel
        self.update_novelty_cand_buff(ns_ref_set=ns_reference_set,
                                      ns_pop=ns_pop, ns_off=ns_off,
                                      emitter_idx=emitter_idx, pool=pool)
        # Update counters
        # step_count += 1
        self.emitters[emitter_idx].steps += 1
        self.evaluated_points += offsprings.size
        self.evaluation_budget -= offsprings.size
        budget_chunk -= offsprings.size
        Logger.data[rew_area] += offsprings.size

        if self.check_stopping_criteria(emitter_idx): # Only if emitter is finished
          self.emitters_data[int(emitter_idx)] = {'generation': generation,
                                                  'steps': self.emitters[emitter_idx].steps,
                                                  'rewards': self.emitters[emitter_idx].values,
                                                  'archived': self.emitters[emitter_idx].archived}

          self.archive_candidates[emitter_idx] = copy.deepcopy(self.emitters[emitter_idx].ns_arch_candidates)
          # Store parent once the emitter is finished
          self.rew_archive.store(self.emitters[emitter_idx].ancestor)
          print("Stopped after {} steps\n".format(self.emitters[emitter_idx].steps))
          del self.emitters[emitter_idx]
          break
        # ---------------------------------------------------
      # This is done only if the emitter still exists
      if emitter_idx in self.emitters:
        self.emitters[emitter_idx].estimate_improvement()