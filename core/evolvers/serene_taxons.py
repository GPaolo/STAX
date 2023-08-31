# Created by Giuseppe Paolo 
# Date: 26/02/2021

from core.evolvers import SERENE, TaxonsSearch, NSGAII
from core.evolvers import utils
import numpy as np
from analysis.logger import Logger
import copy


class SereneTaxons(SERENE, TaxonsSearch):
  """
  This class implements serene where the BD is obtained through TAXONS.
  The class directly inherits from FitNS and TAXONS and merges their capabilities.

  ---------
  The init is taken from the FitNS, so to add all the emitters things
  ---------
  Evaluate performance method uses the one from FitNS. Compared to TaxonsSearch this one
  uses as novelty_ref_set also the reward archive.
  ---------
  The archive is updated with the TAXONS method. This is because we also have to choose the
  criteria
  ---------
  The population is updated with the TAXONS method, so to use the chosen criteria

  """
  def __init__(self, parameters, **kwargs):
    SERENE.__init__(self, parameters, **kwargs)
    TaxonsSearch.__init__(self, parameters, **kwargs)
    self.emitter_based = True

  def evaluate_performances(self, population, offsprings, pool=None):
    SERENE.evaluate_performances(self, population, offsprings, pool=pool)

  def update_archive(self, population, offsprings, generation):
    TaxonsSearch.update_archive(self, population, offsprings, generation)

  def update_population(self, population, offsprings, generation):
    TaxonsSearch.update_population(self, population, offsprings, generation)

  def emitter_step(self, evaluate_in_env, generation, ns_pop, ns_off, budget_chunk, pool=None, extract_bd=None, **kwargs):
    """
    This function performs the emitter step. It does the same stuff as FitNS but calculates the bd using the ae for
    the emitters pops
    :param evaluate_in_env:
    :param generation:
    :param ns_pop:
    :param ns_off:
    :param budget_chunk:
    :param pool:
    :param extract_bd: Function that uses the ae to extract the bd
    :param kwargs:
    :return:
    """
    ns_reference_set = self.get_novelty_ref_set(ns_pop, ns_off)

    budget_chunk = self.candidate_emitter_eval(evaluate_in_env=evaluate_in_env,
                                               budget_chunk=budget_chunk,
                                               ns_reference_set=ns_reference_set,
                                               generation=generation,
                                               extract_bd=extract_bd,
                                               pool=pool)
    # Update rew_archive bd
    # TODO doing it here is very inefficient given that we only want to update the bd of the newly added ones
    extract_bd(self.rew_archive)


    while self.emitters and budget_chunk > 0 and self.evaluation_budget > 0: # Till we have emitters or computation budget
      emitter_idx = self.choose_emitter()
      # Update emitter pop bd (it could be that there are still elements from the first pop that don't have the bd)
      extract_bd(self.emitters[emitter_idx].pop)

      # Calculate parent novelty
      self.emitters[emitter_idx].ancestor['novelty'] = utils.calculate_novelties([self.emitters[emitter_idx].ancestor['bd']], # The ancestor already has its bd
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
        # Now update the BD of the offs. This way no need to update the rew_archive bd
        extract_bd(offsprings)

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

        if self.check_stopping_criteria(emitter_idx):  # Only if emitter is finished
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


class STAXPareto(SereneTaxons, NSGAII):
  """
  This class implements the pareto version of Serene+Taxons. It uses the same approach of NSGA to calculate the fronts
  between Novelty and Surprise and select the population.
  """
  def __init__(self, parameters, **kwargs):
    SereneTaxons.__init__(self, parameters, **kwargs)
    NSGAII.__init__(self, parameters, **kwargs)
    self.update_criteria = ['novelty', 'surprise']
    self.emitter_based = True

  def evaluate_performances(self, population, offsprings, pool=None):
    SERENE.evaluate_performances(self, population, offsprings, pool=pool)

  def update_population(self, population, offsprings, generation):
    NSGAII.update_population(self, population, offsprings, generation)

class STAX_NT(SereneTaxons):
  """
  This class implements STAX in which the AE is not trained. This means no reconstruction error is used
  """
  def __init__(self, parameters, **kwargs):
    SereneTaxons.__init__(self, parameters, **kwargs)
    self.update_criteria = ['novelty']
    self.emitter_based = True

  def update_archive(self, population, offsprings, generation):
    SERENE.update_archive(self, population, offsprings, generation)

  def update_population(self, population, offsprings, generation):
    SERENE.update_archive(self, population, offsprings, generation)
