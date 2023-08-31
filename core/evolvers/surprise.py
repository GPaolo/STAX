# Created by Giuseppe Paolo 
# Date: 17/02/2021

from core.evolvers import BaseEvolver
from core.evolvers import utils

class SurpriseSearch(BaseEvolver):
  """
  This class implements the surprise search. It uses the surprise as update criteria to generate the new population
  """
  def __init__(self, parameters):
    super().__init__(parameters)
    self.update_criteria = 'surprise'

  def evaluate_performances(self, population, offsprings, pool=None):
    """
    No performance need to be evaluated, given that the surprise is already evaluated by the AE.
    :param population:
    :param offsprings:
    :param pool:
    :return:
    """
    return