# Created by Giuseppe Paolo 
# Date: 16/04/2021

class Alternating(object):
  """
  This implements the alternating scheduler used by vanilla SERENE
  """
  def __init__(self, **kwargs):
    self.choice = None

  def action(self, **kwargs):
    if self.choice == 'exploit' or self.choice is None:
      self.choice = 'explore'
    else:
      self.choice = 'exploit'
    return self.choice

  def update(self, **kwargs):
    return