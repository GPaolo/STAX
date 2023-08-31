# Created by Giuseppe Paolo 
# Date: 16/04/2021

class Exploration(object):
  """
  This 'scheduler' will always choose exploraiton. Used for the non emitter based approaches
  """
  def action(self, **kwargs):
    return 'explore'

  def update(self, **kwargs):
    return