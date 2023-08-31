# Created by Giuseppe Paolo 
# Date: 16/02/2021

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys


# ----------------------------------------------------------------
class View(nn.Module):
  """
  This class is used to reshape tensors
  """
  def __init__(self, size):
    """
    Constructor
    :param size: final shape of the tensor
    """
    super(View, self).__init__()
    self.size = size

  def forward(self, tensor):
    return tensor.view(self.size)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
class BaseAE(nn.Module):
  """
  This class defines the Base AutoEncoder.
  """
  # ----------------------------------------------------------------
  def __init__(self, encoding_shape=10, device=None, learning_rate=0.001):
    """
    Constructor
    :param device: Device on which run the computation
    :param learning_rate:
    :param lr_scale: Learning rate scale for the lr scheduler
    :param encoding_shape: Shape of the feature space
    """
    super(BaseAE, self).__init__()

    if device is not None:
      self.device = device
    else:
      # The model is always on the gpu.
      # In the multiprocessing it occupies MUCH more memory but is also MUCH faster
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.encoding_shape = encoding_shape
    # Model definition is done in these functions that are to be overridden
    self._define_encoder()
    self._define_decoder()

    self.rec_loss = nn.MSELoss(reduction='none')
    self.learning_rate = learning_rate
    self.zero_grad()

    self.optimizer = optim.Adam(self.parameters(), self.learning_rate)
    self.to(self.device)
    self.eval()
    self.SEQUENTIAL = False
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def _define_encoder(self):
    """
    Define encoder. Needs to be implemented in inheriting classes
    """
    raise NotImplementedError

  def _define_decoder(self):
    """
    Define decoder. Needs to be implemented in inheriting classes
    """
    raise NotImplementedError
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def save(self, filepath, name='ckpt_ae'):
    """
    Saves AE to given filepath
    :param filepath:
    :param name: name to give the file
    """
    save_ckpt = {
      'ae': self.state_dict(),
      'optimizer': self.optimizer.state_dict()
    }
    try:
      torch.save(save_ckpt, os.path.join(filepath, '{}.pth'.format(name)))
    except:
      print('Cannot save autoencoder.')
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def load(self, filepath):
    """
    Load AE from given filepath
    :param filepath:
    """
    try:
      ckpt = torch.load(filepath, map_location=self.device)
    except Exception as e:
      print('Could not load file: {}'.format(e))
      sys.exit()
    try:
      self.load_state_dict(ckpt['ae'])
    except Exception as e:
      print('Could not load model state dict: {}'.format(e))
      sys.exit()
    try:
      self.optimizer.load_state_dict(ckpt['optimizer'])
    except Exception as e:
      print('Could not load optimizer state dict: {}'.format(e))
      sys.exit()
  # ----------------------------------------------------------------

  # ----------------------------------------------------------------
  def training_step(self, **kwargs):
    """
    Function that performs one training step. Needs to be implemented in inheriting classes
    :param kwargs:
    """
    raise NotImplementedError
  # ----------------------------------------------------------------
# ----------------------------------------------------------------
