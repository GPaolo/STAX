# Created by Giuseppe Paolo 
# Date: 28/07/2020

import numpy as np
import iisignature
from environments import registered_envs

# TODO THE BEHAVIOR DESCRIPTOR SHOULD BE GIVEN AS PARAMETER

sign = None # If this one is not global mp will have issues in pickling stuff


class BehaviorDescriptor(object):
  """
  This class defines the behavior descriptor.
  Each BD function returns a behavior descriptor and the surprise associated with it.
  (Have to do it here cause of efficiency reasons, otherwise will have to rerun the images through the autoencoder later)
  To add more BD add the function and the experiment type in the init.
  """
  def __init__(self, parameters):
    self.params = parameters

    if self.params.env_name not in registered_envs:
      raise NameError("Unknown environment. Given {} - Available {}".format(self.params.env_name, list(registered_envs.keys())))
    self.env = registered_envs[self.params.env_name]
    # Function to extract the observations from the complete trajectory
    self.traj_to_obs = self.env['traj_to_obs']

    try:
      self.descriptor = getattr(self, self.params.behavior_descriptor)
    except:
      raise ValueError("No behavior descriptor defined for experiment type {}".format(self.params.exp_type))

  def __call__(self, traj, agent, **kwargs):
    return self.descriptor(traj, agent, **kwargs)

  def full_trajectory(self, traj, agent):
    """
    This function returns the whole trajectory of states as behavior descriptor
    :param traj: complete trajectory consisting in a list of [obs, rew, done, info]
    :param agent:
    :return: BD
    """
    obs = self.traj_to_obs(traj)
    return obs

  def full_scaled_trajectory(self, traj, agent):
    """
    This function returns the whole trajectory of states as behavior descriptor. The traj is scaled
    :param traj: complete trajectory consisting in a list of [obs, rew, done, info]
    :param agent:
    :return: BD
    """
    obs = self.traj_to_obs(traj)
    obs = (obs - np.min(obs))/(np.max(obs) - np.min(obs)) # TODO This is not good, have to improve
    return obs

  def last_obs(self, traj, agent):
    """
    This function returns the last observation extracted from the trajectory
    :param traj: complete trajectory consisting in a list of [obs, rew, done, info]
    :param agent:
    :return: ground truth BD
    """
    obs = self.traj_to_obs(traj)
    idx = np.linspace(0, len(obs)-1, self.params.samples_per_traj+1, dtype=int)[1:] # We exclude the obs[0] cause it's the same for all traj
    return np.array([self.traj_to_obs(traj)[i] for i in idx]).flatten()

  def time_logsign_bd(self, traj, agent):
    """
    Extract the BD from the trajectory by calculating its signature with the time embedding
    :param agent:
    :return:
    """
    # Get observations with appended timesignal. This way signs are unique.
    obs_traj = self.traj_to_obs(traj)
    time_signal = np.linspace(0, len(traj), num=len(obs_traj)) / self.traj_max_len # Scaled to traj_max_len
    time_signal = np.expand_dims(time_signal, axis=-1)
    obs_traj = np.append(obs_traj, time_signal, axis=-1)

    # This is to prepare the logsig just once.
    global sign
    if sign is None:
      sign = iisignature.prepare(len(obs_traj[0]), self.params.signature_order)

    signature = iisignature.logsig(obs_traj, sign)
    return signature

  def logsign_bd(self, traj, agent):
    """
    Extract the BD from the trajectory by calculating its signature with the linear embedding
    :param agent:
    :return:
    """
    obs_traj = self.traj_to_obs(traj)

    # This is to prepare the logsig just once.
    global sign
    if sign is None:
      sign = iisignature.prepare(len(obs_traj[0]), self.params.signature_order)

    signature = iisignature.logsig(obs_traj, sign)
    return signature

  def sign_bd(self, traj, agent):
    """
        Extract the BD from the trajectory by calculating its signature with the linear embedding
        :param agent:
        :return:
    """
    obs_traj = self.traj_to_obs(traj)

    signature = iisignature.sig(obs_traj, self.params.signature_order)
    return signature

  def time_sign_bd(self, traj, agent):
    """
    Extract the BD from the trajectory by calculating its signature with the time embedding
    :param agent:
    :return:
    """
    # Get observations with appended timesignal. This way signs are unique.
    obs_traj = self.traj_to_obs(traj)
    time_signal = np.linspace(0, len(traj), num=len(obs_traj)) / self.traj_max_len # Scaled to traj_max_len
    time_signal = np.expand_dims(time_signal, axis=-1)
    obs_traj = np.append(obs_traj, time_signal, axis=-1)

    signature = iisignature.sig(obs_traj, self.params.signature_order)
    return signature

  def dummy_bd(self, traj, agent):
    """
    This function implements a dummy bd for algorithms that do not use it
    :param agent:
    :return: 0
    """
    return 0

  def taxons_bd(self, traj, agent):
    """
    This function implements the TAXONS BD.
    The BD is None because it is extracted later from the image.
    The features are extracted outside of here, this way no need to have an NN for each process.
    Moreover, the forward pass is not that costly and it needs to be done not so often, only once per gen.
    :param traj:
    :param agent:
    :return:
    """
    return 0
