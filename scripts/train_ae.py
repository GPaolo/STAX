# Created by Giuseppe Paolo 
# Date: 08/03/2021

# This script trains the autoencoder on the given images

import os
import numpy as np
import pickle as pkl
from parameters import params
import torch
from skimage.color import rgb2gray

ae_type = 'VAE'

if ae_type == 'AE':
  from core.auto_encoders import ConvAE as AE
elif ae_type == 'VAE':
  from core.auto_encoders import ConvBVAE as AE

path = '/home/giuseppe/src/cmans/experiment_data/Curling/'
with open(os.path.join(path, 'images.pkl'), 'rb') as f:
  data = pkl.load(f)
  # data = np.expand_dims(rgb2gray(data), -1)

split = int(len(data) * 0.8)
validation_data = data[split:]
training_data = data[:split]
ae = AE(encoding_shape=params.bd_size)

training_data_idx = np.array(range(len(training_data)))
validation_data_idx = np.array(range(len(validation_data)))

prev_valid_error = np.inf
valid_increase = 0 # Counter for how many times the validation error increased
training_epoch = 0

while training_epoch <= params.max_training_epochs:
  training_epoch += 1

  # Training cycle
  # -------------------------------------------------------------
  batch_idx = [0, params.batch_size]
  np.random.shuffle(training_data_idx) # These are reshuffled at every epoch
  training_total_loss = 0
  training_rec_loss = 0
  training_kl_div = 0
  training_step = 0

  # Yes it's idx_batch[0]< . I tested and it gets all elements, so no need to repeat stuff out of the while
  while batch_idx[0] < len(training_data):
    batch = training_data[training_data_idx[batch_idx[0]:min(batch_idx[1], len(training_data))]]
    batch = torch.Tensor(batch).permute((0, 3, 1, 2)).contiguous() # NHWC -> NCHW

    loss = ae.training_step(batch.to(ae.device))
    batch_idx[0] = batch_idx[1]
    batch_idx[1] += params.batch_size

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
  if training_epoch % params.validation_interval == 0 and training_epoch > 1:
    batch_idx = [0, params.batch_size]
    errors = [] # For the valid we calculate as error the mean over all the rec errors. We store them in a list cause the data is batched
    # No need to shuffle the idx cause no learning happens
    # We still batch it so to help with memory constraints
    while batch_idx[0] < len(validation_data):
      batch = validation_data[validation_data_idx[batch_idx[0]:min(batch_idx[1], len(validation_data))]]

      with torch.no_grad():
        batch = torch.Tensor(batch).permute((0, 3, 1, 2))  # NHWC -> NCHW
        ae_output = ae.forward(batch.to(ae.device))

      batch_idx[0] = batch_idx[1]
      batch_idx[1] += params.batch_size
      errors.append(ae_output['error'].cpu().numpy())

    validation_error = np.mean(np.concatenate(errors))
    print('Validation error: {}'.format(validation_error))

    # Check validation error increase
    if validation_error > prev_valid_error:
      if valid_increase < 3:
        valid_increase += 1
        print("Valid error consecutive increases: {}".format(valid_increase))
      else:
        print("\nValidation error increased. Stopping.")
        break
    else:
      valid_increase = 0
      prev_valid_error = validation_error
  # -------------------------------------------------------------

# print("Final training error {}".format(training_total_loss / training_step))
# print("Final validation error {}".format(validation_error))
print()
print('Total training epochs: {}'.format(training_epoch))
print()
ae.save(os.path.join(path, 'ae'), ae_type)


