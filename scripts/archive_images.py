# Created by Giuseppe Paolo 
# Date: 29/09/2020

# This script is used to generate images from an archive

import os
import gym
import numpy as np
from environments import *
import gym_collectball
from core.population.archive import Archive
from parameters import params
import pickle as pkl
from skimage.transform import resize


env_name = 'Curling'
PATH = '/home/giuseppe/src/cmans/experiment_data/Curling/Curling_NS_all_gen/2021_02_05_15:39_822502'

params.load(os.path.join(PATH, '_params.json'))
arch = Archive(parameters=params)
arch.load(os.path.join(PATH, 'archive_final.pkl'))

env = registered_envs[env_name]
controller = env['controller']['controller']
controller = controller(input_size=env['controller']['input_size'],
                        output_size=env['controller']['output_size'],
                        hidden_layers=env['controller']['hidden_layers'])
gym_env = gym.make(env['gym_name'])

images = []

for genome in arch['genome']:
  controller.load_genome(genome=genome)
  obs = gym_env.reset()
  i = 0
  done = False
  while i < env['max_steps'] and not done:
    action = controller(env['controller']['input_formatter'](i, obs))
    action = env['controller']['output_formatter'](action)
    obs, _, done, _ = gym_env.step(action)
    i += 1

  img = gym_env.render('rgb_array')
  img = resize(img, (64, 64), anti_aliasing=False)
  images.append(img)

with open('/home/giuseppe/src/cmans/experiment_data/Curling/images.pkl', 'wb') as f:
  pkl.dump(np.array(images), f)