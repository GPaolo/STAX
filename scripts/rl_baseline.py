# Created by Giuseppe Paolo 
# Date: 15/09/2020

# Created by Giuseppe Paolo
# Date: 29/07/2020

from parameters import params
import os
from environments import registered_envs
import gym
import argparse
from stable_baselines3 import SAC, PPO, A2C, HerReplayBuffer, DQN
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common import logger
from stable_baselines3.common.logger import configure
import json

if __name__ == "__main__":
  parser = argparse.ArgumentParser('Run RL baseline script')
  parser.add_argument('-env', '--environment', help='Environment to use', choices=list(registered_envs.keys()))
  parser.add_argument('-exp', '--experiment', help='RL Baseline to use.', choices=['PPO', 'SAC', 'SAC_HER', 'A2C'])
  parser.add_argument('-sp', '--save_path', help='Path where to save the experiment')
  parser.add_argument('-mp', '--multiprocesses', help='How many parallel workers need to use', type=int) # TODO find a way to use multiprocessing maybe running mulple runs in parallel
  parser.add_argument('-r', '--runs', help='Number of runs to perform', type=int, default=1)
  parser.add_argument('-epi', '--episodes', help='Number of episodes used for training', type=int, default=5)

  args = parser.parse_args()#['-env', 'Curling', '-exp', 'SAC_HER'])

  for run in range(args.runs):

    if args.environment is not None: params.env_name = args.environment
    if args.experiment is not None: params.exp_type = args.experiment
    if args.save_path is not None: params.save_path = os.path.join(args.save_path, params.save_dir)
    if args.multiprocesses is not None: params.multiprocesses = args.multiprocesses

    print("SAVE PATH: {}".format(params.save_path))
    params.save()

    env = registered_envs[args.environment]
    gym_env = gym.make(env['gym_name'], her=True)

    # gym_env = BitFlippingEnv(n_bits=15, continuous=SAC, max_steps=15)
    # TODO tune the hyperparameters properly
    if args.experiment == 'PPO':
      model = PPO("MlpPolicy", gym_env, verbose=1)
    elif args.experiment == 'SAC':
      model = SAC("MlpPolicy", gym_env, verbose=1)
    elif args.experiment == 'A2C':
      model = A2C("MlpPolicy", gym_env, verbose=1)
    elif args.experiment == 'SAC_HER':
      model = SAC("MultiInputPolicy",gym_env,
                  replay_buffer_class=HerReplayBuffer,
                  replay_buffer_kwargs=dict(n_sampled_goal=4,
                                            goal_selection_strategy='episode',
                                            online_sampling=True,
                                            max_episode_length=15,
                                           ),
                  verbose=1,
                 )

    else:
      raise ValueError('Wrong experiment chosen: {}'.format(args.experiment))

    new_logger = configure(params.save_path, ["stdout", "json"])
    model.set_logger(new_logger)
    model.learn(n_eval_episodes=args.episodes, log_interval=10, total_timesteps=args.episodes*registered_envs['Curling']['max_steps'])
    model.save(os.path.join(params.save_path, "{}_{}".format(args.environment, args.experiment)))

    logs = {'episode':[], 'reward':[]}
    with open(os.path.join(params.save_path, 'progress.json'), 'r') as f:
      for line in f:
        logs['episode'].append(json.loads(line)['time/episodes'])
        logs['reward'].append(json.loads(line)['rollout/ep_rew_mean'])

    with open(os.path.join(params.save_path, 'logs.json'), 'w') as f:
      json.dump(logs, f)