# Created by Giuseppe Paolo 
# Date: 30/04/2021

import os
import gym
import numpy as np
from environments import *
from core.population.archive import Archive
import parameters
import pickle as pkl
from skimage.transform import resize
import pygame as pg
import pygame.freetype  # Import the freetype module.
from analysis.gt_bd import *
import matplotlib.pyplot as plt
import argparse
import ffmpeg
from pygame import gfxdraw
from mujoco_py import GlfwContext
from analysis.utils import make_gradient
GlfwContext(offscreen=True)  # Create a window to init GLFW.

BKG_COLOR = (255, 255, 255)
class Renderer(object):
  """
  This class renders solutions from the archive
  """
  def __init__(self, surface_size=(400, 400), canvas_size=(860, 450), path=None, frame_rate=50, save_video=False):
    self.path = path
    if not os.path.exists(os.path.join(self.path, 'tmp')): os.mkdir(os.path.join(self.path, 'tmp'))

    self.drawn_policies = [2, 4] # [x, y] number of points. This makes 20 points
    self.params = parameters.Params()
    self.params.load(os.path.join(path, '_params.json'))
    self.env = registered_envs[self.params.env_name]
    self.save_video = save_video

    # Init rendering stuff
    # -----------------------------
    pg.init()
    self.canvas_size = np.array(canvas_size)
    self.surface_size = np.array(surface_size)
    self.bs_size = np.array(self.env['grid']['max_coord']) - np.array(self.env['grid']['min_coord'])
    self.bs_min = np.array(self.env['grid']['min_coord'])  # The coordinate min of the bs wrt the pixel coord. Needed to transform the bs coord to pixel
    self.canvas_center = self.canvas_size / 2.
    self.surface_center = self.surface_size / 2.
    self.TARGET_FPS = frame_rate
    # -----------------------------

    if not self.save_video:
      self.screen = pg.display.set_mode(self.canvas_size, 0, 32)
    else:
      self.screen = pg.Surface(self.canvas_size)
    self.screen.fill(BKG_COLOR)  ## Draw white background
    self.clock = pg.time.Clock()
    self.font = pg.font.SysFont('Arial', 24)

    space = 20 + self.surface_size[0]
    y_pose = self.canvas_size[1] - self.surface_size[1] - 10
    self.bkg = pg.Surface(self.surface_size)
    self.bkg.fill(pg.color.THECOLORS["white"])  ## Draw white background

    # Final Archive
    # -----------------------------
    self.arch_surf = pg.Surface(self.surface_size)
    self.arch_surf.fill(pg.color.THECOLORS["white"])  ## Draw white background
    self.arch_surf_zero = [20 + 0 * space, y_pose]
    # text_surface = self.font.render('Archive', True, (0, 0, 0))
    # self.screen.blit(text_surface, (self.arch_surf_zero[0] + self.surface_size[0] / 2 - text_surface.get_width() / 2,
    #                                 y_pose - 30))
    # -----------------------------

    # Policy
    # -----------------------------
    self.policy_surf = pg.Surface(self.surface_size)
    self.policy_surf.fill(pg.color.THECOLORS["white"])  ## Draw white background
    self.policy_surf_zero = [20 + 1 * space, y_pose]
    # text_surface = self.font.render('Policy', True, (0, 0, 0))
    # self.screen.blit(text_surface,
    #                  (self.policy_surf_zero[0] + self.surface_size[0] / 2 - text_surface.get_width() / 2,
    #                   y_pose - 30))
    # -----------------------------

    # Init simulation stuff
    # -----------------------------
    self.gym_env = gym.make(self.env['gym_name'])
    self.gym_env.seed(self.params.seed)
    self.gym_env.action_space.seed(self.params.seed)
    self.gym_env.observation_space.seed(self.params.seed)
    self.controller = self.env['controller']['controller'](**self.env['controller'])
    self.max_steps = self.env['max_steps']
    self.archive = self.load_archive('archive_final.pkl')
    # self.archive = Archive(self.params)
    self.rew_archive = self.load_archive('rew_archive_final.pkl')
    self.rew_archive.name = 'rew_archive'
    self.gt_bd = self.env['gt_bd']
    # -----------------------------

  def load_archive(self, name):
    """
    This function loads the archive
    :return:
    """
    archive = Archive(self.params)
    archive.load(os.path.join(self.path, name))
    return archive

  def draw(self, points, surface, zero, color, size=3):
    """
    This function draws the surface
    :param data:
    :param surface:
    :param zero:
    :return:
    """
    pixels = self.bs2pixel(points)
    [gfxdraw.filled_circle(surface, int(p[0]), int(p[1]), int(size), color) for p in pixels]

    # Draw Borders
    border_thickness = 2
    gfxdraw.box(surface, [0, 0, surface.get_width(), border_thickness], pg.color.THECOLORS['black'])
    gfxdraw.box(surface, [0, 0, border_thickness, surface.get_height()], pg.color.THECOLORS['black'])
    gfxdraw.box(surface, [surface.get_width()-border_thickness, 0, border_thickness, surface.get_height()], pg.color.THECOLORS['black'])
    gfxdraw.box(surface, [0, surface.get_height()-border_thickness, surface.get_width(), border_thickness], pg.color.THECOLORS['black'])
    self.screen.blit(surface, zero)

  def bs2pixel(self, points):
    """
    This function transforms from bs to pixels
    :param points:
    :return:
    """
    return np.array(self.surface_size) * (points - self.bs_min)/self.bs_size

  def choose_policies(self):
    """
    This functions randomly selects a policy in the archive
    :return:
    """
    gt_bds = self.archive['gt_bd'] + self.rew_archive['gt_bd']
    genomes = self.archive['genome'] + self.rew_archive['genome']

    policies = []
    X = np.linspace(self.env['grid']['min_coord'][0], self.env['grid']['max_coord'][0], self.drawn_policies[0]+2)[1:-1]
    Y = np.linspace(self.env['grid']['min_coord'][1], self.env['grid']['max_coord'][1], self.drawn_policies[1]+2)[1:-1]
    for x in X:
      for y in Y:
        idx = np.argmin(np.linalg.norm(np.array(gt_bds)- np.array([x, y]), axis=1))
        policies.append(genomes[idx])
    np.random.shuffle(policies)
    return policies

  def evaluate_policy(self, genome):
    """
    This function evaluates the policy
    :param policy:
    :return:
    """
    self.controller.load_genome(genome)

    img_traj = []
    done = False
    cumulated_reward = 0
    traj = []
    img_interval = 5

    self.gym_env.seed(self.params.seed)
    self.gym_env.action_space.seed(self.params.seed)
    self.gym_env.observation_space.seed(self.params.seed)
    obs = self.gym_env.reset()
    traj.append((obs, 0, done, {}, None))
    t = 0

    goals = np.rot90(self.draw_goals(), k=-1).astype(np.int)
    # bkg = np.ones_like(goals)*255
    # bkg = bkg-goals

    while not done:
      agent_input = self.env['controller']['input_formatter'](t/self.max_steps, obs)
      action = self.env['controller']['output_formatter'](self.controller(agent_input))

      obs, reward, done, info = self.gym_env.step(action)
      cumulated_reward += reward
      if t % img_interval == 0:
        view = self.gym_env.render(mode='rgb_array', bkg=goals)#, camera_name='rgb')
        view = resize(view, self.surface_size, anti_aliasing=True, preserve_range=True)  # This function already normalizes
        # view = resize(view, self.surface_size, anti_aliasing=False)  # This function already normalizes
        img_traj.append(view)

      if t >= self.max_steps-1:
        done = True
      t += 1
      traj.append((obs, reward, done, info, None))

    view = self.gym_env.render(mode='rgb_array', bkg=goals)
    view = resize(view, self.surface_size, anti_aliasing=True, preserve_range=True)  # This function already normalizes
    img_traj.append(view)
    return img_traj, self.gt_bd(traj, self.env['max_steps'])

  def render(self):
    """
    This function renders the policies
    :return:
    """
    self.counter = 0
    policies = self.choose_policies()

    for idx, genome in enumerate(policies):
      print("Evaluating policy: {}".format(idx))
      # Draw archives
      # -----------------------------
      self.arch_surf.fill(pg.color.THECOLORS["white"])
      self.draw(np.stack(self.archive['gt_bd'])*self.scale_bds, self.arch_surf, self.arch_surf_zero, (60,190,210))
      self.draw(np.stack(self.rew_archive['gt_bd'])*self.scale_bds, self.arch_surf, self.arch_surf_zero, (255,190,50))
      # -----------------------------

      # Evaluate policy and collect images
      # -----------------------------
      images, gt_bd = self.evaluate_policy(genome)
      self.draw(np.array([gt_bd])*self.scale_bds, self.arch_surf, self.arch_surf_zero, (214,40,40), size=10) # TODO dopo aver evaluato la policy ricordati di cancellare il punto!
      # -----------------------------

      # Draw images
      # -----------------------------
      for img in images:
        img = self.transform_img(img)
        pg.surfarray.blit_array(self.policy_surf, img)
        self.screen.blit(self.policy_surf, self.policy_surf_zero)

        if not self.save_video:
          pg.display.flip()  ## Need to flip cause of drawing reasons
          self.clock.tick(self.TARGET_FPS)
        else:
          imgdata = pg.surfarray.array3d(self.screen)
          plt.imsave(os.path.join(self.path, 'tmp', f'{self.counter:10}.jpg'), imgdata.swapaxes(0, 1))
          self.counter += 1

          # This is here so to have a break at the end of each policy
      for i in range(self.TARGET_FPS):
        if not self.save_video:
          pg.display.flip()  ## Need to flip cause of drawing reasons
          self.clock.tick(self.TARGET_FPS)
        else:
          imgdata = pg.surfarray.array3d(self.screen)
          plt.imsave(os.path.join(self.path, 'tmp', f'{self.counter:10}.jpg'), imgdata.swapaxes(0, 1))
          self.counter += 1
      # -----------------------------

  def draw_goals(self):
    colors = [(226, 153, 67), (42, 163,52)]
    gradients = []
    for goal, radius, color in zip(self.gym_env.goals, self.gym_env.goalRadius, colors):
      goal = np.array(goal)/3.
      goalRadius = radius /np.array([3., 3.])
      grad = make_gradient(self.surface_size[0], self.surface_size[1], goal[0], goal[1], goalRadius[0], goalRadius[1], 0)
      grad = np.broadcast_to(np.atleast_3d(grad), (*self.surface_size, 3))

      gradients.append(grad[:, :] * (255-np.array(color)))

    return np.sum(gradients, axis=0)

  def transform_img(self, img):
    """
    This function transforms the images according to the different env
    :return:
    """
    if self.params.env_name == 'HardMaze':
      return np.swapaxes(img, 0, 1)
    elif self.params.env_name == 'Curling':
      return np.swapaxes(img, 0, 1)
    elif self.params.env_name == 'NDofArm':
      img = np.flip(img, axis=1)
      return ((img*-1)+1)*255
    elif self.params.env_name == 'AntMaze':
      return np.swapaxes(img, 0, 1)
    else:
      return img

  @property
  def scale_bds(self):
    if self.params.env_name == 'HardMaze':
      return np.array([1, 1])
    elif self.params.env_name == 'Curling':
      return np.array((1, -1))
    elif self.params.env_name == 'AntMaze':
      return np.array([1, -1])
    else:
      return np.array([1, 1])


if __name__ == "__main__":
  parser = argparse.ArgumentParser('Run archive eval script')
  parser.add_argument('-p', '--path', help='Path of experiment')
  parser.add_argument('-s', '--save', help="Save video", action='store_true')
  parser.add_argument('-fr', '--frame_rate', help='Frame rate', default=50)
  parser.add_argument('-gif', help='Select save video as GIF.', action='store_true')

  args = parser.parse_args()

  path = args.path
  save_video = args.save
  frame_rate = args.frame_rate
  video_format = 'gif' if args.gif else 'mp4'

  save_video = True
  video_format = 'mp4'

  # path = '/home/giuseppe/src/cmans/experiment_data/AntMaze/AntMaze_FitNS_all_gen/2021_02_09_00:09_829130'
  # path = '/home/giuseppe/src/cmans/experiment_data/Curling/Curling_FitNS_all_gen/2021_02_05_15:46_008255'
  # path = '/home/giuseppe/src/cmans/experiment_data/HardMaze/HardMaze_FitNS_all_gen/2021_02_05_15:48_735192'
  path = '/home/giuseppe/src/cmans/experiment_data/NDofArm/NDofArm_FitNS_all_gen/2021_02_09_10:53_478150'
  # path = '/home/giuseppe/src/cmans/experiment_data/Humanoid/Humanoid_STAX_std/2021_05_04_20:55_677646'

  renderer = Renderer(path=path, save_video=save_video, frame_rate=frame_rate)
  renderer.render()

  if save_video:
    print("Generating video...")
    stream = ffmpeg.input(os.path.join(path, 'tmp/*.jpg'), pattern_type='glob', framerate=frame_rate)
    stream = ffmpeg.output(stream, os.path.join(path, 'analyzed_data/policies.{}'.format(video_format)))
    ffmpeg.run(stream, overwrite_output=True)

    print("Deleting tmp files...")
    os.system('rm -r {}'.format(os.path.join(path, 'tmp')))
    print("Done.")
