# Created by Giuseppe Paolo 
# Date: 30/04/2021

"""
This file is to create videos specifically for the ant environment, with the multiple views
"""
import gc
import os
import gym
import numpy as np
from environments import *
from core.population.archive import Archive
import parameters
import pickle as pkl
from skimage.transform import resize
from skimage.draw import line_aa
import pygame as pg
import pygame.freetype  # Import the freetype module.
from analysis.gt_bd import *
import matplotlib.pyplot as plt
import argparse
import ffmpeg
from pygame import gfxdraw
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)  # Create a window to init GLFW.

BKG_COLOR = (255, 255, 255)
class Renderer(object):
  """
  This class renders solutions from the archive
  """
  def __init__(self, surface_size=(950, 950), traj_surf_size=(300, 300),
               canvas_size=(1000, 1000), path=None, frame_rate=50, save_video=False, show_arch=False, skipped_frames=10, drawn_policies=3):
    self.path = path
    if not os.path.exists(os.path.join(self.path, 'tmp')): os.mkdir(os.path.join(self.path, 'tmp'))

    self.drawn_policies = drawn_policies
    self.params = parameters.Params()
    self.params.load(os.path.join(path, '_params.json'))
    self.env = registered_envs[self.params.env_name]
    self.save_video = save_video
    self.show_arch = show_arch
    self.show_traj = True
    self.traj_surf_size = traj_surf_size
    self.img_interval = skipped_frames
    self.bkgnd = self.load_bkgd()


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

    space = 50 + self.surface_size[0]
    y_pose = self.canvas_size[1] - self.surface_size[1] - 10
    self.bkg = pg.Surface(self.surface_size)
    self.bkg.fill(pg.color.THECOLORS["white"])  ## Draw white background

    if self.show_traj:
      # Policy
      # -----------------------------
      self.policy_surf = pg.Surface(self.surface_size)
      self.policy_surf.fill(pg.color.THECOLORS["white"])  ## Draw white background
      self.policy_surf_zero = [20 + 0 * space, y_pose]
      # text_surface = self.font.render('Policy', True, (0, 0, 0))
      # self.screen.blit(text_surface,
      #                  (self.policy_surf_zero[0] + self.surface_size[0] / 2 - text_surface.get_width() / 2,
      #                   y_pose - 30))
      # -----------------------------

    # Front View
    # -----------------------------
    self.view_surf = pg.Surface(self.traj_surf_size)
    self.view_surf.fill(pg.color.THECOLORS["white"])  ## Draw white background
    self.view_surf_zero = [20 + 0 * space + self.surface_size[0] - self.traj_surf_size[0]-2, y_pose+2]
    # self.screen.blit(text_surface, (self.view_surf_zero[0] + self.surface_size[0] / 2 - text_surface.get_width() / 2,
    #                                 y_pose - 30))
    # -----------------------------




    if self.show_arch:
      # Final Archive
      # -----------------------------
      self.arch_surf = pg.Surface(self.surface_size)
      self.arch_surf.fill(pg.color.THECOLORS["white"])  ## Draw white background
      self.arch_surf_zero = [20 + 1 * space, y_pose]
      # text_surface = self.font.render('Archive', True, (0, 0, 0))
      # self.screen.blit(text_surface, (self.arch_surf_zero[0] + self.surface_size[0] / 2 - text_surface.get_width() / 2,
      #                                 y_pose - 30))
      # -----------------------------

    # Init simulation stuff
    # -----------------------------
    self.gym_env = gym.make(self.env['gym_name'])
    self.gym_env.seed(self.params.seed)
    self.gym_env.action_space.seed(self.params.seed)
    self.gym_env.observation_space.seed(self.params.seed)
    self.controller = self.env['controller']['controller'](**self.env['controller'])
    self.max_steps = self.env['max_steps']
    # self.archive = self.load_archive('archive_final.pkl')
    self.archive = Archive(self.params)
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

  def load_bkgd(self):
    import imageio
    bkg = imageio.imread(os.path.join(os.path.dirname(os.path.dirname(self.path)), 'ant_maze.png'))[:, :, :3]
    return np.swapaxes(bkg, 0, 1)

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
    # gt_bds = self.archive['gt_bd'] + self.rew_archive['gt_bd']
    genomes = self.archive['genome'] + self.rew_archive['genome']
    idxs = []
    next_area = 0
    while len(idxs) < self.drawn_policies:
      id = np.random.choice(np.array(range(len(genomes))), 1, True)
      if self.rew_archive['rew_area'][id[0]] == next_area:
        idxs.append(id[0])
        if next_area == 0: next_area=1
        else: next_area=0


    # policies = []
    # X = np.linspace(self.env['grid']['min_coord'][0], self.env['grid']['max_coord'][0], self.drawn_policies[0]+2)[1:-1]
    # Y = np.linspace(self.env['grid']['min_coord'][1], self.env['grid']['max_coord'][1], self.drawn_policies[1]+2)[1:-1]
    # for x in X:
    #   for y in Y:
    #     idx = np.argmin(np.linalg.norm(np.array(gt_bds)- np.array([x, y]), axis=1))
    #     policies.append(genomes[idx])
    # np.random.shuffle(policies)
    return np.array([genomes[i] for i in idxs])

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
    img_interval = self.img_interval
    positions = []

    self.gym_env.seed(self.params.seed)
    self.gym_env.action_space.seed(self.params.seed)
    self.gym_env.observation_space.seed(self.params.seed)
    obs = self.gym_env.reset()
    positions.append(obs[:2])
    traj.append((obs, 0, done, {}, None))
    t = 0

    while not done:
      agent_input = self.env['controller']['input_formatter'](t/self.max_steps, obs)
      action = self.env['controller']['output_formatter'](self.controller(agent_input))

      obs, reward, done, info = self.gym_env.step(action)
      cumulated_reward += reward
      if t % img_interval == 0:
        view_top = self.gym_env.render(mode='rgb_array', camera_name='static')
        view_top = resize(view_top, self.surface_size, anti_aliasing=True, preserve_range=True)

        view_robot = self.gym_env.render(mode='rgb_array', camera_name='track')
        view_robot = resize(view_robot, self.traj_surf_size, anti_aliasing=True, preserve_range=True)
        img_traj.append([view_robot, view_top])
        positions.append(obs[:2])

      if t >= self.max_steps-1:
        done = True
      t += 1
      traj.append((obs, reward, done, info, None))

    view_top = self.gym_env.render(mode='rgb_array', camera_name='static')
    view_top = resize(view_top, self.surface_size, anti_aliasing=True, preserve_range=True)

    view_robot = self.gym_env.render(mode='rgb_array', camera_name='track')
    view_robot = resize(view_robot, self.surface_size, anti_aliasing=True, preserve_range=True)
    img_traj.append([view_robot, view_top])
    return img_traj, self.gt_bd(traj, self.env['max_steps']), np.array(positions)

  def draw_traj(self, positions, image):
    """
    This function draws te trajectory
    :param positions: positions of the traj to be drawn up until image
    :param image: image on which to draw
    :return:
    """
    for idx in range(1, len(positions)):
      start = positions[idx-1]
      end = positions[idx]
      rows, cols, w = line_aa(start[0], start[1], end[0], end[1])
      w = w.reshape([-1, 1])  # reshape anti-alias weights
      lineColorRgb = [255, 50, 50]  # color of line, orange here

      image[rows, cols, 0:3] = (np.multiply((1 - w) * np.ones([1, 3]), image[rows, cols, 0:3]) + w * np.array([lineColorRgb]))
    return image

  def render(self):
    """
    This function renders the policies
    :return:
    """
    self.counter = 0
    policies = self.choose_policies()
    # 8 was calculated experimentally. This way we can change the surface size and kee the traj centered
    self.recenter_traj = np.array([8, -8]) * self.traj_surf_size/np.array((400, 400))
    if self.show_arch:
      self.bkgnd = resize(self.bkgnd, self.surface_size, anti_aliasing=True, preserve_range=True)
      pg.surfarray.blit_array(self.arch_surf, self.bkgnd)
      self.screen.blit(self.arch_surf, self.arch_surf_zero)

    for idx, genome in enumerate(policies):
      print("Evaluating policy: {}".format(idx))

      # Evaluate policy and collect images
      # -----------------------------
      images, gt_bd, positions = self.evaluate_policy(genome)

      # Draw images
      # -----------------------------
      positions = ((positions - self.bs_min) / self.bs_size * self.surface_size).astype(np.int)
      positions = positions * np.array([[1, -1]]) #+ self.recenter_traj.astype(np.int)
      for idx, img in enumerate(images):
        img_top = self.transform_img(img[1])
        img_top = self.draw_traj(positions[:idx + 1], img_top)
        pg.surfarray.blit_array(self.policy_surf, img_top)
        self.screen.blit(self.policy_surf, self.policy_surf_zero)

        img_robot = self.transform_img(img[0])
        img_robot = resize(img_robot, np.array(self.traj_surf_size)-10)
        border = np.ones((*self.traj_surf_size, 3))*255
        border[5:self.traj_surf_size[0]-5, 5:self.traj_surf_size[1]-5] = img_robot
        pg.surfarray.blit_array(self.view_surf, border)
        self.screen.blit(self.view_surf, self.view_surf_zero)

        if not self.save_video:
          pg.display.flip()  ## Need to flip cause of drawing reasons
          self.clock.tick(self.TARGET_FPS)
        else:
          imgdata = pg.surfarray.array3d(self.screen)
          plt.imsave(os.path.join(self.path, 'tmp', f'{self.counter:10}.jpg'), imgdata.swapaxes(0, 1))
          self.counter += 1
      if self.show_arch:
        self.draw(np.array([gt_bd]) * self.scale_bds, self.arch_surf, self.arch_surf_zero, (214, 40, 40), size=10)

      # This is here so to have a break at the end of each policy
      for i in range(self.TARGET_FPS):
        if not self.save_video:
          pg.display.flip()  ## Need to flip cause of drawing reasons
          self.clock.tick(self.TARGET_FPS)
        else:
          imgdata = pg.surfarray.array3d(self.screen)
          plt.imsave(os.path.join(self.path, 'tmp', f'{self.counter:10}.jpg'), imgdata.swapaxes(0, 1))
          self.counter += 1

      del images
      gc.collect()
      # -----------------------------

  def transform_img(self, img):
    """
    This function transforms the images according to the different env
    :return:
    """
    if self.params.env_name == 'AntMaze':
      return np.swapaxes(img, 0, 1)
    else:
      return img

  @property
  def scale_bds(self):
    if self.params.env_name == 'AntMaze':
      return np.array([1, -1])
    else:
      return np.array([1, 1])


if __name__ == "__main__":
  parser = argparse.ArgumentParser('Run archive eval script')
  parser.add_argument('-p', '--path', help='Path of experiment')
  parser.add_argument('-s', '--save', help="Save video", action='store_true')
  parser.add_argument('-fr', '--frame_rate', help='Frame rate', default=50)
  parser.add_argument('-gif', help='Select save video as GIF.', action='store_true')
  parser.add_argument('-arch', help='Select if to show archive', action='store_true')

  args = parser.parse_args()

  path = args.path
  save_video = args.save
  frame_rate = args.frame_rate
  video_format = 'gif' if args.gif else 'mp4'
  show_arch = False

  save_video = True
  video_format = 'mp4'


  path = '/home/giuseppe/src/cmans/experiment_data/AntMaze/AntMaze_FitNS_all_gen/2021_02_09_00:09_829130'


  # def __init__(self, surface_size=(400, 400), canvas_size=(860, 450), path=None, frame_rate=50, save_video=False, show_arch=False, skipped_frames=10):

  renderer = Renderer(path=path, save_video=save_video, frame_rate=frame_rate, show_arch=show_arch, traj_surf_size=(300, 300), surface_size=(800, 800),
                      skipped_frames=10, canvas_size=(1700, 850), drawn_policies=6)
  renderer.render()

  if save_video:
    print("Generating video...")
    stream = ffmpeg.input(os.path.join(path, 'tmp/*.jpg'), pattern_type='glob', framerate=frame_rate)
    stream = ffmpeg.output(stream, os.path.join(path, 'analyzed_data/policies.{}'.format(video_format)))
    ffmpeg.run(stream, overwrite_output=True)

    print("Deleting tmp files...")
    os.system('rm -r {}'.format(os.path.join(path, 'tmp')))
    print("Done.")
