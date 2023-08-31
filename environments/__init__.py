# Created by Giuseppe Paolo 
# Date: 27/07/2020
import sys
import os
from parameters import ROOT_DIR

assets = os.path.join(ROOT_DIR, 'environments/assets/')
sys.path.append(assets)

from environments import ant_maze
from environments import curling
from environments import point2D
from environments import redundant_arm

registered_envs = {
  'Curling': curling.environment,
  'Point2D': point2D.environment,
  'AntMaze': ant_maze.environment,
  'RedArm': redundant_arm.environment
}