
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.h1.h1_teleop_config import H1TeleopCfg, H1TeleopCfgPPO

from .base.legged_robot import LeggedRobot
from .g1_23.g1_23_walk_2_hop_2_walk import G1_23_Walk_2_Hop_2_Walk
from .g1_23.g1_23_walk_2_hop_2_walk_config import G1_23_Walk_2_Hop_2_Walk_Cfg, G1_23_Walk_2_Hop_2_Walk_CfgPPO
from legged_gym.utils.task_registry import task_registry

task_registry.register( "h1:teleop", LeggedRobot, H1TeleopCfg(), H1TeleopCfgPPO())
task_registry.register( "g1_23:walk_2_hop_2_walk", G1_23_Walk_2_Hop_2_Walk, G1_23_Walk_2_Hop_2_Walk_Cfg(), G1_23_Walk_2_Hop_2_Walk_CfgPPO())