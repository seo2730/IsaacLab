# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a simple environment with a cartpole. It combines the concepts of
scene, action, observation and event managers to create an environment.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/create_cartpole_base_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch

import isaaclab.envs.mdp as mdp
# ManagerBasedRLEnv는 강화학습에 유용한 모듈이고 ManagerBasedEnv는 기존 로봇 제어가 쓰인다.
# ManagerBasedRLEnv는 reward, termination, curriculum, command generation 등을 포함함
# 다음 아래 4개가 ManagerBasedRLEnv의 하위 모듈이다.
# InteractiveScene : 시뮬레이션 사용될 scene
# ActionManager : action 매니저
# ObservationManager : observation 매니저
# EventManager : start, reset, periodic intervals 작업을 예약 관리(스케줄링)하는 매니저
# 이러한 구성 요소를 구성함으로써 사용자는 최소한의 노력으로 동일한 환경의 다양한 변형을 만들 수 있다.
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleSceneCfg


@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    # 기존 Articulation.set_joint_effort_target()과 달리 Action 매니저를 통해 action를 설정한다.
    # 카트 좌우로 움직이게 하여 pole를 균형을 맞추도록 하기 위해 slider_to_cart만 액션으로 설정한다.
    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=5.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    # ObservationGroupCfg를 상속받음
    # 이 클래스는 다른 observation term들을 정의하고 관리하는 모듈
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # ObservationTermCfg는 관측할 값들의 함수 및 클래스 정의한다.
        # 관측할 값들을 노이즈 모델, clipping, scaling 등을 통해 변환할 수 있다.
        # 여기서는 디폴트 값만 사용
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
# 시뮬레이션의 상태를 변경과 관련된 이벤트 클래스이다.
# 예시 : resetting (or randomizing) the scene, 
#       randomizing physical properties (such as mass, friction, etc.)
#       varying visual properties (such as colors, textures, etc.).
# EventTermCfg를 통해 정의할 수 있다.
# mode : startup, reset, periodic 3개를 정의하고 싶으면 ManagerBasedEnv 클래스를 사용해야 한다.
# startup : 시뮬레이션 시작시 실행
# reset : 환경을 리셋할 때 실행
# periodic : 특정 주기 때 이벤트를 실행
class EventCfg:
    """Configuration for events."""

    # on startup
    # mass를 랜덤하게 정의
    add_pole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
            "mass_distribution_params": (0.1, 0.5),
            "operation": "add",
        },
    )

    # on reset
    # joint state를 랜덤하게 초기화
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )


@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene = CartpoleSceneCfg(num_envs=1024, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz


def main():
    """Main function."""
    # parse the arguments
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, _ = env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
