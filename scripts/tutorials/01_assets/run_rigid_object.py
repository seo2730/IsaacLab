# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a rigid object and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_rigid_object.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a rigid object.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext

# Step 1 : Design the scene
# 전 코드와 다르게 design_scene() 함수를 만들어서 시뮬레이션에 필요한 요소들을 추가한다.
# 이렇게 함으로써 시뮬레이션을 초기화할 때 예상치 못한 오류를 방지할 수 있다.
def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a robot in it
    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # Rigid Object
    cone_cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/Cone", # /World/Origin{1}/Cone, /World/Origin{2}/Cone, /World/Origin{3}/Cone, /World/Origin{4}/Cone
        spawn=sim_utils.ConeCfg(
            radius=0.1,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    # Wrap the spawning configuration into RigidObject class
    cone_object = RigidObject(cfg=cone_cfg)

    # return the scene information
    scene_entities = {"cone": cone_object}
    return scene_entities, origins

# Step 2 : Run Simulation
# scene 속 prims과 상호작용하며, 예를 들어 위치를 변경하고, 명령을 적용하는 역할
# 3가지 단계(resetting the simulation state at fixed intervals, stepping the simulation, and updating the internal buffers of the rigid object)로 이루어져 있음.
def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    # 총 4개의 cone이 cone_object에 저장됨
    cone_object = entities["cone"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    # Xform frame이 아닌 simulation world frame에서 정의해야됨
    while simulation_app.is_running():
        # 1. reset
        # pose와 velocity를 세팅하는 과정
        if count % 250 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset root state
            # [pos, quat, lin_vel, ang_vel]
            root_state = cone_object.data.default_root_state.clone()
            # sample a random position on a cylinder around the origins
            root_state[:, :3] += origins # 위치
            # 추가적으로 random한 위치 생성
            root_state[:, :3] += math_utils.sample_cylinder(
                radius=0.1, h_range=(0.25, 0.5), size=cone_object.num_instances, device=cone_object.device
            )
            # write root state to simulation
            # pose와 velocity를 sim에 쓰는 과정
            cone_object.write_root_pose_to_sim(root_state[:, :7]) # 위치와 자세
            cone_object.write_root_velocity_to_sim(root_state[:, 7:])
            # reset buffers
            cone_object.reset()
            print("----------------------------------------")
            print("[INFO]: Resetting object state...")
        # 2. apply sim data
        cone_object.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # 3. update buffers
        cone_object.update(sim_dt)
        # print the root position
        if count % 50 == 0:
            print(f"Root position (in world): {cone_object.data.root_state_w[:, :3]}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[1.5, 0.0, 1.0], target=[0.0, 0.0, 0.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
