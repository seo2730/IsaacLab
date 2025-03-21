# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run IsaacSim via the AppLauncher

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/launch_app.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on running IsaacSim via the AppLauncher.")
# 인자 추가
parser.add_argument("--size", type=float, default=1.0, help="Side-length of cuboid")
# SimulationApp arguments https://docs.omniverse.nvidia.com/py/isaacsim/source/isaacsim.simulation_app/docs/index.html?highlight=simulationapp#isaacsim.simulation_app.SimulationApp
parser.add_argument(
    "--width", type=int, default=1280, help="Width of the viewport and generated images. Defaults to 1280"
)
parser.add_argument(
    "--height", type=int, default=720, help="Height of the viewport and generated images. Defaults to 720"
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

'''
--help 옵션을 통해 2가지 타입의 옵션을 확인할 수 있다.
예시:
options:
-h, --help            show this help message and exit
--size SIZE           Side-length of cuboid
--width WIDTH         Width of the viewport and generated images. Defaults to 1280
--height HEIGHT       Height of the viewport and generated images. Defaults to 720

app_launcher arguments:
--headless            Force display off at all times.
--livestream {0,1,2}
                      Force enable livestreaming. Mapping corresponds to that for the "LIVESTREAM" environment variable.
--enable_cameras      Enable cameras when running without a GUI.
--verbose             Enable verbose terminal logging from the SimulationApp.
--experience EXPERIENCE
                      The experience file to load when launching the SimulationApp.

                      * If an empty string is provided, the experience file is determined based on the headless flag.
                      * If a relative path is provided, it is resolved relative to the `apps` folder in Isaac Sim and
                        Isaac Lab (in that order).
'''

"""Rest everything follows."""

import isaaclab.sim as sim_utils

# Step 1 : Design the scene
# 전 코드와 다르게 design_scene() 함수를 만들어서 시뮬레이션에 필요한 요소들을 추가한다.
# 이렇게 함으로써 시뮬레이션을 초기화할 때 예상치 못한 오류를 방지할 수 있다.
def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # spawn a cuboid
    cfg_cuboid = sim_utils.CuboidCfg(
        size=[args_cli.size] * 3,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
    )
    # Spawn cuboid, altering translation on the z-axis to scale to its size
    cfg_cuboid.func("/World/Object", cfg_cuboid, translation=(0.0, 0.0, args_cli.size / 2))

# Step 2 : Run Simulation
# scene 속 prims과 상호작용하며, 예를 들어 위치를 변경하고, 명령을 적용하는 역할
def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    # Design scene by adding assets to it
    design_scene()

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
