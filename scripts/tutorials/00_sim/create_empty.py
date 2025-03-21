# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

"""

"""Launch Isaac Sim Simulator first."""


# 인자 받아오는 모듈
import argparse
# isaac sim 실행하기 전 app.AppLauncher를 import해야함
# 인자와 환경변수를 통해 시뮬레이터에 필요한 메카니즘을 제공함
from isaaclab.app import AppLauncher

# 1. create argparser (인자 요소 생성)
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# 2. append AppLauncher cli args (인자 요소를 AppLauncher에 추가)
AppLauncher.add_app_launcher_args(parser)
# 3. parse the arguments (인자 요소 받아오기)
args_cli = parser.parse_args()
# 4. launch omniverse app (받아온 인자 요소들을 입력으로 AppLauncher 실행)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Applauncher가 실행되면 issacsim의 파이썬 모듈들을 import 할 수 있다.
# SimulationContext를 통해 시뮬레이터의 재생, 일시 중지 및 스텝을 완벽하게 제어할 수 있으며 다양한 타임라인 이벤트를 처리하고 시뮬레이션을 위한 물리적 장면도 구성한다.
from isaaclab.sim import SimulationCfg, SimulationContext


def main():
    """Main function."""

    # Initialize the simulation context
    # SimulationCfg는 시간, 중력, 랜더링 등 초기 시뮬레이터 세팅에 관한 설정을 설정한다.
    sim_cfg = SimulationCfg(dt=0.01)
    # SimulationCfg에서 설정한 것들을 SimaulationContext에 넣어준다.
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Play the simulator
    # 시뮬레이터를 초기화한다. -> 이거 설정 안하면 시뮬레이터가 초기에 적절히 작동 안될 수 있다.
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        # render라는 인자(default=True이고 SimulationCfg에 있음)를 통해 step이 실행된다.
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
