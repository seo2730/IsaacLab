# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/spawn_prims.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

'''
USD(Universal Scene Description) : 3D 모델링 소프트웨어에서 만든 3D 모델을 저장하는 파일 형식
- IssacSim에서 사용
- USDdml 3가지 컨셉
    1. Primitives (Prims): USD에 기본 구성, Scene graph의 노드로 각 노드는 meshm, light, camera 등을 나타내며 다른 prims과 그룹으로 이루어져 있을 수 있다.
    2. Attributes : prims에 대한 속성, key-value pair로 구성 (예시 : color - red)
    3. Relationships : prims간의 연결로 다른 prims에 대한 포인터로 생각하면 된다. (mesh prims은 음영 처리를 위한 material prim과 관계)
- USD stage : 모든 prims으로 이루어진 하나의 컨테이너
- Isaac Lab는 USD API를 사용하여 USD 파일을 읽고 쓰는 기능을 제공한다. 
- prim을 spawning할 때 각 prim에 대한 configuration class(속성, 관계)를 만들어야 한다.

아래 의사 코드가 전반적인 prim에 대한 configuration class(속성, 관계)를 만드는 틀이다.
# Create a configuration class instance
cfg = MyPrimCfg()
prim_path = "/path/to/prim"

# Spawn the prim into the scene using the corresponding spawner function
spawn_my_prim(prim_path, cfg, translation=[0, 0, 0], orientation=[1, 0, 0, 0], scale=[1, 1, 1])
# OR
# Use the spawner function directly from the configuration class
cfg.func(prim_path, cfg, translation=[0, 0, 0], orientation=[1, 0, 0, 0], scale=[1, 1, 1])
'''

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
    # GrouonPlaneCfg는 group plane에 대한 속성을 정의한다.
    # 해당 path는 Issac sim에서 World 아래에 있는 defaultGroundPlane을 가리킨다.
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    # DistantLightCfg는 먼 거리에서 오는 빛, 원통형 빛, 디스크 빛, 원뿔형 빛 등을 정의한다.
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # create a new xform prim for all objects to be spawned under
    # Xform는 prim을 그룹화하거나 위치, 회전, 크기를 조정하는 데 사용된다.
    # 해당 path는 Issac sim에서 World 아래에 있는 Objects를 가리키며 아래 다른 도형들을 포함한다 -> 관계성이 생김
    prim_utils.create_prim("/World/Objects", "Xform")
    # spawn a red cone
    cfg_cone = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )
    cfg_cone.func("/World/Objects/Cone1", cfg_cone, translation=(-1.0, 1.0, 1.0))
    cfg_cone.func("/World/Objects/Cone2", cfg_cone, translation=(-1.0, -1.0, 1.0))

    # spawn a green cone with colliders and rigid body
    # red cone와 같은 속성을 가지지만, rigid body, mass, collider를 가진다.
    cfg_cone_rigid = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    cfg_cone_rigid.func(
        "/World/Objects/ConeRigid", cfg_cone_rigid, translation=(-0.2, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0)
    )

    # spawn a blue cuboid with deformable body
    # 변형 가능한 물리학적 특성을 포함한 직육면체 생성
    # rigid body와 달리 변형 가능하므로 천, 고무, 젤리와 같은 부드러운 체를 시뮬레이션하는데 유용
    # deformable body는 GPU 장치에서만 실행된다.
    cfg_cuboid_deformable = sim_utils.MeshCuboidCfg(
        size=(0.2, 0.5, 0.2),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_cuboid_deformable.func("/World/Objects/CuboidDeformable", cfg_cuboid_deformable, translation=(0.15, 0.0, 2.0))

    # spawn a usd file of a table into the scene
    # 다른 경로에 있는 USD, URDF, OBJ 파일들을 불러와서 spawn할 수 있다.
    # 정확히는 USD 파일을 scene에 추가하는 것이 아닌 포인터처럼 테이블 asset를 추가하는 것이다.
    # 해당 USD 파일을 수정하면 이것을 pointer한 scene들은 바로 적용이 된다.
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func("/World/Objects/Table", cfg, translation=(0.0, 0.0, 1.05))


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
