from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage_async
import numpy as np
import math
import os

# Setup simulation world
world = World(stage_units_in_meters=1.0)

# Path to your URDF file (change to your path)
urdf_path = "/home/dafodilrat/Documents/bu/RASTIC/Bittle_URDF/urdf/bittle.urdf"  # Update to your full URDF path
usd_robot_prim = "/World/Bittle"

# Import URDF using the Isaac Sim importer
if not is_prim_path_valid(usd_robot_prim):
    from isaacsim.asset.importer.urdf import URDFImporter
    importer = URDFImporter()
    importer.import_robot(urdf_path, usd_robot_prim)

# Reset world to ensure robot is initialized
world.reset()

# Get Articulation
assert isinstance(robot, Articulation)

# Get DOF info
num_dofs = robot.num_dof
joint_names = robot.dof_names
lower_limits = robot.get_dof_lower_limits()
upper_limits = robot.get_dof_upper_limits()

# Set default positions and animation parameters
default_positions = np.zeros(num_dofs)
speeds = np.zeros(num_dofs)
for i in range(num_dofs):
    low, high = lower_limits[i], upper_limits[i]
    low = max(low, -math.pi) if not math.isnan(low) else -1.0
    high = min(high, math.pi) if not math.isnan(high) else 1.0
    lower_limits[i] = low
    upper_limits[i] = high
    speeds[i] = max(0.2, min(2.0, abs(high - low)))

# Animation states
ANIM_SEEK_LOWER = 1
ANIM_SEEK_UPPER = 2
ANIM_SEEK_DEFAULT = 3
ANIM_FINISHED = 4

anim_state = ANIM_SEEK_LOWER
current_dof = 0
positions = default_positions.copy()
print(f"Animating DOF {current_dof} ('{joint_names[current_dof]}')")

# Animate joints
for _ in range(3000):
    world.step(render=True)
    dt = world.get_physics_dt()
    speed = speeds[current_dof]

    if anim_state == ANIM_SEEK_LOWER:
        positions[current_dof] -= speed * dt
        if positions[current_dof] <= lower_limits[current_dof]:
            positions[current_dof] = lower_limits[current_dof]
            anim_state = ANIM_SEEK_UPPER

    elif anim_state == ANIM_SEEK_UPPER:
        positions[current_dof] += speed * dt
        if positions[current_dof] >= upper_limits[current_dof]:
            positions[current_dof] = upper_limits[current_dof]
            anim_state = ANIM_SEEK_DEFAULT

    elif anim_state == ANIM_SEEK_DEFAULT:
        positions[current_dof] -= speed * dt
        if positions[current_dof] <= default_positions[current_dof]:
            positions[current_dof] = default_positions[current_dof]
            anim_state = ANIM_FINISHED

    elif anim_state == ANIM_FINISHED:
        positions[current_dof] = default_positions[current_dof]
        current_dof = (current_dof + 1) % num_dofs
        anim_state = ANIM_SEEK_LOWER
        print(f"Animating DOF {current_dof} ('{joint_names[current_dof]}')")

    robot.set_joint_positions(positions)

simulation_app.close()
