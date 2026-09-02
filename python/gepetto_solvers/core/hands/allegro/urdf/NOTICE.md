# Allegro hand URDF — provenance

`allegro_hand_right.urdf` is vendored from the Drake model set:

    https://github.com/RobotLocomotion/models
    allegro_hand_description/urdf/allegro_hand_description_right.urdf

Drake and its model repository are **BSD-3-Clause** (Toyota Research Institute et
al.). The file's own header comment records its lineage: it was converted with
`xacro` from `allegro_hand_description_right.xacro` in an unofficial ROS package
for the Allegro hand, and Drake then **recomputed every inertia matrix**, because
the upstream values are not symmetric positive definite and joint damping was
overstated by up to ~300x.

## Why this copy rather than the manufacturer's

Non-physical inertias diverge a dynamics solve. Nothing here runs dynamics today
— the kinematics likelihood needs only the joint tree — but taking the corrected
file costs nothing now and removes a trap from whoever adds dynamics later.

## The meshes are vendored, and are VISUAL ONLY

`meshes/` holds the 11 `.gltf` files the URDF's `<visual>` blocks name, with
their `.bin` buffers — 22 files, about 850 KB. Small enough to commit, which is
why there is no fetch script: the workbench draws the hand on a fresh clone with
no network.

**Nothing in a solve reads them.** Collision in this repository is the sphere set
carried on each digit's sites (`DigitState.collision_sites`), and the factor
graph never sees a mesh. `pinocchio::urdf::buildModel` reads only the kinematic
tree. Deleting `meshes/` costs the picture its skin and changes no number —
`tests/core/test_allegro_hand.py` asserts exactly that, by solving with and
without them and comparing.

Note the URDF's own `<collision>` blocks are boxes and spheres, not meshes, so
even a mesh-based collision pipeline would not need these files.

## What this repo depends on in it

The 16 revolute joints `joint_0`..`joint_15` and the link frames `link_0`..
`link_15` plus `link_3_tip`, `link_7_tip`, `link_11_tip`, `link_15_tip`, grouped
four to a digit. `RigidChainModel` raises naming any joint or frame it cannot
find, so a replacement URDF that renames them fails loudly at construction rather
than posing a wrong hand.
