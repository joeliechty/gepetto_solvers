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

## Meshes are deliberately absent

The URDF references `package://drake_models/.../meshes/*.gltf` for visual and
collision geometry. **Those files are not vendored and are not needed**:
`pinocchio::urdf::buildModel` reads only the kinematic tree, so every solve here
works without them. Only `buildGeom` — mesh-based collision, and the workbench's
link rendering — would want them, and that fetches them separately.

So a missing mesh is never a solver failure. If rendering ever reports one, it is
a rendering problem.

## What this repo depends on in it

The 16 revolute joints `joint_0`..`joint_15` and the link frames `link_0`..
`link_15` plus `link_3_tip`, `link_7_tip`, `link_11_tip`, `link_15_tip`, grouped
four to a digit. `RigidChainModel` raises naming any joint or frame it cannot
find, so a replacement URDF that renames them fails loudly at construction rather
than posing a wrong hand.
