# Allegro hand URDF — provenance

`allegro_hand_v5_right_B.urdf` is vendored **verbatim** from Wonik Robotics'
official ROS 2 package for the Allegro Hand V5:

    https://github.com/Wonikrobotics-git/allegro_hand_ros2_v5
    src/allegro_hand_controllers/urdf/allegro_hand_description_right_B.urdf
    @ 80bd4a88d2c59b8ad0242ec3730302bde61c84fb

Byte-identical to upstream so it can be diffed against a fresh clone. Refresh it
with `scripts/vendor_allegro_v5_meshes.py`, which copies the URDF and converts
the meshes it names.

## Which hand this is

**V5, right, type B.** Upstream ships four descriptions and they are four
different robots, not four skins:

* **V5 vs V4.** V5 is a different mechanism. The link naming carries a hand
  index (`joint_0_0`, `link_3_0_tip`, where V4 wrote `joint_0`, `link_3_tip`),
  the root is a bare `world` link joined to `palm_link` at the origin (V4 put
  the palm 95 mm up a fixed joint), and the thumb's first joint sweeps
  0..1.78 rad where V4 gave it ±0.47 and could not truly oppose.
* **type A vs type B.** Different finger links and different joint limits: A's
  proximal-to-tip chain is 21/51/38.4 mm with a 26.7 mm tip, B's is
  17/43.1/38 mm with a 40 mm tip. The driver reads the type off the hand's own
  firmware (`AllegroHandDrv.cpp`), so a hand knows which it is; posing a type A
  with this file would be a wrong hand, not a rounding error.
* **right vs left.** Mirrored, and the fingers are renumbered with them.

To vendor a different one:

    python scripts/vendor_allegro_v5_meshes.py --variant right_A

and repoint `spec.URDF_PATH`. Nothing else in the package is variant-specific
— but `AllegroHand`'s default posture and wrist pose are CALIBRATED against
this variant's link lengths, and its `tip_radii` against this variant's
fingertip, so both need remeasuring.

## Licence

BSD 2-Clause, Copyright (c) 2024 WonikRobotics_official. Full text in the
upstream repository's `LICENSE`:

> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice, this
>    list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice,
>    this list of conditions and the following disclaimer in the documentation
>    and/or other materials provided with the distribution.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. […]

## The meshes are vendored, DECIMATED, and VISUAL ONLY

`meshes/` holds the 11 meshes the URDF's `<visual>` blocks name — about 1.5 MB.
Small enough to commit, which is why there is no fetch script: the workbench
draws the hand on a fresh clone with no network.

They are **not** upstream's files. Upstream ships ~21 MB of STL for this
variant; `scripts/vendor_allegro_v5_meshes.py` decimates each to about 8000
triangles and writes it as binary glTF under the same stem, which is why the
URDF says `palm.stl` and the directory holds `palm.glb` (`meshes.py` swaps the
suffix). Deviation from the originals is under 2 mm at the 99th percentile on a
128 mm palm.

Two things the conversion deliberately does NOT do, because the URDF must stay
the one authority on placement:

* **It does not scale.** The meshes stay in MILLIMETRES, as upstream authored
  them, and the URDF's `scale="0.001 0.001 0.001"` is applied at draw time.
* **It does not re-centre or re-orient.** Every V5 mesh is authored in one
  shared assembly frame, and each link's `<visual><origin>` brings its own part
  back to its joint. Those origins are large — the palm's is
  `xyz="0.02 0 -0.1" rpy="0 3.14 1.57"` — so a renderer that assumes an identity
  visual origin draws twenty-one parts scattered over a 200 mm cube instead of a
  hand.

**Nothing in a solve reads them.** Collision in this repository is the sphere set
carried on each digit's sites (`DigitState.collision_sites`), and the factor
graph never sees a mesh. `pinocchio::urdf::buildModel` reads only the kinematic
tree. Deleting `meshes/` costs the picture its skin and changes no number —
`tests/core/test_allegro_hand.py` asserts exactly that, by solving with and
without them and comparing.

Note this URDF gives the SAME mesh as both `<visual>` and `<collision>` for
every link. We draw it and ignore the collision claim; the solve's geometry is
the sphere set, not upstream's.

## What this repo depends on in it

The 16 revolute joints `joint_0_0`..`joint_15_0` and the link frames
`link_0_0`..`link_15_0` plus `link_3_0_tip`, `link_7_0_tip`, `link_11_0_tip`,
`link_15_0_tip`, grouped four to a digit. `RigidChainModel` raises naming any
joint or frame it cannot find, so a replacement URDF that renames them fails
loudly at construction rather than posing a wrong hand.
