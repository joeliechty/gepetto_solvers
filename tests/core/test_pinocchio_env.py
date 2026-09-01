"""Guards on the Pinocchio dependency itself, not on any code that uses it.

Two things here are environment facts that the C++ rigid-body kinematics is built
on, and both fail silently or confusingly if they ever stop holding:

1. **Pinocchio is installed and can parse a URDF.** It is a conda C++ dependency
   found through the interpreter prefix (like GTSAM and OpenVDB), so a fresh
   machine that skipped the setup script has a Python layer that imports fine and
   a C++ build that does not.

2. **Pinocchio orders spatial quantities [linear; angular].** GTSAM's ``Pose3``
   tangent is the other way round, ``[angular; linear]``, so the FK factor swaps
   the top and bottom three rows of every Jacobian it gets from Pinocchio. That
   swap is the single easiest thing in the whole integration to get wrong, and
   getting it wrong does not raise -- it just converges badly. Asserting the
   convention here pins it independently of the C++ that relies on it.

The Eigen-version constraint that also governs this dependency (GTSAM bakes in
``GTSAM_EIGEN_VERSION_*`` and static-asserts it, so Pinocchio must be the build
compiled against the same Eigen major.minor) cannot be checked from Python -- it
is a compile-time error, so the C++ build is its own guard. See the note in
``CMakeLists.txt`` beside ``EIGEN3_INCLUDE_DIR`` and in the conda setup scripts.
"""

from __future__ import annotations

import numpy as np
import pytest

pin = pytest.importorskip(
    "pinocchio",
    reason="pinocchio is a conda C++ dependency; see conda_setup_*.sh")


# A two-joint arm, inline so the test needs no asset on disk (the suite's
# hermeticity rule). j1 rotates about +z at the origin, j2 about +y, 0.2 m out.
TOY_URDF = """<?xml version="1.0"?>
<robot name="toy">
  <link name="base"/>
  <link name="l1"/>
  <link name="l2"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5" effort="1" velocity="1"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5" effort="1" velocity="1"/>
  </joint>
</robot>
"""


@pytest.fixture
def toy():
    model = pin.buildModelFromXML(TOY_URDF)
    return model, model.createData()


def test_pinocchio_parses_a_urdf_from_a_string(toy):
    """``buildModelFromXML`` is what lets every test here stay hermetic, and what
    the C++ side uses for its own fixtures."""
    model, _ = toy
    assert model.nq == 2
    assert model.nv == 2
    assert list(model.names) == ["universe", "j1", "j2"]


def test_urdf_parsing_needs_no_mesh_files(toy):
    """``buildModel`` reads only the kinematic tree. Nothing in this build
    depends on a robot's visual or collision geometry being on disk -- which is
    what lets a URDF be committed without its meshes."""
    model, _ = toy
    assert model.njoints == 3   # universe + 2, with no <visual> in the source


def test_joint_limits_are_read(toy):
    """The limits exist on the model even though nothing enforces them yet.
    Whoever adds joint-limit constraints reads them from here."""
    model, _ = toy
    np.testing.assert_allclose(model.lowerPositionLimit, [-1.5, -1.5])
    np.testing.assert_allclose(model.upperPositionLimit, [1.5, 1.5])


def test_forward_kinematics_places_the_frame(toy):
    """At zero configuration l2 sits at j1's origin plus j2's offset."""
    model, data = toy
    fid = model.getFrameId("l2")
    pin.forwardKinematics(model, data, np.zeros(model.nq))
    pin.updateFramePlacement(model, data, fid)
    np.testing.assert_allclose(data.oMf[fid].translation, [0.2, 0.0, 0.1],
                               atol=1e-12)


def test_the_local_jacobian_is_linear_then_angular(toy):
    """THE ROW-SWAP GUARD.

    Pinocchio stacks a spatial velocity as [v(3); w(3)]; GTSAM's Pose3 tangent is
    [w(3); v(3)]. The FK factor therefore multiplies by a swap matrix, and if
    Pinocchio ever reordered, that factor would go quietly wrong rather than
    raise.

    j2 rotates about its own +y and l2's frame is coincident with it, so in the
    LOCAL frame its column is pure angular velocity about y: v = 0, w = (0,1,0).
    Rows 3-5 carrying the 1.0 is what says angular comes second.
    """
    model, data = toy
    fid = model.getFrameId("l2")
    q = np.zeros(model.nq)
    J = pin.computeFrameJacobian(model, data, q, fid, pin.LOCAL)

    assert J.shape == (6, 2)
    np.testing.assert_allclose(J[:, 1], [0, 0, 0, 0, 1, 0], atol=1e-12)


def test_the_local_frame_is_the_body_frame(toy):
    """``pin.LOCAL`` (not LOCAL_WORLD_ALIGNED) is what the factor asks for,
    because GTSAM's ``compose`` takes its second argument's Jacobian in the body
    frame.

    Bending j2 turns l2's frame relative to j1's axis, so j1's column expressed
    in that body frame must change. (Turning j1 would NOT show this: it carries
    its own axis and the body frame around together, leaving the local column
    invariant.) The two reference frames also disagree at any non-identity
    orientation, which is the distinction the factor depends on.
    """
    model, data = toy
    fid = model.getFrameId("l2")
    bent = np.array([0.0, np.pi / 2])

    J_straight = pin.computeFrameJacobian(model, data, np.zeros(2), fid, pin.LOCAL)
    J_bent = pin.computeFrameJacobian(model, data, bent, fid, pin.LOCAL)
    J_world = pin.computeFrameJacobian(model, data, bent, fid,
                                       pin.LOCAL_WORLD_ALIGNED)

    # j1's column, read in l2's own frame, moves when the body frame turns...
    assert not np.allclose(J_straight[:, 0], J_bent[:, 0], atol=1e-9)
    # ...and body-frame is a genuinely different quantity from world-aligned.
    assert not np.allclose(J_bent[:, 0], J_world[:, 0], atol=1e-9)
