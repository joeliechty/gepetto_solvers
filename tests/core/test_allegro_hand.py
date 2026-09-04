"""The Allegro hand driven through the ordinary Python solver stack.

Phase 3 proved the rigid kinematics against the C++ interface. This is the layer
above: an ``AllegroHand`` handed to ``HandFKSolver`` / ``HandIKSolver`` exactly
as ``TendonHand5F`` is, with the same params object and the same task
constraints.

What makes these tests worth having is that they are the same calls the tendon
hand makes. If the seam leaked -- if a solver reached for a tendon field, a
scalar commanded value, or the tendon hand's measured wrist pose -- it would show
up here and nowhere else.
"""

from __future__ import annotations

import numpy as np
import pytest

from _pkg import hands, solvers

pytest.importorskip(
    "pinocchio",
    reason="pinocchio is a conda C++ dependency; see conda_setup_*.sh")

from gepetto_solvers.core.hands.allegro import meshes as allegro_meshes  # noqa: E402
from gepetto_solvers.core.hands.allegro import spec as allegro_spec  # noqa: E402


@pytest.fixture(scope="module")
def hand():
    return hands.get_hand("allegro")


@pytest.fixture
def params(hand):
    """Params posed at the hand's own default, with the three-digit grasp set."""
    wrist, means = hand.default_pose()
    p = solvers.HandSolveParams()
    p.wrist_pose = wrist
    p.joint_targets = [list(m) for m in means]
    p.contact_fingers = [n in hand.default_contact_digits
                         for n in hand.digit_names]
    return p


# ---------------------------------------------------------------------------
# The hand itself.
# ---------------------------------------------------------------------------

def test_it_is_registered(hand):
    assert hand.name == "allegro"
    assert hand.kinematics == "rigid_urdf"
    assert hand.digit_names == ["index", "middle", "ring", "thumb"]


def test_every_joint_is_driven(hand):
    """The first hand to drive more than one actuator per digit. Anything
    reading ``drive_indices[0]`` is wrong here, which is what the
    ``single_drive`` feature gates."""
    assert hand.actuation.n == 4
    assert hand.actuation.drive_indices == (0, 1, 2, 3)
    assert hand.actuation.passive_indices == ()
    assert "single_drive" not in hand.features


def test_it_declares_no_tendon_features(hand):
    """No tendons, no pinch table, no calibration landmarks, no measured close
    ramp -- each gates a workbench panel off.

    ``robot_plan`` is the exception and IS declared: this hand has a robot
    behind it. What it does NOT have is ``displacement``, and that absence is
    load-bearing -- it is the single thing ``robot_plan.command_kind`` reads to
    decide that a plan for this hand carries joint positions in radians rather
    than tendon displacements in metres.
    """
    assert hand.features == frozenset({"robot_plan"})
    assert hand.features <= hands.FEATURES
    assert "displacement" not in hand.features


def test_asking_for_a_single_driven_value_is_refused(hand):
    """``drive_value`` would have to pick one of four. It says so instead."""
    with pytest.raises(ValueError, match="4 actuators per digit"):
        hand.actuation.drive_value(np.zeros(4))


def test_it_has_no_measured_pinch_table(hand):
    """None is the honest answer, and the pre-grasp centroid constraint has to
    be able to get it rather than an exception or a wrong default."""
    assert hand.pinch_pose([True, True, False, True]) is None


def test_digit_configs_are_fresh_each_call(hand):
    """The attach_* family mutates these in place."""
    a, b = hand.digit_configs(), hand.digit_configs()
    assert a[0][1] is not b[0][1]


# ---------------------------------------------------------------------------
# Commanding it.
# ---------------------------------------------------------------------------

def test_joint_targets_command_the_posture(hand, params):
    means = hand.actuation_means(params)
    assert len(means) == 4
    np.testing.assert_allclose(means[0], hand.DEFAULT_FINGER_Q)
    np.testing.assert_allclose(means[3], hand.DEFAULT_THUMB_Q)


def test_an_uncommanded_hand_falls_back_to_neutral(hand):
    p = solvers.HandSolveParams()
    p.joint_targets = None
    for m in hand.actuation_means(p):
        np.testing.assert_allclose(m, np.zeros(4))


def test_a_wrongly_shaped_joint_target_is_refused(hand):
    p = solvers.HandSolveParams()
    p.joint_targets = [[0.0, 0.1, 0.2]] * 4      # three joints, not four
    with pytest.raises(ValueError, match="expected"):
        hand.actuation_means(p)


def test_the_wrong_number_of_digits_is_refused(hand):
    p = solvers.HandSolveParams()
    p.joint_targets = [[0.0] * 4] * 5
    with pytest.raises(ValueError, match="5 entries"):
        hand.actuation_means(p)


# ---------------------------------------------------------------------------
# Through the solvers.
# ---------------------------------------------------------------------------

def test_fk_poses_it_exactly(hand, params):
    """Seeded at the same posture the joint prior is centred on, FK has nothing
    to solve: zero residual in one iteration."""
    result = solvers.HandFKSolver(params, hand).solve()
    assert result.meta.iterations <= 2
    assert result.meta.error < 1e-12
    for i, name in enumerate(result.finger_names):
        np.testing.assert_allclose(result.frames[0][name].actuation(),
                                   params.joint_targets[i], atol=1e-9)


def test_fk_reports_the_commanded_wrist(hand, params):
    result = solvers.HandFKSolver(params, hand).solve()
    np.testing.assert_allclose(result.wrist_pose(0), params.wrist_pose, atol=1e-6)


def test_the_default_pose_opens_a_pre_grasp_around_the_object(hand, params):
    """A default that aims the hand nowhere is worse than none. The tendon
    hand's measured wrist pose points THIS hand away from the object -- its
    fingers extend +z where the tendon palm lies along -x -- so Allegro carries
    its own.

    AROUND the object, not ON it: the default is an open pre-grasp (see
    ``DEFAULT_FINGER_Q``), so what is asserted is that every grasp digit is
    outside the surface and within closing distance of it, and that the digit
    the grasp does NOT command is further out than all of them. An earlier
    default seated the tips on this particular object to 0.6 mm, which read as a
    tighter test than it was: it was a grasp already made, and it sat inside the
    object the workbench actually opens on.
    """
    result = solvers.HandFKSolver(params, hand).solve()
    gaps = result.surface_gaps(0)
    for name in hand.default_contact_digits:
        assert 0.0 < gaps[name] < 0.05, (name, gaps[name])
    assert gaps["ring"] > max(gaps[n] for n in hand.default_contact_digits)


def test_ik_drives_the_grasp_digits_onto_the_object(hand, params):
    """The whole point: the same contact constraint the tendon hand uses,
    applied to a mechanism that is not a rod.

    From the open pre-grasp the default now starts at, one IK solve closes
    ~37 mm of gap to under 3 mm and stops there -- not for want of iterations
    (it converges in 19 of the 40 it may take), but because the last of the
    approach is a WRIST translation, and ``sigma_wrist_pos`` defaults to 1e-4:
    the wrist prior holds it. That is why the residual is asserted EQUAL across
    the three digits as well as small -- a common offset is the signature of a
    held wrist, where a finger that failed to reach would miss on its own. The
    workbench's phase pipeline loosens the wrist for exactly this stage.
    """
    result = solvers.HandIKSolver(params, hand).solve()
    gaps = result.surface_gaps(0)
    for name in hand.default_contact_digits:
        assert abs(gaps[name]) < 5e-3, (name, gaps[name])
    grasped = [gaps[n] for n in hand.default_contact_digits]
    assert max(grasped) - min(grasped) < 1e-4, grasped


def test_ik_leaves_the_uncommanded_digit_alone(hand, params):
    """`ring` is not in the contact set, so nothing should be driving it onto
    the object -- a masked-off digit keeps collision avoidance and no more."""
    result = solvers.HandIKSolver(params, hand).solve()
    assert result.surface_gaps(0)["ring"] > 0.01


def test_the_solved_state_has_the_neutral_shape(hand, params):
    result = solvers.HandFKSolver(params, hand).solve()
    state = result.state(0)
    assert len(state.digits) == 4
    for d in state.digits:
        assert d.extras is None            # no tendon payload
        assert list(d.displacement) == []  # actuation IS position
        # mount + every moving link + the fingertip
        assert len(d.sites) == allegro_spec.SITES_PER_DIGIT


def test_the_finger_sol_accessors_work_on_a_rigid_hand(hand, params):
    """The accessors are what let a reader stay mechanism-neutral. `tendon()`
    returning None is how one that cannot finds out."""
    result = solvers.HandFKSolver(params, hand).solve()
    fs = result.frames[0]["index"]
    assert fs.num_sites() == allegro_spec.SITES_PER_DIGIT
    # Every site but the fixed mount carries a collision sphere.
    assert fs.sphere_sites() == list(range(1, allegro_spec.SITES_PER_DIGIT))
    assert fs.tendon() is None
    assert fs.tip_pose().shape == (4, 4)
    np.testing.assert_allclose(fs.tip_point(), fs.tip_pose()[:3, 3])


# ---------------------------------------------------------------------------
# Where it bolts to the arm.
# ---------------------------------------------------------------------------

def test_the_mount_is_the_measured_flange_transform(hand):
    """130 mm straight out the flange's +z, axes aligned.

    Measured on the robot with the hand's own driver up::

        ros2 run tf2_ros tf2_echo lbr_link_ee palm_link
        - Translation: [0.000, 0.000, 0.130]
        - Rotation: in Quaternion (xyzw) [0.000, 0.000, 0.000, 1.000]
    """
    T = hand.mount_pose()
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [0.0, 0.0, 0.130], atol=1e-12)


def test_the_mount_is_fresh_every_call(hand):
    """Callers assign it into ``params.wrist_pose`` and mutate poses in place."""
    a, b = hand.mount_pose(), hand.mount_pose()
    assert a is not b
    a[2, 3] = 99.0
    assert hand.mount_pose()[2, 3] == pytest.approx(0.130)


def test_the_mount_measurement_is_against_the_frame_the_solver_uses(hand):
    """THE ASSUMPTION UNDER THE MEASUREMENT.

    The tf was measured flange-to-`palm_link`, but the solver's wrist variable
    is the model ROOT -- the rigid kinematics resolves every digit's mount
    against it. Storing the tf verbatim as ``T_flange<-wrist`` is only correct
    while those two frames coincide, which on this description they do: a bare
    `world` link joined to `palm_link` at the origin.

    V4's did NOT -- it put `palm_link` 95 mm up the root's +z -- so a variant
    swap could reintroduce exactly that offset and mount the hand 95 mm off the
    arm with nothing on screen to say so. This is the check that would catch it.
    """
    from gepetto_solvers.core.hands.allegro import meshes as m

    np.testing.assert_allclose(m._fixed_placement("palm_link"), np.eye(4),
                               atol=1e-12)


def test_the_two_hands_mount_differently():
    """They bolt to the arm differently and the transforms were measured
    differently -- a CAD fit against an Onshape assembly for one, a `tf2_echo`
    off the running robot for the other. One shared constant would mount
    whichever hand it was not measured on badly wrong."""
    a = hands.get_hand("allegro").mount_pose()
    t = hands.get_hand("tendon_5f").mount_pose()
    assert not np.allclose(a, t)


# ---------------------------------------------------------------------------
# Both hands, side by side.
# ---------------------------------------------------------------------------

def test_the_two_hands_report_different_defaults():
    """They are different robots; a shared default would aim one of them wrong."""
    a_wrist, _ = hands.get_hand("allegro").default_pose()
    t_wrist, _ = hands.get_hand("tendon_5f").default_pose()
    assert not np.allclose(a_wrist, t_wrist)


def test_both_hands_satisfy_the_protocol():
    for name in hands.registered_hands():
        hand = hands.get_hand(name)
        assert isinstance(hand, hands.Hand), name
        assert hand.features <= hands.FEATURES, name
        assert hand.opposing_index in range(len(hand.digit_names)) or \
            hand.opposing_index == -1, name


# ---------------------------------------------------------------------------
# The chain is the whole chain.
# ---------------------------------------------------------------------------

def test_every_moving_link_has_a_site(hand):
    """One site per movable joint, plus the fingertip, plus the fixed mount.

    Leaving a link out does not merely coarsen the picture -- it MERGES two
    joints. Omitting the distal link made joint_2 and joint_3 each move exactly
    one drawn point (the tip), so the two sliders appeared to drive the same
    thing, and the last segment was drawn as one 78 mm bar where the hand has
    38 mm and 40 mm about a joint between them.
    """
    assert allegro_spec.SITES_PER_DIGIT == allegro_spec.DOF_PER_DIGIT + 2
    for name in hand.digit_names:
        assert len(allegro_spec._SITE_FRAMES[name]) == allegro_spec.DOF_PER_DIGIT + 1


def test_each_joint_moves_a_distinct_part_of_the_chain(hand, params):
    """THE SLIDER BUG, as a measurement.

    In a serial chain joint j moves every site below it and nothing above, so
    the moved sets are NESTED. That alone is not enough to tell the sliders
    apart, though: the two base joints -- abduction and flexion -- move the same
    set and differ only in direction. What has to hold is that no two joints
    produce the same MOTION, which is what the bug broke: with the distal link
    missing, joint_2 and joint_3 each moved exactly one drawn point, the tip,
    along nearly the same arc.
    """
    from gepetto_solvers.core.solvers import HandFKSolver

    def chain(targets):
        p = solvers.HandSolveParams()
        p.wrist_pose = params.wrist_pose
        p.joint_targets = targets
        fs = HandFKSolver(p, hand).solve().frames[0]["index"]
        return np.array([fs.site_point(i) for i in range(fs.num_sites())])

    base_targets = [list(q) for q in params.joint_targets]
    base = chain(base_targets)

    moved_sets, fields = [], []
    for j in range(hand.actuation.n):
        targets = [list(q) for q in base_targets]
        targets[0][j] += 0.3
        delta = chain(targets) - base
        moved_sets.append({i for i, d in enumerate(np.linalg.norm(delta, axis=1))
                           if d > 1e-4})
        fields.append(delta.ravel() / np.linalg.norm(delta))

    # Nested: a deeper joint never moves a site a shallower one leaves alone.
    for a, b in zip(moved_sets, moved_sets[1:]):
        assert b <= a, (a, b)

    # And STRICTLY nested past the base pair: each joint down the chain has one
    # fewer link below it, so it moves strictly fewer sites. This is the
    # assertion the bug trips -- with the distal link missing, joint_2 and
    # joint_3 moved the identical set {tip}.
    #
    # The base pair is excluded because abduction and flexion legitimately move
    # the same set: the link between them lies on the abduction axis, so it does
    # not translate. They differ in DIRECTION, which the check below covers.
    for a, b in zip(moved_sets[1:], moved_sets[2:]):
        assert b < a, (a, b)

    # ...and no two joints move the hand the same way.
    for i in range(len(fields)):
        for k in range(i + 1, len(fields)):
            assert abs(float(fields[i] @ fields[k])) < 0.99, (
                f"j{i} and j{k} move the chain almost identically")


def test_the_drawn_segments_are_the_real_link_lengths(hand, params):
    """The finger a viewer sees must be the finger the URDF describes.

    The V5 type-B index is 17 mm base, 43.1 mm proximal, 38 mm medial and 40 mm
    to the fingertip frame. A missing site shows up here as one long bar instead
    of two.

    These numbers are also what separates the vendored description from its
    neighbours: type A's chain is 21 / 51 / 38.4 / 26.7 mm, and V4's was
    16.4 / 54 / 38.4 / 26.7. Swap the URDF for another variant and this fails,
    which is the point -- the calibrated default posture below it is measured
    against THESE lengths.
    """
    from gepetto_solvers.core.solvers import HandFKSolver

    fs = HandFKSolver(params, hand).solve().frames[0]["index"]
    pts = np.array([fs.site_point(i) for i in range(fs.num_sites())])
    segments = np.linalg.norm(np.diff(pts, axis=0), axis=1) * 1e3

    # The first is the mount-to-first-link bar, which is degenerate by
    # construction: a link's frame sits at its own joint's origin.
    assert segments[0] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(segments[1:], [17.0, 43.1, 38.0, 40.0], atol=0.2)


# ---------------------------------------------------------------------------
# Visual meshes are visual only.
# ---------------------------------------------------------------------------

def test_the_link_meshes_are_present_and_placed(hand):
    """One per link plus the palm, resolved to files that actually exist.

    The URDF is vendored VERBATIM, so the names it gives are upstream's STLs
    while the directory holds the decimated glTF we converted them to. That
    suffix swap is the one place the two can drift apart, and a checkout where
    it went wrong degrades silently to a skeleton rather than raising.
    """
    assert allegro_meshes.available()
    meshes = hand.visual_meshes()
    assert all(path.suffix == ".glb" for _, path, _ in meshes)
    assert len(meshes) == 1 + len(hand.digit_names) * (
        allegro_spec.SITES_PER_DIGIT - 1)
    assert sum(1 for attach, _, _ in meshes if attach is None) == 1, "one palm"
    for attach, path, T_local in meshes:
        assert path.exists(), path
        assert np.asarray(T_local).shape == (4, 4)
        if attach is not None:
            digit, site = attach
            assert 0 <= digit < len(hand.digit_names)
            assert 1 <= site < allegro_spec.SITES_PER_DIGIT


def test_collision_is_spheres_and_never_the_meshes(hand, params):
    """THE SEPARATION.

    The hand's shape is drawn from its URDF link meshes; what the SOLVE reasons
    about is the sphere set on its sites. Those must stay disjoint -- a mesh
    must never reach the factor graph.

    Checked at the boundary the graph is actually built from: the HandSpec's
    per-digit EnvironmentConfig carries site indices and sphere radii, and
    nothing that could name a mesh.
    """
    solver = solvers.HandIKSolver(params, hand)
    solver._attach_environment()
    spec = solver._hand_spec()

    sites, _ = hand.collision_sites(0)
    for env in spec.env:
        assert env is not None
        assert list(env.collision_node_indices) == sites
        assert len(env.collision_node_radii) == len(sites)
        # Every collision radius is a real sphere, not a placeholder.
        assert all(r > 0 for r in env.collision_node_radii)

    # The kinematics payload names joints and frames -- never a mesh file.
    text = " ".join(
        j for d in spec.kinematics_config.digits
        for j in list(d.joints) + list(d.site_frames))
    for suffix in (".glb", ".gltf", ".stl", ".obj"):
        assert suffix not in text, suffix


def test_a_solve_is_identical_without_the_meshes(hand, params, monkeypatch):
    """The strongest form of the same claim: delete the meshes and the numbers
    do not move. If a mesh ever leaked into the graph, this is what would
    catch it."""
    with_meshes = solvers.HandIKSolver(params, hand).solve()

    monkeypatch.setattr(type(hand), "visual_meshes", lambda self: [])
    without = solvers.HandIKSolver(params, hand).solve()

    for a, b in zip(with_meshes.state(0).digits, without.state(0).digits):
        np.testing.assert_allclose(a.actuation.mean, b.actuation.mean, atol=1e-12)
        np.testing.assert_allclose(a.sites[-1].pose.mean, b.sites[-1].pose.mean,
                                   atol=1e-12)


def test_the_meshes_carry_the_urdf_s_own_visual_placement(hand):
    """THE SCATTERED-PARTS BUG, and the thousand-times-too-big one.

    Wonik authors every V5 mesh in one shared assembly frame, in MILLIMETRES,
    and each link's ``<visual>`` block carries the ``<origin>`` that brings its
    own part back to its joint plus the ``scale`` that puts it in metres. Both
    are large: the palm's origin is ``xyz="0.02 0 -0.1" rpy="0 3.14 1.57"``.

    A renderer that assumes an identity visual origin -- which is what a hand
    whose meshes are authored per link gets away with, and what this repository
    assumed when its only URDF hand was V4 -- draws twenty-one parts scattered
    across a 200 mm cube. One that ignores the scale draws a 70 m finger.

    Checked against the KINEMATIC TREE, which is independent of the visual
    blocks: a link's mesh is the part bolted to that link, so it must contain
    the link's own frame origin and be the size of a hand part.
    """
    trimesh = pytest.importorskip("trimesh")

    checked = 0
    for _, path, T_local in hand.visual_meshes():
        mesh = trimesh.load(path, force="mesh")
        mesh.apply_transform(np.asarray(T_local, float))
        lo, hi = mesh.bounds

        # Its own frame origin is inside it (4 mm of slack for the fingertip,
        # whose frame sits just proximal of the moulded pad).
        assert np.all(lo <= 4e-3) and np.all(hi >= -4e-3), (
            f"{path.name}: placed at {(lo * 1e3).round(1)}..{(hi * 1e3).round(1)} mm, "
            f"which does not contain the frame it hangs on -- its <visual> "
            f"<origin> was probably dropped")

        # And it is a hand part, not a metre-scale one: the palm is the biggest
        # thing here at 128 mm.
        assert mesh.extents.max() < 0.15, (
            f"{path.name}: {(mesh.extents * 1e3).round(1)} mm -- the mesh scale "
            f"was probably not applied")
        checked += 1

    assert checked == 21, f"expected every link mesh, checked {checked}"


def test_the_placed_meshes_assemble_into_a_hand(hand):
    """The parts, put where the SOLVE says their sites are, are one solid.

    The per-link check above cannot see a mesh attached to the wrong site: each
    part would still sit correctly in its own frame. This one composes each
    ``T_local`` with the site pose the renderer actually draws it at, and asks
    for what a hand looks like -- consecutive links touching, and the whole
    thing inside a hand-sized box rather than scattered.
    """
    trimesh = pytest.importorskip("trimesh")

    p = solvers.HandSolveParams()
    p.wrist_pose = np.eye(4)
    p.joint_targets = [[0.0] * hand.actuation.n for _ in hand.digit_names]
    frame = solvers.HandFKSolver(p, hand).solve().frames[0]

    placed = {}
    for attach, path, T_local in hand.visual_meshes():
        if attach is None:
            world = np.eye(4)                      # the palm rides on the wrist
        else:
            digit, site = attach
            world = np.asarray(
                frame[hand.digit_names[digit]].marginals.sites[site].pose.mean,
                float)
        mesh = trimesh.load(path, force="mesh")
        mesh.apply_transform(world @ np.asarray(T_local, float))
        placed[attach] = mesh.bounds

    # Consecutive links of a digit overlap -- a finger is not a string of beads.
    for d, name in enumerate(hand.digit_names):
        for site in range(1, allegro_spec.SITES_PER_DIGIT - 1):
            a, b = placed[(d, site)], placed[(d, site + 1)]
            separation = np.maximum(np.maximum(a[0], b[0]) - np.minimum(a[1], b[1]),
                                    0.0)
            assert np.all(separation < 2e-3), (
                f"{name} site {site} and {site + 1} are "
                f"{(separation * 1e3).round(1)} mm apart -- the chain is broken")

    # ...and the assembled hand is hand-sized. A dropped visual origin scatters
    # the parts over a box several times this.
    bounds = np.array(list(placed.values()))
    span = bounds[:, 1].max(axis=0) - bounds[:, 0].min(axis=0)
    assert np.all(span < 0.30), f"assembled hand spans {(span * 1e3).round(1)} mm"


def test_the_fingertip_radius_is_the_measured_pad(hand):
    """The contact sphere is a claim about the real fingertip, so measure it.

    ``tip_radii`` is what the contact constraint drives onto a surface, centred
    on the ``link_*_tip`` frame. That frame sits INSIDE the moulded fingertip,
    so the radius has to come off the mesh rather than off a drawing -- and it
    changed by 8 mm between V4 and V5, which is enough to seat a whole default
    posture inside the object.
    """
    trimesh = pytest.importorskip("trimesh")

    tips = [e for e in hand.visual_meshes()
            if e[0] is not None and e[0][1] == allegro_spec.SITES_PER_DIGIT - 1]
    assert len(tips) == len(hand.digit_names)

    for (digit, _), path, T_local in tips:
        mesh = trimesh.load(path, force="mesh")
        mesh.apply_transform(np.asarray(T_local, float))
        v = mesh.vertices
        # The pad: the surface on the flexion side (+x), which is what closes
        # onto an object. Taken along a narrow ray so the collar behind the
        # fingertip cannot stand in for it.
        pad = v[np.linalg.norm(v[:, 1:], axis=1) < 3e-3][:, 0].max()
        assert hand.tip_radii[digit] == pytest.approx(pad, abs=1e-3), (
            f"{path.name}: pad is {pad * 1e3:.1f} mm but tip_radii says "
            f"{hand.tip_radii[digit] * 1e3:.1f} mm")


def test_the_palm_mesh_meets_the_finger_mounts(hand):
    """THE DISPLACED-PALM BUG: the palm mesh was attached to the wrong frame.

    It rides on the WRIST, but the mesh belongs to `palm_link` -- and on V4 the
    URDF put that 95 mm up the root's +Z through a fixed joint, so a palm drawn
    at the wrist landed a whole palm-height low, hanging below the finger bases
    with a ~93 mm gap instead of meeting them. On the V5 description the two
    frames coincide (a bare `world` link joined to `palm_link` at the origin),
    which makes that particular offset the identity -- and makes this check
    matter MORE, not less: nothing about the current file would complain if the
    composition were dropped, and the next variant to move the palm would come
    back silently wrong.

    The check is that the palm's top face reaches the digits' mounts -- which is
    a statement about the assembled hand, not about one transform, so it stays
    true however the placement is arrived at.
    """
    trimesh = pytest.importorskip("trimesh")

    p = solvers.HandSolveParams()
    p.wrist_pose = np.eye(4)
    p.joint_targets = [[0.0] * hand.actuation.n for _ in hand.digit_names]
    frame = solvers.HandFKSolver(p, hand).solve().frames[0]

    palm = next(e for e in hand.visual_meshes() if e[0] is None)
    mesh = trimesh.load(palm[1], force="mesh")
    mesh.apply_transform(np.asarray(palm[2], float))
    palm_top = mesh.bounds[1][2]

    mounts = [frame[n].site_point(0)[2] for n in hand.digit_names]
    # Every digit hangs off the palm, so no mount may sit above its top face by
    # more than a millimetre or float outside it below.
    assert max(mounts) <= palm_top + 1e-3, (
        f"finger mounts reach {max(mounts):.4f} but the palm stops at "
        f"{palm_top:.4f} -- the palm is too low")
    assert palm_top - max(mounts) < 0.01, (
        f"palm top {palm_top:.4f} floats {palm_top - max(mounts):.4f} m above "
        f"the highest mount -- the palm is too high")
