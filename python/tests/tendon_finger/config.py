import numpy as np

from crest_sparse import TendonFingerSolverConfig, TendonInput, PerDiscTendonInput, RoutingAngleFunction, RoutingFunctionParams


def get_K_inv():
    rod_diameter = 0.000512
    youngs_modulus = 40.0e9
    shear_modulus = 15.0e9 
    
    I = (np.pi * rod_diameter**4) / 64.0
    J = 2 * I
    A = (np.pi * rod_diameter**2) / 4.0

    k_bending = youngs_modulus * I
    k_torsion = shear_modulus * J
    k_shear = shear_modulus * A
    k_extension = youngs_modulus * A

    K_inv = np.eye(6)
    K_inv[0,0] = 1 / k_bending
    K_inv[1,1] = 1 / k_bending
    K_inv[2,2] = 1 / k_torsion
    K_inv[3,3] = 1 / k_shear
    K_inv[4,4] = 1 / k_shear
    K_inv[5,5] = 1 / k_extension

    return K_inv


def build_K_inv_per_segment(num_discs, num_between_nodes, segment_types, K_inv_joint=None, bone_stiffness_scale=1e-6):
    """Build a per-segment compliance list for a finger with alternating bones and joints.

    Parameters
    ----------
    num_discs : int
        Number of discs (same as config.num_discs).
    num_between_nodes : int
        Number of nodes between each pair of discs (same as config.num_between_nodes).
    segment_types : list of str
        One entry per *inter-disc gap*, i.e. len == num_discs - 1.
        Each entry is either ``"bone"`` or ``"joint"``.
        Example for 3-bone / 2-joint finger (proximal→distal):
            ["bone", "joint", "bone", "joint", "bone"]
    K_inv_joint : np.ndarray, shape (6,6), optional
        Compliance matrix for a flexible joint segment.
        Defaults to ``get_K_inv()``.
    bone_stiffness_scale : float, optional
        ``K_inv_bone = bone_stiffness_scale * K_inv_joint``.
        Default 1e-6 makes bones ~1e6× stiffer than joints.

    Returns
    -------
    list of np.ndarray
        List of (6,6) compliance matrices, one per rod segment
        (length = num_discs + (num_discs-1)*num_between_nodes - 1).
    """
    if len(segment_types) != num_discs - 1:
        raise ValueError(
            f"segment_types must have num_discs - 1 = {num_discs - 1} entries, "
            f"got {len(segment_types)}")

    if K_inv_joint is None:
        K_inv_joint = get_K_inv()

    K_inv_bone = bone_stiffness_scale * K_inv_joint

    # Total nodes and segments
    num_nodes = num_discs + (num_discs - 1) * num_between_nodes
    num_segments = num_nodes - 1  # = (num_discs - 1) * (num_between_nodes + 1)

    K_inv_list = []
    # Segments per inter-disc gap = num_between_nodes + 1
    segs_per_gap = num_between_nodes + 1

    for gap_idx, seg_type in enumerate(segment_types):
        K = K_inv_bone if seg_type == "bone" else K_inv_joint
        for _ in range(segs_per_gap):
            K_inv_list.append(K.copy())

    assert len(K_inv_list) == num_segments, \
        f"Internal error: expected {num_segments} segments, got {len(K_inv_list)}"

    return K_inv_list


def build_disc_positions_and_segments(bone_joint_lengths):
    """Build disc positions and segment types from bone/joint length specifications.
    
    Each bone gets a disc at its start and end (except first bone has no start disc, 
    last bone has no end disc). Each joint is sandwiched between two discs.
    
    Parameters
    ----------
    bone_joint_lengths : list of tuples
        List of (type, length_m) tuples, where type is "bone" or "joint".
        Example: [("bone", 0.03), ("joint", 0.003), ("bone", 0.03), ("joint", 0.003), ("bone", 0.02)]
    
    Returns
    -------
    disc_positions_normalized : list of float
        Normalized positions (0.0 to 1.0) for each disc.
    segment_types : list of str
        List of "bone" or "joint" for each segment between discs.
    num_discs : int
        Total number of discs.
    total_length : float
        Total rod length in meters.
    
    Example
    -------
    For bone (3cm), joint (2mm), bone (3cm):
        - Disc 0: 0.0 (start of bone 1)
        - Disc 1: 3.0cm (end of bone 1 / start of joint 1)
        - Disc 2: 3.2cm (end of joint 1 / start of bone 2)
        - Disc 3: 6.2cm (end of bone 2)
    """
    # Calculate cumulative positions
    positions = [0.0]
    segment_types = []
    
    for seg_type, length in bone_joint_lengths:
        positions.append(positions[-1] + length)
        segment_types.append(seg_type)
    
    total_length = positions[-1]
    
    # Normalize positions to [0, 1]
    disc_positions_normalized = [pos / total_length for pos in positions]
    
    num_discs = len(disc_positions_normalized)
    
    return disc_positions_normalized, segment_types, num_discs, total_length


def get_tendon_input():
    tendon_input = TendonInput()

    tendon_input.routing_radius = 0.01

    # All tendons have constant (non-spiraling) routing
    tendon_input.functions = [
        RoutingAngleFunction.CONSTANT,  # Tendon 0: 0° (primary actuator)
        RoutingAngleFunction.CONSTANT,  # Tendon 1: 90°
        RoutingAngleFunction.CONSTANT,  # Tendon 2: 180°
        RoutingAngleFunction.CONSTANT   # Tendon 3: 270°
    ]

    # Position tendons at 90° intervals around the finger
    tendon_input.params = [
        RoutingFunctionParams(angle_offset=0.0,           total_angle=0.0),  # 0°
        RoutingFunctionParams(angle_offset=np.pi / 2,     total_angle=0.0),  # 90°
        RoutingFunctionParams(angle_offset=np.pi,         total_angle=0.0),  # 180°
        RoutingFunctionParams(angle_offset=3 * np.pi / 2, total_angle=0.0)   # 270°
    ]

    return tendon_input


def get_base_config():
    config = TendonFingerSolverConfig()

    config.base.use_dense = False
    config.base.linear_solver_type = "MULTIFRONTAL_QR"
    config.rod_length = 0.25
    config.num_discs = 9
    config.num_between_nodes = 3
    config.K_inv = get_K_inv()
    config.sigma_twist_rot = 1.0e-2
    config.sigma_twist_pos = 1.0e-4
    config.sigma_stress_force = 1.0e-4
    config.sigma_stress_moment = 1.0e-5
    config.sigma_base_pos = 1.0e-4
    config.sigma_base_rot = 1.0e-3
    config.tendon_input = get_tendon_input()

    # config.tip_position_meas_std = 1e-3

    # config.dist_load_prior_std = 2e-2
    # config.dist_load_smoothness_std = 1e-2
    # config.fbg_strain_meas_std = 3e-6
    # config.tension_drift_std = 1e-2
    # config.tip_force_drift_std = 1e-1
    # config.dist_load_drift_std = 5e-3

    return config


def _get_tendon_angle_at_disc(tendon_segments, disc_idx, num_segments):
    """Compute the angle (radians) for one tendon at one disc, or NaN if terminated.

    tendon_segments : list of tuples or None, length == num_segments
        Each entry is (start_deg, end_deg), (start_deg, None) for termination,
        or None for already-terminated.
    disc_idx : int
        Which disc (0-indexed).
    num_segments : int
        Total number of bone/joint segments.
    """
    NAN = float('nan')

    # disc 0 -> start of segment 0
    if disc_idx == 0:
        seg = tendon_segments[0]
        if seg is None:
            return NAN
        return np.deg2rad(seg[0])

    # disc_idx > 0 -> end of segment (disc_idx - 1)
    prev_seg_idx = disc_idx - 1

    if prev_seg_idx >= num_segments:
        return NAN

    prev_seg = tendon_segments[prev_seg_idx]
    if prev_seg is None:
        return NAN

    start_deg, end_deg = prev_seg
    if end_deg is None:
        # Tendon terminates: this disc gets the start angle (it's the last disc with a hole)
        return np.deg2rad(start_deg)

    # This disc is the end of previous segment -> use end angle
    return np.deg2rad(end_deg)


def _get_tendon_radius_at_disc(tendon_radius_segments, disc_idx, num_segments, default_radius):
    """Compute the radius for one tendon at one disc, or NaN if terminated.

    tendon_radius_segments : list of tuples or None, length == num_segments
        Each entry is (start_radius, end_radius), (start_radius, None) for termination,
        or None for already-terminated.  Works identically to the angle version but
        interpolates radii instead of angles.
    disc_idx : int
    num_segments : int
    default_radius : float
        Returned when tendon_radius_segments is None (uniform radius mode).
    """
    NAN = float('nan')

    if tendon_radius_segments is None:
        # No per-segment radii specified for this tendon — use default
        return default_radius

    if disc_idx == 0:
        seg = tendon_radius_segments[0]
        if seg is None:
            return NAN
        return seg[0]

    prev_seg_idx = disc_idx - 1
    if prev_seg_idx >= num_segments:
        return NAN

    prev_seg = tendon_radius_segments[prev_seg_idx]
    if prev_seg is None:
        return NAN

    start_r, end_r = prev_seg
    if end_r is None:
        return start_r

    return end_r


def get_6tendon_per_disc_input(bone_joint_spec, disc_positions, num_discs,
                               routing_radius=0.005, tendon_routing_radii=None):
    """Generate per-disc tendon routing for a 6-tendon underactuated finger.

    Tendon routing specification per segment (start_angle -> end_angle in degrees):
      Tendon 1 (passive, lateral):  0->0 metacarpal, 0->0 MCP, 0->90 prox, 90->90 PIP,
                                     90->0 mid, 0->0 DIP, stops at 1st disc distal @ 0
      Tendon 2 (passive, lateral):  0->0 metacarpal, 0->0 MCP, 0->270 prox, 270->270 PIP,
                                     270->0 mid, 0->0 DIP, stops at 1st disc distal @ 0
      Tendon 3 (passive, abduct):   270->270 metacarpal, 270->270 MCP, 270->0 prox, 0->0 PIP,
                                     stops at 1st disc middle @ 0
      Tendon 4 (passive, abduct):   90->90 metacarpal, 90->90 MCP, 90->0 prox, 0->0 PIP,
                                     stops at 1st disc middle @ 0
      Tendon 5 (passive, extensor): 180->180 everywhere, stops at 1st disc middle @ 180
      Tendon 6 (active, flexor):    180->180 everywhere, stops at 1st disc distal @ 180

    Parameters
    ----------
    bone_joint_spec : list of tuples
        The (type, length) spec used to build disc positions.
    disc_positions : list of float
        Normalized disc positions.
    num_discs : int
    routing_radius : float
        Default radius used when tendon_routing_radii is None or for tendons
        without per-segment radii.
    tendon_routing_radii : list of list of tuples or None, optional
        Per-tendon per-segment radius specification, same structure as
        tendon_routing but with (start_radius, end_radius) tuples.
        When None, all tendons use the uniform routing_radius.

    Returns
    -------
    PerDiscTendonInput
    """
    NUM_TENDONS = 6
    num_segments = len(bone_joint_spec)

    # Each tendon: list of (start_angle_deg, end_angle_deg) per segment, or None for terminated.
    # (start_deg, None) means tendon terminates: the start disc of this segment is its LAST hole.
    # None means tendon has already terminated before this segment.
    # Segments: metacarpal(bone), MCP(joint), proximal(bone), PIP(joint), middle(bone), DIP(joint), distal(bone)
    tendon_routing = [
        # Tendon 1: lateral stabilizer (0 deg -> spirals to 90 -> spirals back to 0, stops at distal)
        [(0, 0), (0, 0), (0, 90), (90, 90), (90, 0), (0, 0), (0, None)],
        # Tendon 2: lateral stabilizer opposite (0 deg -> spirals to 270 -> spirals back to 0, stops at distal)
        [(0, 0), (0, 0), (0, 270), (270, 270), (270, 0), (0, 0), (0, None)],
        # Tendon 3: abduction (270 deg -> spirals to 0 along proximal, stops at middle)
        [(270, 270), (270, 270), (270, 0), (0, 0), (0, None), None, None],
        # Tendon 4: adduction (90 deg -> spirals to 0 along proximal, stops at middle)
        [(90, 90), (90, 90), (90, 0), (0, 0), (0, None), None, None],
        # Tendon 5: passive extensor (180 deg constant, stops at middle)
        [(180, 180), (180, 180), (180, 180), (180, 180), (180, None), None, None],
        # Tendon 6: active flexor (180 deg constant, goes to distal)
        [(180, 180), (180, 180), (180, 180), (180, 180), (180, 180), (180, 180), (180, None)],
    ]

    # Expand to per-disc angles
    hole_angles = []
    for disc_idx in range(num_discs):
        angles = []
        for tendon_idx in range(NUM_TENDONS):
            angle = _get_tendon_angle_at_disc(
                tendon_routing[tendon_idx], disc_idx, num_segments)
            angles.append(angle)
        hole_angles.append(angles)

    # Expand to per-disc radii (only if per-tendon radii are specified)
    hole_radii = []
    if tendon_routing_radii is not None:
        for disc_idx in range(num_discs):
            radii = []
            for tendon_idx in range(NUM_TENDONS):
                r = _get_tendon_radius_at_disc(
                    tendon_routing_radii[tendon_idx], disc_idx, num_segments,
                    default_radius=routing_radius)
                radii.append(r)
            hole_radii.append(radii)

    per_disc = PerDiscTendonInput()
    per_disc.num_tendons = NUM_TENDONS
    per_disc.routing_radius = routing_radius
    per_disc.hole_angles = hole_angles
    if hole_radii:
        per_disc.hole_radii = hole_radii

    return per_disc


def _build_default_tendon_routing_radii(bone_joint_spec, base_radius=0.005, min_radius=0.003, num_tendons=6):
    """Build a default tapering radius spec matching any bone_joint_spec length.

    Creates a linear taper from base_radius at the proximal end to
    min_radius at the distal end, identical for all tendons.
    """
    n_segments = len(bone_joint_spec)
    radii_at_boundaries = np.linspace(base_radius, min_radius, n_segments + 1)
    single_tendon = [
        (radii_at_boundaries[i], radii_at_boundaries[i + 1])
        for i in range(n_segments)
    ]
    return [list(single_tendon) for _ in range(num_tendons)]


def get_6tendon_config(bone_joint_spec=None, tendon_routing_radii=None):
    """Get a full solver config for a 6-tendon underactuated finger.

    Parameters
    ----------
    bone_joint_spec : list of tuples, optional
        List of (type, length_m) tuples defining the finger anatomy.
        Must have exactly 7 segments (4 bones + 3 joints).
        Defaults to a standard finger if not provided.
    tendon_routing_radii : list of list of tuples, optional
        Per-tendon per-segment radius specification.
        Defaults to a linear taper from 5mm to 3mm if not provided.
    """
    config = TendonFingerSolverConfig()

    config.base.use_dense = False
    config.base.linear_solver_type = "MULTIFRONTAL_QR"
    config.num_tendons = 6
    config.num_between_nodes = 3

    if bone_joint_spec is None:
        bone_joint_spec = [
            ("bone", 0.05),     # metacarpal
            ("joint", 0.01),    # MCP
            ("bone", 0.05),     # proximal phalanx
            ("joint", 0.01),    # PIP
            ("bone", 0.03),     # middle phalanx
            ("joint", 0.01),    # DIP
            ("bone", 0.02),     # distal phalanx
        ]

    disc_positions, segment_types, num_discs, total_length = \
        build_disc_positions_and_segments(bone_joint_spec)

    config.rod_length = total_length
    config.num_discs = num_discs
    config.disc_positions_normalized = disc_positions

    config.K_inv = get_K_inv()
    config.K_inv_per_segment = build_K_inv_per_segment(
        num_discs=num_discs,
        num_between_nodes=config.num_between_nodes,
        segment_types=segment_types,
        bone_stiffness_scale=1e-6
    )

    if tendon_routing_radii is None:
        tendon_routing_radii = _build_default_tendon_routing_radii(bone_joint_spec)

    config.per_disc_tendon_input = get_6tendon_per_disc_input(
        bone_joint_spec, disc_positions, num_discs, tendon_routing_radii=tendon_routing_radii
    )

    config.sigma_twist_rot = 1.0e-2
    config.sigma_twist_pos = 1.0e-4
    config.sigma_stress_force = 1.0e-4
    config.sigma_stress_moment = 1.0e-5
    config.sigma_base_pos = 1.0e-4
    config.sigma_base_rot = 1.0e-3

    return config

