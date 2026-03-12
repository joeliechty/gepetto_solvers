import numpy as np

from crest_sparse import TendonRobotSolverConfig, TendonInput, RoutingAngleFunction, RoutingFunctionParams


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
        Example: [("bone", 0.03), ("joint", 0.002), ("bone", 0.03), ("joint", 0.002), ("bone", 0.02)]
    
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
    config = TendonRobotSolverConfig()

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

