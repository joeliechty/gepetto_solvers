#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "TendonFingerModel.h"
#include "TendonFingerSolver.h"
#include "utils/EnvironmentFactors.h"

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>

namespace py = pybind11;

void bind_tendon_finger(py::module& m) {
    // Tendon routing types. Declared in TendonFingerModel.h and bound here --
    // they used to be registered from tendon_robot/pybind.cpp, which was their
    // only binding site even though the finger and hand are what use them.
    py::enum_<RoutingAngleFunction>(m, "RoutingAngleFunction")
        .value("CONSTANT", RoutingAngleFunction::CONSTANT)
        .value("LINEAR", RoutingAngleFunction::LINEAR)
        .export_values();

    py::class_<RoutingFunctionParams>(m, "RoutingFunctionParams")
        .def(py::init<>())
        .def(py::init<double, double>(), py::arg("angle_offset"), py::arg("total_angle"))
        .def_readwrite("angle_offset", &RoutingFunctionParams::angle_offset)
        .def_readwrite("total_angle", &RoutingFunctionParams::total_angle);

    py::class_<TendonInput>(m, "TendonInput")
        .def(py::init<>())
        .def_readwrite("functions", &TendonInput::functions)
        .def_readwrite("params", &TendonInput::params)
        .def_readwrite("routing_radius", &TendonInput::routing_radius);

    py::class_<PerDiscTendonInput>(m, "PerDiscTendonInput")
        .def(py::init<>())
        .def_readwrite("num_tendons", &PerDiscTendonInput::num_tendons)
        .def_readwrite("routing_radius", &PerDiscTendonInput::routing_radius)
        .def_readwrite("hole_angles", &PerDiscTendonInput::hole_angles)
        .def_readwrite("hole_radii", &PerDiscTendonInput::hole_radii);

    // Resolved routing, read back off a finger's marginals. solvers.py uses
    // disc_pose_idx to map a disc index to its rod node.
    py::class_<TendonConfig>(m, "TendonConfig")
        .def(py::init<>())
        .def_readwrite("num_discs", &TendonConfig::num_discs)
        .def_readwrite("num_tendons", &TendonConfig::num_tendons)
        .def_readwrite("routing_radius", &TendonConfig::routing_radius)
        .def_readwrite("disc_pose_idx", &TendonConfig::disc_pose_idx)
        .def_readwrite("no_disc_pose_idx", &TendonConfig::no_disc_pose_idx)
        .def_readwrite("hole_locations", &TendonConfig::hole_locations);

    py::class_<TendonFingerSolverConfig>(m, "TendonFingerSolverConfig")
        .def(py::init<>())
        .def_readwrite("base", &TendonFingerSolverConfig::base)
        .def_readwrite("rod_length", &TendonFingerSolverConfig::rod_length)
        .def_readwrite("num_discs", &TendonFingerSolverConfig::num_discs)
        .def_readwrite("num_between_nodes", &TendonFingerSolverConfig::num_between_nodes)
        .def_readwrite("num_tendons", &TendonFingerSolverConfig::num_tendons)
        .def_readwrite("K_inv", &TendonFingerSolverConfig::K_inv)
        .def_readwrite("K_inv_per_segment", &TendonFingerSolverConfig::K_inv_per_segment)
        .def_readwrite("disc_positions_normalized", &TendonFingerSolverConfig::disc_positions_normalized)
        .def_readwrite("tip_radius", &TendonFingerSolverConfig::tip_radius)
        .def_readwrite("sigma_twist_rot", &TendonFingerSolverConfig::sigma_twist_rot)
        .def_readwrite("sigma_twist_pos", &TendonFingerSolverConfig::sigma_twist_pos)
        .def_readwrite("sigma_stress_force", &TendonFingerSolverConfig::sigma_stress_force)
        .def_readwrite("sigma_stress_moment", &TendonFingerSolverConfig::sigma_stress_moment)
        .def_readwrite("sigma_base_pos", &TendonFingerSolverConfig::sigma_base_pos)
        .def_readwrite("sigma_base_rot", &TendonFingerSolverConfig::sigma_base_rot)
        // --- Planar-bending approximation (keyed discs: bend about local +y only) ---
        .def_readwrite("planar_bending", &TendonFingerSolverConfig::planar_bending)
        .def_readwrite("sigma_planar_bend", &TendonFingerSolverConfig::sigma_planar_bend)
        .def_readwrite("sigma_planar_twist", &TendonFingerSolverConfig::sigma_planar_twist)
        .def_readwrite("tendon_input", &TendonFingerSolverConfig::tendon_input)
        .def_readwrite("per_disc_tendon_input", &TendonFingerSolverConfig::per_disc_tendon_input)
        .def_readwrite("base_pose", &TendonFingerSolverConfig::base_pose)
        .def_readwrite("use_hand_base", &TendonFingerSolverConfig::use_hand_base)
        .def_readwrite("hand_base_offset", &TendonFingerSolverConfig::hand_base_offset)
        .def_readwrite("sphere_contact", &TendonFingerSolverConfig::sphere_contact)
        .def_readwrite("sdf_contact", &TendonFingerSolverConfig::sdf_contact);

    py::class_<SpherePrimitiveContactConfig>(m, "SpherePrimitiveContactConfig")
        .def(py::init<>())
        .def_readwrite("finger_node_index",  &SpherePrimitiveContactConfig::finger_node_index)
        .def_readwrite("finger_node_radius", &SpherePrimitiveContactConfig::finger_node_radius)
        .def_readwrite("sphere_center",      &SpherePrimitiveContactConfig::sphere_center)
        .def_readwrite("sphere_radius",      &SpherePrimitiveContactConfig::sphere_radius)
        .def_readwrite("sphere_pose_cov",    &SpherePrimitiveContactConfig::sphere_pose_cov)
        .def_readwrite("witness",            &SpherePrimitiveContactConfig::witness);

    // Per-finger results. TendonFingerSolver is gone, but TendonHandMarginals
    // holds a vector of these -- this is what every hand script reads.
    py::class_<TendonFingerMarginals>(m, "TendonFingerMarginals")
        .def(py::init<>())
        .def_readwrite("rod", &TendonFingerMarginals::rod)
        .def_readwrite("tendon_config", &TendonFingerMarginals::tendon_config)
        .def_readwrite("external_wrenches", &TendonFingerMarginals::external_wrenches)
        .def_readwrite("tensions", &TendonFingerMarginals::tensions)
        .def_readwrite("J_pose_tensions", &TendonFingerMarginals::J_pose_tensions)
        .def_readwrite("tendon_lengths", &TendonFingerMarginals::tendon_lengths);

    // --- Environment (collision/contact) ---

    // One member of an ellipsoid set (Section 1.2). local_pose is exposed as a
    // 4x4 matrix rather than a Pose3: gtsam::Pose3 is not bound anywhere in this
    // module, and every other pose that crosses this boundary
    // (object_pose_mean, Pose3Gaussian.mean, base_pose) is a Matrix4 already.
    py::class_<gepetto_solvers::EllipsoidPrimitive>(m, "EllipsoidPrimitive")
        .def(py::init<>())
        .def_readwrite("semi_axes", &gepetto_solvers::EllipsoidPrimitive::semi_axes)
        .def_property("local_pose",
            [](const gepetto_solvers::EllipsoidPrimitive& p) {
                return p.local_pose.matrix();
            },
            [](gepetto_solvers::EllipsoidPrimitive& p, const gtsam::Matrix4& m) {
                p.local_pose = gtsam::Pose3(m);
            });

    py::class_<gepetto_solvers::EnvironmentConfig>(m, "EnvironmentConfig")
        .def(py::init<>())
        .def_readwrite("object_pose_mean",     &gepetto_solvers::EnvironmentConfig::object_pose_mean)
        .def_readwrite("object_pose_cov",      &gepetto_solvers::EnvironmentConfig::object_pose_cov)
        .def_readwrite("object_pose_per_step", &gepetto_solvers::EnvironmentConfig::object_pose_per_step)
        .def_readwrite("collision_avoidance",    &gepetto_solvers::EnvironmentConfig::collision_avoidance)
        .def_readwrite("self_collision",         &gepetto_solvers::EnvironmentConfig::self_collision)
        .def_readwrite("collision_sigma",        &gepetto_solvers::EnvironmentConfig::collision_sigma)
        .def_readwrite("collision_node_indices", &gepetto_solvers::EnvironmentConfig::collision_node_indices)
        .def_readwrite("collision_node_radii",   &gepetto_solvers::EnvironmentConfig::collision_node_radii)
        .def_readwrite("collision_node_is_proximal", &gepetto_solvers::EnvironmentConfig::collision_node_is_proximal)
        .def_readwrite("collision_cull_margin",  &gepetto_solvers::EnvironmentConfig::collision_cull_margin)
        .def_readwrite("target_contact_node", &gepetto_solvers::EnvironmentConfig::target_contact_node)
        .def_readwrite("contact_node_radius", &gepetto_solvers::EnvironmentConfig::contact_node_radius)
        .def_readwrite("witness_point_seed",  &gepetto_solvers::EnvironmentConfig::witness_point_seed)
        .def_readwrite("ellipsoid_semi_axes", &gepetto_solvers::EnvironmentConfig::ellipsoid_semi_axes)
        .def_readwrite("ellipsoid_set",       &gepetto_solvers::EnvironmentConfig::ellipsoid_set)
        .def_readwrite("ellipsoid_set_beta",  &gepetto_solvers::EnvironmentConfig::ellipsoid_set_beta)
        .def_readwrite("contact_ellipsoid_subset",
                       &gepetto_solvers::EnvironmentConfig::contact_ellipsoid_subset)
        // --- Tendon-aligned in-plane object contact (Eq 11 / Eq 13) ---
        .def_readwrite("object_contact_in_plane",
                       &gepetto_solvers::EnvironmentConfig::object_contact_in_plane)
        .def_readwrite("contact_plane_centroid",
                       &gepetto_solvers::EnvironmentConfig::contact_plane_centroid)
        .def_readwrite("contact_plane_rho_lo",
                       &gepetto_solvers::EnvironmentConfig::contact_plane_rho_lo)
        .def_readwrite("contact_plane_rho_hi",
                       &gepetto_solvers::EnvironmentConfig::contact_plane_rho_hi)
        .def_readwrite("contact_plane_gap_lo",
                       &gepetto_solvers::EnvironmentConfig::contact_plane_gap_lo)
        .def_readwrite("contact_plane_gap_hi",
                       &gepetto_solvers::EnvironmentConfig::contact_plane_gap_hi)
        .def_readwrite("plane_origin",        &gepetto_solvers::EnvironmentConfig::plane_origin)
        .def_readwrite("plane_normal",        &gepetto_solvers::EnvironmentConfig::plane_normal)
        .def_readwrite("plane_avoidance",     &gepetto_solvers::EnvironmentConfig::plane_avoidance)
        .def_readwrite("table_contact_node",  &gepetto_solvers::EnvironmentConfig::table_contact_node)
        .def_readwrite("table_contact_radius",&gepetto_solvers::EnvironmentConfig::table_contact_radius)
        // --- Section 1.8 phased controller ---
        .def_readwrite("support_contact_node",   &gepetto_solvers::EnvironmentConfig::support_contact_node)
        .def_readwrite("support_contact_radius", &gepetto_solvers::EnvironmentConfig::support_contact_radius)
        .def_readwrite("half_space_enabled",     &gepetto_solvers::EnvironmentConfig::half_space_enabled)
        .def_readwrite("half_space_node",        &gepetto_solvers::EnvironmentConfig::half_space_node)
        .def_readwrite("half_space_split_point", &gepetto_solvers::EnvironmentConfig::half_space_split_point)
        .def_readwrite("half_space_normal",      &gepetto_solvers::EnvironmentConfig::half_space_normal)
        .def_readwrite("half_space_margin",      &gepetto_solvers::EnvironmentConfig::half_space_margin)
        .def_readwrite("object_contact_center_direct",
                       &gepetto_solvers::EnvironmentConfig::object_contact_center_direct)
        .def_readwrite("contact_drop_normal_row",
                       &gepetto_solvers::EnvironmentConfig::contact_drop_normal_row)
        .def_readwrite("witness_target",     &gepetto_solvers::EnvironmentConfig::witness_target)
        .def_readwrite("witness_target_cov", &gepetto_solvers::EnvironmentConfig::witness_target_cov)
        // --- Pre-grasp hand-centering (Section 2.2.1, Eq 2.18-2.19) ---
        .def_readwrite("pregrasp_center_node",
                       &gepetto_solvers::EnvironmentConfig::pregrasp_center_node)
        .def_readwrite("pregrasp_clearance_height",
                       &gepetto_solvers::EnvironmentConfig::pregrasp_clearance_height)
        .def_readwrite("pregrasp_clearance_normal",
                       &gepetto_solvers::EnvironmentConfig::pregrasp_clearance_normal)
        // --- Pre-grasp short-axis alignment ---
        .def_readwrite("pregrasp_align_node",
                       &gepetto_solvers::EnvironmentConfig::pregrasp_align_node)
        .def_readwrite("pregrasp_align_axis",
                       &gepetto_solvers::EnvironmentConfig::pregrasp_align_axis)
        // --- Pre-grasp pinch-centroid centering (hardcoded hand-frame point) ---
        .def_readwrite("pregrasp_centroid_point",
                       &gepetto_solvers::EnvironmentConfig::pregrasp_centroid_point)
        .def_readwrite("pregrasp_centroid_clearance",
                       &gepetto_solvers::EnvironmentConfig::pregrasp_centroid_clearance)
        .def_readwrite("pregrasp_centroid_normal",
                       &gepetto_solvers::EnvironmentConfig::pregrasp_centroid_normal)
        .def("load_sdf", [](gepetto_solvers::EnvironmentConfig& self, const std::string& path) {
            openvdb::initialize();
            openvdb::io::File f(path);
            f.open();
            auto names = f.getGrids();
            if (names->empty()) {
                f.close();
                throw std::runtime_error("VDB file contains no grids: " + path);
            }
            self.sdf_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(names->at(0));
            f.close();
            if (!self.sdf_grid) {
                throw std::runtime_error("First grid in VDB file is not a FloatGrid: " + path);
            }
        }, py::arg("path"));

    // Evaluate the tendon-aligned planar distance (Eq 11 / Eq 13) at one state.
    //
    // A free function rather than a bound factor class: gtsam::Pose3 and the factor
    // hierarchy are not bound in this module, and the only caller (the visualizer's
    // in-plane overlay) wants numbers, not a graph object. Poses cross as Matrix4,
    // the convention every other pose on this boundary already uses.
    //
    // The point of exposing it at all is that the OVERLAY AND THE FACTOR AGREE BY
    // CONSTRUCTION. A python re-derivation of this geometry would be free to drift
    // from the C++ -- and an overlay that draws a plane the solver never used is
    // worse than no overlay.
    m.def("ellipsoid_set_planar_gap",
        [](const gtsam::Matrix4& tip_pose, const gtsam::Matrix4& object_pose,
           const gtsam::Matrix4& wrist_pose, double radius,
           const std::vector<gepetto_solvers::EllipsoidPrimitive>& ellipsoids,
           double beta, const gtsam::Vector3& base_local,
           const gtsam::Vector3& centroid_local,
           double rho_lo, double rho_hi, double gap_lo, double gap_hi) {
            gepetto_solvers::EllipsoidSetPlanarGapFactor factor(
                0, 1, 2, radius, ellipsoids, beta, base_local, centroid_local,
                gtsam::noiseModel::Isotropic::Sigma(1, 1.0),
                rho_lo, rho_hi, gap_lo, gap_hi);
            const auto r = factor.report(gtsam::Pose3(tip_pose),
                                         gtsam::Pose3(object_pose),
                                         gtsam::Pose3(wrist_pose));
            py::dict out;
            out["distance"] = r.d_set;              // the fused in-plane distance
            out["gap"] = radius - r.d_set;          // the factor's residual, c_pen
            out["d3"] = r.d3;                       // per member, 3D
            out["d_planar"] = r.d_planar;           // per member, in-plane
            out["rho"] = r.rho;                     // per member, support ratio
            out["lam"] = r.lambda;                  // per member, 3D blend weight
            out["weight"] = r.weight;               // per member, planar weight
            out["mu"] = r.mu;                       // how well defined the plane is
            out["axis_gap"] = r.axis_gap;           // tip standoff from the axis (m)
            out["normal"] = r.normal;               // n_pull, world; zero if undefined
            return out;
        },
        py::arg("tip_pose"), py::arg("object_pose"), py::arg("wrist_pose"),
        py::arg("radius"), py::arg("ellipsoids"), py::arg("beta"),
        py::arg("base_local"), py::arg("centroid_local"),
        py::arg("rho_lo") = 0.90, py::arg("rho_hi") = 1.00,
        py::arg("gap_lo") = 0.002, py::arg("gap_hi") = 0.010,
        "In-plane (Eq 11 pulling-plane) distance from a fingertip to an ellipsoid "
        "set, with the per-member breakdown. Falls back to the 3D distance where "
        "the plane misses a member or is degenerate.");

}
