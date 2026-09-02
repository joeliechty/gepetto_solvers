#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "gepetto_solvers/digits/rigid/PinocchioFKFactor.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>

namespace py = pybind11;
using gepetto_solvers::PinocchioFKFactor;
using gepetto_solvers::RigidChainModel;

// Poses cross this boundary as 4x4 matrices, matching how every other pose in
// this module is exposed (Pose3Gaussian::mean and HandState::wrist_pose are both
// gtsam::Matrix4). Nothing here binds a GTSAM type.


void bind_digits_rigid(py::module& m) {
    py::class_<RigidChainModel, std::shared_ptr<RigidChainModel>>(
        m, "RigidChainModel",
        "A URDF-described rigid-body model, shared by every FK factor built on "
        "it. Immutable; each factor keeps its own scratch data.")
        .def_static("from_urdf_xml", &RigidChainModel::from_urdf_xml,
                    py::arg("urdf_xml"),
                    "Build from URDF TEXT, so a caller can hold the robot "
                    "description however it likes and a test can write one "
                    "inline. Reads no mesh files -- kinematics only.")
        .def_static("from_urdf_file", &RigidChainModel::from_urdf_file,
                    py::arg("path"))
        .def("frame_id", &RigidChainModel::frame_id, py::arg("name"),
             "Frame index by name; raises if absent, because a mistyped frame "
             "would otherwise read as a pose at the origin.")
        .def("joint_indices", &RigidChainModel::joint_indices, py::arg("name"),
             "(configuration index, velocity index) for a 1-DOF joint. They "
             "differ for joints parameterized with more coordinates than "
             "degrees of freedom; raises for anything that is not 1-DOF.")
        .def_property_readonly("nq", &RigidChainModel::nq)
        .def_property_readonly("nv", &RigidChainModel::nv)
        .def_property_readonly("lower_position_limits",
                               &RigidChainModel::lower_position_limits)
        .def_property_readonly("upper_position_limits",
                               &RigidChainModel::upper_position_limits);

    py::class_<PinocchioFKFactor, std::shared_ptr<PinocchioFKFactor>>(
        m, "PinocchioFKFactor",
        "The kinematics likelihood p(T_i | T_w, q) for one frame, as a ternary "
        "factor over (wrist, joints, site).\n\n"
        "`sigma` is the diagonal of the per-frame relaxation covariance "
        "Sigma_fk,i, ordered [rot(3), pos(3)] to match GTSAM's Pose3 tangent. "
        "As it tightens the likelihood approaches a hard kinematic constraint.")
        .def(py::init([](gtsam::Key wrist_key, gtsam::Key joint_key,
                         gtsam::Key site_key,
                         std::shared_ptr<RigidChainModel> chain,
                         int frame_id, std::vector<int> q_index,
                         std::vector<int> v_index, const gtsam::Vector6& sigma) {
                 return std::make_shared<PinocchioFKFactor>(
                     wrist_key, joint_key, site_key, std::move(chain), frame_id,
                     std::move(q_index), std::move(v_index),
                     gtsam::noiseModel::Diagonal::Sigmas(sigma));
             }),
             py::arg("wrist_key"), py::arg("joint_key"), py::arg("site_key"),
             py::arg("chain"), py::arg("frame_id"),
             py::arg("q_index"), py::arg("v_index"), py::arg("sigma"))
        .def("predict",
             [](const PinocchioFKFactor& f, const gtsam::Matrix4& wrist,
                const gtsam::Vector& q) {
                 return f.predict(gtsam::Pose3(wrist), q).matrix();
             },
             py::arg("wrist"), py::arg("q"),
             "f_fk,i(T_w, q) as a 4x4 -- the frame placement this factor "
             "predicts. Seeding a site here starts the solve on the kinematics "
             "manifold rather than somewhere it must be pulled back from.")
        .def("error_and_jacobians",
             [](const PinocchioFKFactor& f, const gtsam::Matrix4& wrist,
                const gtsam::Vector& q, const gtsam::Matrix4& site) {
                 gtsam::Matrix Hw, Hq, Hs;
                 gtsam::Vector e = f.evaluateError(
                     gtsam::Pose3(wrist), q, gtsam::Pose3(site), &Hw, &Hq, &Hs);
                 return py::make_tuple(e, Hw, Hq, Hs);
             },
             py::arg("wrist"), py::arg("q"), py::arg("site"),
             "(error, H_wrist, H_q, H_site) at the given values.")
        .def("error",
             [](const PinocchioFKFactor& f, const gtsam::Matrix4& wrist,
                const gtsam::Vector& q, const gtsam::Matrix4& site) {
                 return f.evaluateError(gtsam::Pose3(wrist), q,
                                        gtsam::Pose3(site),
                                        nullptr, nullptr, nullptr);
             },
             py::arg("wrist"), py::arg("q"), py::arg("site"),
             "The 6-vector residual, with no Jacobians computed.");

    // GTSAM's own SE(3) retraction, exposed so a numerical derivative taken in
    // Python perturbs poses exactly the way the analytic Jacobians assume.
    //
    // This is the whole point of binding it rather than writing an exp map in
    // the test: a hand-rolled retraction that differs from GTSAM's -- in the
    // translation coupling, or in the [rot; pos] ordering -- would make the
    // comparison meaningless in either direction, passing a wrong Jacobian or
    // failing a right one.
    m.def("pose3_retract",
          [](const gtsam::Matrix4& T, const gtsam::Vector6& xi) {
              return gtsam::Pose3(T).retract(xi).matrix();
          },
          py::arg("pose"), py::arg("xi"),
          "GTSAM's Pose3 retraction: pose (+) xi, with xi = [rot(3), pos(3)].");

    m.def("pose3_local",
          [](const gtsam::Matrix4& a, const gtsam::Matrix4& b) {
              return gtsam::Pose3(a).localCoordinates(gtsam::Pose3(b));
          },
          py::arg("a"), py::arg("b"),
          "GTSAM's Pose3 Local(a, b) = Log(a^-1 b): the on-manifold difference "
          "b (-) a, as [rot(3), pos(3)].");
}
