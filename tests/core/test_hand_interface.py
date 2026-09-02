"""The Hand seam: that a hand OTHER than the built-in one drives the solvers.

The point of these tests is that they do not use ``TendonHand5F``. A stub hand
-- two digits, four actuators, no opposing digit, a pinch table that says
nothing -- is registered and handed to the solver stack, and the assertions are
about what the stack then does with it: which digit list it sizes itself to,
which actuator it drives, which spec it hands the C++ side.

That is the difference between an abstraction that is real and one that merely
looks like it. Every test in ``test_hand.py`` would still pass if the solvers
had ``FINGER_NAMES`` and ``FLEXOR_IDX`` baked in, because the built-in hand
agrees with those constants; nothing here would.

The stub deliberately does NOT solve. It carries the built-in hand's digit
configs (a stub with no kinematics could not build a graph at all), so what is
exercised is the interface, the config assembly and the spec crossing the pybind
boundary -- not a second mechanism, which would need a second C++ kinematics.
"""

from __future__ import annotations

import numpy as np
import pytest

import gepetto_solvers
from _pkg import config, hands, solvers

# ---------------------------------------------------------------------------
# A hand that is not the tendon hand.
# ---------------------------------------------------------------------------

class StubHand:
    """Two digits, four actuators, two of them driven, no opposition.

    Every number here is chosen to DISAGREE with the built-in hand, so a solver
    that fell back to a baked-in constant would produce something visibly wrong
    rather than something accidentally right.
    """

    name = "stub2"
    kinematics = "tendon"
    opposing_digit = None
    default_contact_digits = ("alpha",)

    actuation = hands.Actuation(
        n=4,
        names=("a", "b", "c", "d"),
        drive_indices=(1, 3),
    )
    motion = hands.MotionProfile(close_steps=3, lift_steps=2, lift_height_m=0.05)

    #: A deliberately sparse feature set: this hand has tendons and a single
    #: driven actuator per digit but nothing else, so a panel gated on
    #: "calibration" or "pinch_table" must switch itself off for it.
    features = frozenset({"tendons", "single_drive"})

    def __init__(self):
        # Borrowed geometry: the seam under test is the interface, not a second
        # mechanism. Renamed so nothing can match on the built-in digit names.
        borrowed = config.TendonHand5F(config.DEFAULT_HAND_DIMENSIONS)
        self._source = borrowed
        self.digit_names = ["alpha", "beta"]
        self.tip_radii = list(borrowed.tip_radii[:2])
        self.hardware = hands.HardwareMap(
            actuator_names={"alpha": "alpha_drive", "beta": "beta_drive"},
            open_passive=0.25,
            open_drive={"alpha": 0.5, "beta": 0.5},
        )

    def digit_configs(self):
        return [(name, cfg)
                for name, (_, cfg) in zip(self.digit_names,
                                          self._source.digit_configs())]

    def contact_node(self, digit):
        return self._source.contact_node(digit)

    def collision_sites(self, digit):
        return self._source.collision_sites(digit)

    def pinch_pose(self, mask):
        """No measured pinch geometry for this hand -- which is the honest
        answer, and the one the pre-grasp centroid constraint must handle."""
        return None

    def actuation_means(self, params):
        """Passive background everywhere, the commanded value at BOTH driven
        indices -- which is what a hand driving more than one actuator per digit
        looks like, and what a caller assuming a single one gets wrong."""
        means = []
        for i in range(len(self.digit_names)):
            mean = np.full(self.actuation.n, params.passive_tension)
            self.actuation.set_drive(mean, params.flexor_tensions[i])
            means.append(mean)
        return means

    @property
    def opposing_index(self):
        return hands.opposing_index_of(self.digit_names, self.opposing_digit)

    def build_spec(self, configs, params=None):
        return gepetto_solvers.make_tendon_hand_spec(
            configs, opposing_digit=self.opposing_index)


@pytest.fixture
def stub_hand():
    return StubHand()


# ---------------------------------------------------------------------------
# The registry.
# ---------------------------------------------------------------------------

def test_the_built_in_hand_is_registered():
    assert "tendon_5f" in hands.registered_hands()
    assert hands.get_hand("tendon_5f").name == "tendon_5f"


def test_get_hand_defaults_to_the_named_default():
    assert hands.get_hand().name == hands.DEFAULT_HAND


def test_an_unknown_hand_raises_and_names_what_is_registered():
    with pytest.raises(KeyError) as excinfo:
        hands.get_hand("no_such_hand")
    assert "tendon_5f" in str(excinfo.value)


def test_a_registered_hand_is_built_fresh_each_time():
    """Two get_hand() calls must not share solver configs: the attach_* family
    mutates them in place, so a shared instance would leak one solve's
    constraints into the next."""
    a, b = hands.get_hand(), hands.get_hand()
    assert a is not b
    assert a.digit_configs()[0][1] is not b.digit_configs()[0][1]


def test_a_stub_hand_can_be_registered_and_fetched(stub_hand):
    hands.register_hand("stub2", StubHand)
    try:
        assert hands.get_hand("stub2").digit_names == ["alpha", "beta"]
    finally:
        hands.registry._FACTORIES.pop("stub2", None)


# ---------------------------------------------------------------------------
# The solver sizes itself to the hand it is given.
# ---------------------------------------------------------------------------

def test_the_solver_takes_its_digits_from_the_hand(stub_hand):
    params = solvers.HandSolveParams()
    params.flexor_tensions = [1.0, 1.0]
    params.contact_fingers = [True, True]
    base = solvers.HandSolverBase(params, stub_hand)

    assert base.hand is stub_hand
    assert base.finger_names == ["alpha", "beta"]
    assert len(base.configs) == 2
    assert base.tip_radii == stub_hand.tip_radii


def test_the_solver_prefers_an_explicit_hand_over_the_params_name(stub_hand):
    """``params.hand`` names the DEFAULT; an explicit hand must win, or a caller
    holding a Hand object could not use it without also editing its params."""
    params = solvers.HandSolveParams()
    params.hand = "tendon_5f"
    params.flexor_tensions = [1.0, 1.0]
    params.contact_fingers = [True, True]
    base = solvers.HandSolverBase(params, stub_hand)
    assert base.finger_names == ["alpha", "beta"]


def test_actuation_priors_are_sized_and_driven_by_the_hand(stub_hand):
    """The prior means are the thing a baked-in FLEXOR_IDX would get wrong: this
    hand has four actuators and drives numbers 1 and 3."""
    params = solvers.HandSolveParams()
    params.flexor_tensions = [2.0, 3.0]
    params.contact_fingers = [True, True]
    params.passive_tension = 0.5
    base = solvers.HandSolverBase(params, stub_hand)

    priors = base._tension_priors(np.eye(4))
    assert len(priors) == 2
    np.testing.assert_allclose(priors[0].mean, [0.5, 2.0, 0.5, 2.0])
    np.testing.assert_allclose(priors[1].mean, [0.5, 3.0, 0.5, 3.0])


def test_the_prior_covariance_loosens_exactly_the_driven_actuators(stub_hand):
    params = solvers.HandSolveParams()
    params.flexor_tensions = [1.0, 1.0]
    params.contact_fingers = [True, True]
    params.passive_tension_sigma = 1e-3
    params.flexor_tension_sigma = 1e-1
    base = solvers.HandSolverBase(params, stub_hand)

    cov = base._flexor_tension_cov()
    assert cov.shape == (4, 4)
    np.testing.assert_allclose(np.diag(cov), [1e-6, 1e-2, 1e-6, 1e-2])


def test_a_hand_with_no_opposing_digit_reports_minus_one(stub_hand):
    assert stub_hand.opposing_index == -1


def test_the_built_in_hand_names_its_opposing_digit():
    hand = hands.get_hand()
    assert hand.opposing_digit == "thumb"
    assert hand.digit_names[hand.opposing_index] == "thumb"


def test_naming_an_opposing_digit_the_hand_lacks_is_an_error():
    with pytest.raises(ValueError, match="not one of this hand's digits"):
        hands.opposing_index_of(["alpha", "beta"], "thumb")


# ---------------------------------------------------------------------------
# The spec that crosses into C++.
# ---------------------------------------------------------------------------

def test_the_spec_carries_the_hand_s_digits_and_opposition(stub_hand):
    params = solvers.HandSolveParams()
    params.flexor_tensions = [1.0, 1.0]
    params.contact_fingers = [True, True]
    base = solvers.HandSolverBase(params, stub_hand)

    spec = base._hand_spec()
    assert spec.kinematics == "tendon"
    assert list(spec.digit_names) == ["alpha", "beta"]
    assert spec.opposing_digit == -1
    assert len(spec.env) == 2
    spec.validate()


def test_the_spec_splits_the_task_env_off_from_the_kinematics():
    """The whole point of HandSpec: the env the environment layer wrote is on
    the TASK half, where the graph builder reads it, and the rod/tendon geometry
    is an opaque payload the graph builder never sees."""
    hand = hands.get_hand()
    params = solvers.HandSolveParams()
    base = solvers.HandIKSolver(params, hand)
    base._attach_environment()
    spec = base._hand_spec()

    assert len(spec.env) == len(hand.digit_names)
    # attach_contact ran, so every digit has an env and the contacting ones
    # carry a target node.
    assert all(e is not None for e in spec.env)
    assert any(e.target_contact_node is not None for e in spec.env)
    # ...and the kinematics payload is the tendon fingers, untouched.
    assert isinstance(spec.kinematics_config,
                      gepetto_solvers.TendonHandKinematicsConfig)
    assert len(spec.kinematics_config.fingers) == len(hand.digit_names)


def test_the_spec_declares_the_opposing_digit_as_an_index():
    """The C++ graph builder used to match the literal string "thumb". It now
    reads this index, so a hand states its own opposition."""
    hand = hands.get_hand()
    spec = hand.build_spec(hand.digit_configs())
    assert spec.opposing_digit == hand.digit_names.index("thumb")


def test_a_malformed_spec_is_rejected_rather_than_half_built():
    spec = gepetto_solvers.HandSpec()
    spec.kinematics = "tendon"
    spec.digit_names = ["only_one"]
    with pytest.raises(ValueError, match="kinematics_config is null"):
        spec.validate()


# ---------------------------------------------------------------------------
# The C++ kinematics registry.
# ---------------------------------------------------------------------------

def test_the_tendon_kinematics_is_registered_in_the_binding():
    assert "tendon" in gepetto_solvers.registered_hand_kinematics()


def test_every_hand_names_a_kinematics_the_binding_can_load():
    """A hand naming an unregistered kinematics fails at solve time, deep in
    C++. Checking it here says so at the point the hand is defined."""
    known = set(gepetto_solvers.registered_hand_kinematics())
    for name in hands.registered_hands():
        assert hands.get_hand(name).kinematics in known, name


def test_an_unknown_kinematics_names_what_is_registered():
    hand = hands.get_hand()
    spec = hand.build_spec(hand.digit_configs())
    spec.kinematics = "no_such_kinematics"
    cfg = gepetto_solvers.HandSolverConfig()
    with pytest.raises(ValueError) as excinfo:
        gepetto_solvers.HandSolver(spec, cfg)
    message = str(excinfo.value)
    assert "no_such_kinematics" in message
    assert '"tendon"' in message
