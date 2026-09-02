# Adding a hand

Two registrations, on the two sides of the pybind boundary:

1. **A `HandKinematics` in C++**, registered under a `kinematics` name. It
   contributes every factor internal to the mechanism, and answers "which pose
   variable is at this place on the hand".
2. **A `Hand` in Python**, registered under a hand name. It names that
   kinematics, lists the digits, describes the actuation, and carries whatever
   was measured about this particular hand.

They are separate because they vary separately. Two tendon hands of different
morphology are two `Hand`s over one `HandKinematics`; a servo-driven hand is a
new one of each.

You need step 1 only if the mechanism is genuinely new. A second hand built the
same way as `tendon_5f` — rods, tendons, a shared wrist — is step 2 alone.

---

## What the graph builder already does for you

`HandModel::build_graph`
([src/hand/HandGraph.cpp](../src/hand/HandGraph.cpp)) builds the task half of
the problem: object contact, finger–object and finger–finger collision, the
support plane, the opposition half-space, and the three pre-grasp constraints.
All of it is driven by the per-digit `EnvironmentConfig`s on the `HandSpec`, and
none of it mentions a mechanism.

It reaches the hand through exactly four calls:

| call | what it asks |
|---|---|
| `site_pose_key({digit, node})` | the pose variable at this place |
| `site_is_root({digit, node})` | whether that place has no free pose of its own |
| `digit_base_offset(digit)` | this digit's fixed attachment to the wrist |
| `add_kinematics_factors(...)` | everything internal to the mechanism |

`node` keeps the addressing `EnvironmentConfig` already uses
(`target_contact_node`, `collision_node_indices`, `table_contact_node`, …):
`>= 0` counts from the digit's base, `< 0` from its tip. A hand that is not a
chain of rod nodes is free to map those integers onto whatever variables it
owns — which is what lets a mechanism defined as a whole, rather than as digits
combined, still be addressed per digit.

**Constraint ordering is load-bearing.** The Augmented Lagrangian indexes
multipliers by a constraint's position in the graph, so any hard constraint your
kinematics emits must go through the `ConstraintTagger` it is handed, never
straight onto the graph. `tests/core/test_constraint_tags.py` guards this.

---

## 0. First: does your hand need C++ at all?

Two kinematics are registered:

| name | mechanism |
|---|---|
| `tendon` | Cosserat-rod digits driven by tendons |
| `rigid_urdf` | serial 1-DOF chains from a URDF, posed by Pinocchio FK |

**`rigid_urdf` is generic.** Any hand that is a set of serial 1-DOF chains
hanging off a common palm — which is most articulated hands — is a *config* for
it, not new C++. The Allegro hand is exactly that: about 100 lines of table in
[core/hands/allegro/spec.py](../python/gepetto_solvers/core/hands/allegro/spec.py)
naming which joints and frames make up each digit, plus its URDF.

So for a URDF hand, skip to step 2 and write a `Hand` whose `kinematics` is
`"rigid_urdf"` and whose `build_spec` fills a `RigidHandKinematicsConfig`. Step 1
is only for a mechanism neither of the two describes.

What `rigid_urdf` will refuse, loudly rather than silently:

* a joint that is not 1-DOF (a `continuous` URDF joint is nq=2 — give it limits
  to make it `revolute`);
* a digit whose first joint sits below another movable joint, because then its
  mount is not a constant and the `T_0 = T_wrist ∘ offset` invariant breaks;
* a frame or joint name that is not in the URDF.

It also does **not enforce joint limits** yet. They are read from the URDF and
exposed, but nothing constrains against them, so IK can hyperextend — see
[formulation_vs_code.md §6](formulation_vs_code.md#6-joint-limits-are-read-but-not-enforced),
which records that gap along with every other place the code departs from the
written maths.

---

## 1. The C++ kinematics

Implement [`HandKinematics`](../include/gepetto_solvers/hand/HandKinematics.h).
`TendonHandKinematics`
([include](../include/gepetto_solvers/hand/kinematics/tendon/TendonHandKinematics.h),
[src](../src/hand/kinematics/tendon/TendonHandKinematics.cpp)) is the worked
example.

```
include/gepetto_solvers/hand/kinematics/<name>/<Name>Kinematics.h
src/hand/kinematics/<name>/<Name>Kinematics.cpp
```

Carry the config across the boundary by deriving from `HandKinematicsConfig`:

```cpp
struct MyHandKinematicsConfig : gepetto_solvers::HandKinematicsConfig {
    std::vector<MyJointSpec> joints;
};
```

Self-register at static init, and downcast the payload:

```cpp
namespace {
const bool registered = [] {
    gepetto_solvers::register_hand_kinematics(
        "myhand",
        [](const HandKinematicsConfig& config,
           const std::vector<std::string>& digit_names,
           const Pose3& wrist_pose, Key wrist_key)
            -> std::unique_ptr<HandKinematics> {
            const auto* c = dynamic_cast<const MyHandKinematicsConfig*>(&config);
            if (!c)
                throw std::invalid_argument(
                    "the \"myhand\" kinematics needs a MyHandKinematicsConfig");
            return std::make_unique<MyHandKinematics>(
                *c, digit_names, wrist_pose, wrist_key);
        });
    return true;
}();
}
```

Then:

* add both files to `GEPETTO_SOLVERS_SOURCES` in
  [CMakeLists.txt](../CMakeLists.txt) — the list is explicit on purpose, so a
  missing file is a configure-time error rather than a link-time one, and
  **listing your `.cpp` there is what makes the kinematics loadable at all**;
* bind your config struct in [src/bindings/bind_hand.cpp](../src/bindings/bind_hand.cpp),
  alongside `TendonHandKinematicsConfig`, plus a `make_*_hand_spec` helper if
  the Python side needs one;
* check it registered: `gepetto_solvers.registered_hand_kinematics()`.

### The wrist

Every hand shares one floating wrist variable, and `HandModel` owns its prior.
On the tendon hand each digit's node 0 is *not* a variable: it is the
composition `T_0 = T_wrist ∘ T_offset`, which is what expresses the joint prior
over the wrist and the digit bases as one Gaussian times a deterministic SE(3)
composition per digit, rather than as a soft rigidity penalty with a null space
in it. Your mechanism may instead own the wrist outright and read it directly.
Either way that choice lives inside the kinematics — see
`TendonHandKinematics::insert_from_state`, which has to invert the offset to
recover a wrist no digit carries.

### The state you fill

`HandState` ([HandState.h](../include/gepetto_solvers/hand/HandState.h)) is the
transport for a finished solve, and it is mechanism-neutral: per digit, a list of
`SiteState` (pose, plus continuum stress and wrench that a rigid hand leaves
zero), the `actuation` vector, an optional `displacement` readout, the
`collision_sites` indices, and `extras`.

`extras` is where anything only YOUR mechanism has goes — a
`DigitExtras` subclass, the same pattern as `HandKinematicsConfig` on the input
side. `TendonDigitExtras` carries the tendon routing, the per-disc external
wrenches and the tension Jacobian; `RigidHandKinematics` leaves it null. A
reader that needs it checks for null first, which is how the viser overlays draw
tendon routes on one hand and nothing on the other.

`HandState` also carries `wrist_pose`, so each kinematics answers "where did the
wrist end up" for itself rather than callers inverting a mounting offset.

---

## 2. The Python hand

Implement the [`Hand`](../python/gepetto_solvers/core/hands/base.py) protocol.
`TendonHand5F`
([hand.py](../python/gepetto_solvers/core/hands/tendon_5f/hand.py)) is the
worked example; it is about 130 lines, most of them the measured tables.

```
python/gepetto_solvers/core/hands/<name>/
    __init__.py     exports the Hand class
    hand.py         the Hand itself
    ...             whatever it is built from
```

The protocol in full:

| member | what it is |
|---|---|
| `name` | registry key for this hand |
| `kinematics` | registry key of the C++ kinematics to load |
| `digit_names` | digit order — every per-digit list in the solvers uses it |
| `tip_radii` | contact-sphere radius per digit |
| `actuation` | an `Actuation`: how many actuators, which are driven |
| `opposing_digit` | the digit opposing the rest, or `None` |
| `digit_configs()` | `[(name, config)]`, **freshly built each call** |
| `contact_node(digit)` | the site task constraints contact with |
| `collision_sites(digit)` | `(node indices, is_proximal flags)` |
| `pinch_pose(mask)` | measured pinch geometry, or `None` |
| `features` | a `frozenset` from `hands.FEATURES` — what this hand supports |
| `default_pose()` | `(wrist 4x4, actuation means)` — where this hand starts |
| `actuation_means(params)` | one mean vector per digit: q_S, or the tensions |
| `build_spec(configs, params)` | the C++ `HandSpec` |
| `opposing_index` | index of `opposing_digit`, or `-1` |

Optional, read where they are relevant: `hardware` (a `HardwareMap`), `motion`
(a `MotionProfile`), `default_contact_digits`, `joint_limits()`.

Three of those are worth a sentence each, because they are what stop the
workbench and the solvers assuming your hand is the tendon one:

* **`features`** gates whole GUI panels. Declare only what you have; a control
  for something you lack is made ABSENT, not greyed out. `AllegroHand.features`
  is empty.
* **`default_pose()`** is where your hand starts. There is no shared default that
  is right for two hands — the tendon hand's measured hover aims Allegro nowhere
  near the object, because their fingers extend along different axes.
* **`actuation_means(params)`** turns the params into the vector your actuation
  variable takes. `params.flexor_tensions` is one *scalar* per digit and cannot
  command four independent joints; a joint-space hand reads
  `params.joint_targets` instead.

Register in [core/hands/\_\_init\_\_.py](../python/gepetto_solvers/core/hands/__init__.py):

```python
register_hand(MyHand.name, MyHand)
```

Three things that bite:

* **`digit_configs()` must build fresh every call.** The `attach_*` environment
  family mutates those configs in place, so a shared list leaks one solve's
  constraints into the next.
* **Annotate `opposing_digit: str | None`.** A bare `= "thumb"` infers `str`,
  and a protocol's attributes are invariant, so the class stops satisfying its
  own interface. mypy catches this.
* **`pinch_pose` returning `None` is a real answer**, not a failure. A caller
  must handle it rather than substituting a default.

### Sizing the params

`HandSolveParams.flexor_tensions` and `.contact_fingers` are positional, one per
digit, and default to the *default* hand's digit count. Posing a hand with a
different count means setting both:

```python
params = HandSolveParams()
params.flexor_tensions = [1.0] * len(hand.digit_names)
params.contact_fingers = [True] * len(hand.digit_names)
solver = HandIKSolver(params, hand)
```

Pass the hand to the solver directly, or name it in `params.hand` where the
choice has to ride along in a serialized or preset params object.

---

## 3. Check it

```bash
pytest tests/core/test_hand_interface.py    # the seam itself
pytest tests/projects/test_viz_hands.py     # the workbench, per registered hand
pytest -m "not slow"
pytest -m slow                              # golden sums must NOT move
python scripts/viz_interactive.py --hand <name>
```

Three suites are worth knowing about:

* `tests/core/test_hand_interface.py` registers a two-digit stub hand with four
  actuators and no opposing digit, and drives the solver stack with it. Adding a
  case there is the cheapest way to prove the solvers size themselves to your
  hand rather than to the built-in one.
* `tests/projects/test_viz_hands.py` builds the workbench for **every registered
  hand**, so your hand is covered by it the moment you register. It also asserts
  `app.fk_solver.hand is app.hand` — the app once built its solvers without
  passing its hand and silently posed the default one, rendering a perfectly
  good picture of the wrong robot.
* `tests/core/test_allegro_hand.py` is the worked example of testing a hand end
  to end, including the chain-integrity checks (§below) that catch a mis-listed
  site.

**Check your site list against the mechanism.** Leaving a link out does not
merely coarsen the picture — it merges two joints. On Allegro, omitting the
distal link made two sliders appear to drive the same thing and drew one 65 mm
bar where the hand has 38 mm and 27 mm about a joint. Assert that the number of
sites matches the DOF count, that each joint moves a strictly smaller set of
sites than the one above it, and that the drawn segments match the URDF's link
lengths.

If the golden sums in `tests/test_golden_solves.py` move, something changed the
graph for the *existing* hand — that is a bug, not a rebaseline.
