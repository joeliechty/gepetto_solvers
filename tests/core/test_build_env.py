"""Guards on the BUILT ENVIRONMENTS: what the extension linked against at runtime.

Everything here is a fact about the conda envs that ``conda_setup_*.sh`` produce,
not about any algorithm. It exists because the failure mode these pins prevent is
not a test failure -- it is a solve that loads the wrong ``libboost_serialization``
and segfaults hours later, or a ``conda install`` that silently swaps a transitive
library out from under an extension nobody rebuilt.

**Every env the scripts declare is checked, not just the running one.** The three
setup scripts build ``gepetto_py10`` / ``gepetto_py11`` / ``gepetto_py12``, and they
do not pin identically -- py10 and py12 pin the linux-64 ``libpinocchio`` build
``h2844b27_0`` while the mac script pins ``h9a60d09_0``, and py12 resolves a
different OpenVDB (13.0 vs 12.1) because conda-forge stopped building 13 for py310.
A regression that only bites one interpreter is exactly what a suite run from
whichever env happens to be active would miss. Env names are read out of the
scripts themselves (:func:`_declared_env_names`), so adding a fourth script needs
no edit here. Envs that are absent, or present but not yet built, are skipped --
this must stay runnable on a machine that set up only one.

The pins being guarded (see the comment block above each ``conda install`` line):

* **A single Boost.** GTSAM links Boost by unversioned soname, so a prefix holding
  both 1.85 and 1.88 links and loads and then corrupts memory across the
  serialization boundary. ``libboost-devel`` replaced the frozen ``boost``
  metapackage precisely to keep one version in the prefix.
* **Eigen 3.4, matching GTSAM.** GTSAM bakes ``GTSAM_EIGEN_VERSION_WORLD/MAJOR``
  into ``gtsam/config.h`` and static-asserts them, so Pinocchio must be the build
  compiled against the same Eigen major.minor. That assert is compile-time and
  cannot be read from Python -- but the *consequence* can be:
  :func:`test_pinocchio_and_extension_coexist` loads both in one process, which is
  what the pin protects and what a user actually does.
* **Everything inside its own prefix.** ``CMakeLists.txt`` resolves GTSAM and
  OpenVDB relative to the Python prefix, so a stray system copy on the loader path
  is a different library than the one it compiled against.

``tests/core/test_pinocchio_env.py`` is the neighbouring guard on Pinocchio's own
API (the [linear; angular] ordering); this file is about linkage.

Marked ``slow``: each env is probed in a subprocess that imports the full stack.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[2]

# Run inside each env's own interpreter and report what it sees. Kept to the stdlib
# plus the things under test so a half-built env reports its state instead of dying.
PROBE = r'''
import json, sys, sysconfig
out = {"prefix": sys.prefix, "python": ".".join(map(str, sys.version_info[:3])),
       "ext_suffix": sysconfig.get_config_var("EXT_SUFFIX")}
try:
    import gepetto_solvers._gepetto_solvers as ext
    import gepetto_solvers
except Exception as exc:
    out["built"] = False
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
    print(json.dumps(out)); raise SystemExit(0)

out["built"] = True
out["ext_path"] = ext.__file__
symbols = [n for n in dir(ext) if not n.startswith("_")]
out["symbols"] = symbols
out["not_reexported"] = [n for n in symbols if not hasattr(gepetto_solvers, n)]

mods = {}
for name in ("numpy", "scipy", "pinocchio", "trimesh", "pyvista", "viser"):
    try:
        mods[name] = getattr(__import__(name), "__version__", "unknown")
    except Exception as exc:
        mods[name] = "FAIL: %s" % type(exc).__name__
out["modules"] = mods

# Pinocchio and the extension in one process: the Eigen-ABI pin, observed.
try:
    import pinocchio as pin
    model = pin.buildSampleModelHumanoid()
    data = model.createData()
    pin.forwardKinematics(model, data, pin.neutral(model))
    gepetto_solvers.CosseratRodDynamicsConfig()
    out["coexist"] = {"ok": True, "nq": model.nq}
except Exception as exc:
    out["coexist"] = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}

print(json.dumps(out))
'''


def _declared_envs() -> dict[str, str]:
    """env name -> declared python version, read out of the setup scripts.

    Both halves come from the same ``conda create`` line, so neither the list of
    envs nor the version each is supposed to hold can drift from the scripts that
    build them. Nothing here assumes the ``pyNN`` naming means anything.
    """
    declared: dict[str, str] = {}
    for script in sorted(REPO_ROOT.glob("conda_setup_py*.sh")):
        for match in re.finditer(
            r"conda create -n ([A-Za-z0-9_]+)[^\n]*?python=([\d.]+)", script.read_text()
        ):
            declared.setdefault(match.group(1), match.group(2))
    return declared


def _conda_env_prefixes() -> dict[str, Path]:
    """name -> prefix for every conda env on this machine.

    Reads ``conda info --envs``; falls back to the running interpreter's sibling
    envs directory when conda is not on PATH (a bare ``pytest`` from an activated
    env, which is the common case).
    """
    found: dict[str, Path] = {}
    try:
        out = subprocess.run(
            ["conda", "info", "--json"], capture_output=True, text=True, timeout=60
        )
        if out.returncode == 0:
            for raw in json.loads(out.stdout).get("envs", []):
                found[Path(raw).name] = Path(raw)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass

    envs_dir = Path(sys.prefix).resolve().parent
    if envs_dir.name == "envs":
        for child in envs_dir.iterdir():
            if child.is_dir():
                found.setdefault(child.name, child)
    return found


def _discover() -> list[tuple[str, Path]]:
    prefixes = _conda_env_prefixes()
    return [(n, prefixes[n]) for n in DECLARED if n in prefixes]


DECLARED = _declared_envs()
ENVS = _discover()

if not ENVS:
    pytest.skip(
        "none of the conda_setup_*.sh envs exist here; run one of the scripts",
        allow_module_level=True,
    )


@pytest.fixture(scope="session")
def probes() -> dict[str, dict]:
    """Probe every discovered env once, in its own interpreter."""
    results = {}
    for name, prefix in ENVS:
        python = prefix / "bin" / "python"
        if not python.exists():  # Windows layout; no setup script targets it
            continue
        proc = subprocess.run(
            [str(python), "-c", PROBE], capture_output=True, text=True, timeout=600
        )
        line = proc.stdout.strip().splitlines()
        if proc.returncode != 0 or not line:
            pytest.fail(f"probe of env '{name}' failed:\n{proc.stderr[-2000:]}")
        results[name] = json.loads(line[-1])
    return results


@pytest.fixture(params=[n for n, _ in ENVS], ids=[n for n, _ in ENVS])
def env_any(request, probes) -> dict:
    """One discovered env, built or not.

    For the checks that are meaningful before ``pip install`` has run -- an env
    holding the wrong python is worth reporting whether or not the extension made
    it in, and reporting it only once the build succeeds would be backwards.
    """
    name = request.param
    if name not in probes:
        pytest.skip(f"env '{name}' has no bin/python")
    probe = dict(probes[name], name=name)
    return probe


@pytest.fixture(params=[n for n, _ in ENVS], ids=[n for n, _ in ENVS])
def env(request, probes) -> dict:
    """One built env's probe result. Skips an env that exists but is not built."""
    name = request.param
    if name not in probes:
        pytest.skip(f"env '{name}' has no bin/python")
    probe = probes[name]
    if not probe["built"]:
        pytest.skip(
            f"env '{name}' exists but gepetto_solvers is not built into it "
            f"({probe['error']}); run conda_setup for it"
        )
    probe["name"] = name
    return probe


def _linked_libraries(binary: Path) -> dict[str, Path]:
    """soname -> resolved path for everything the loader pulls in for *binary*.

    ``ldd`` on Linux, ``otool -L`` on macOS. Entries the loader fails to resolve are
    dropped here; :func:`test_no_unresolved_libraries` checks for those off the raw
    output, because an unresolved entry is precisely what must not be filtered away
    silently.
    """
    if sys.platform == "darwin":
        out = subprocess.run(
            ["otool", "-L", str(binary)], capture_output=True, text=True, check=True
        ).stdout
        return {
            Path(p).name: Path(p)
            for p in (ln.strip().split(" (")[0] for ln in out.splitlines()[1:])
            if p
        }

    out = subprocess.run(
        ["ldd", str(binary)], capture_output=True, text=True, check=True
    ).stdout
    found: dict[str, Path] = {}
    for line in out.splitlines():
        if "=>" not in line:
            continue
        soname, _, target = line.strip().partition("=>")
        target = target.strip().split(" (")[0].strip()
        if target and target != "not found":
            found[soname.strip()] = Path(target)
    return found


needs_loader_tool = pytest.mark.skipif(
    sys.platform not in ("linux", "darwin"),
    reason="needs ldd (Linux) or otool (macOS) to inspect linkage",
)


# ---------------------------------------------------------------------------
# Sanity on the discovery itself
# ---------------------------------------------------------------------------


def test_every_setup_script_declares_an_env():
    """Guards the parametrization: a renamed script must not silently cover nothing.

    Without this, a typo in the regex above turns the whole file into a no-op that
    reports as passing.
    """
    scripts = sorted(REPO_ROOT.glob("conda_setup_py*.sh"))
    assert scripts, "no conda_setup_py*.sh found at the repo root"
    assert len(DECLARED) == len(scripts), (
        f"{len(scripts)} setup scripts but {len(DECLARED)} env names parsed: "
        f"{DECLARED}"
    )


# ---------------------------------------------------------------------------
# The extension itself
# ---------------------------------------------------------------------------


def test_extension_is_inside_its_own_prefix(env):
    """The .so belongs to the env that would load it, not another one's.

    Without this, every assertion below could describe an extension that env's
    python would never load -- e.g. one reached through a stray PYTHONPATH entry
    pointing at a second conda env, which is easy to do with three of them around.
    """
    prefix = Path(env["prefix"]).resolve()
    assert prefix in Path(env["ext_path"]).resolve().parents, (
        f"[{env['name']}] extension at {env['ext_path']} is outside its prefix "
        f"{prefix}; the env is layered over another install"
    )


def test_extension_abi_tag_matches_its_interpreter(env):
    """A py3.10 .so shadowed by a py3.12 build would import-error late, not here."""
    assert Path(env["ext_path"]).name.endswith(env["ext_suffix"]), (
        f"[{env['name']}] {Path(env['ext_path']).name} does not carry that "
        f"interpreter's ABI tag {env['ext_suffix']}"
    )


def test_env_python_matches_the_scripts_name(env_any):
    """An env must hold the python its own setup script asked for.

    A hand-edited env that drifted from its script is a trap for anyone reading
    the pin comments, since the pins differ per interpreter -- py12 resolves a
    different OpenVDB than py10, for one.
    """
    expected = DECLARED[env_any["name"]]
    assert env_any["python"].startswith(expected + "."), (
        f"[{env_any['name']}] holds python {env_any['python']}, but its setup "
        f"script creates it with python={expected}"
    )


def test_package_reexports_the_extensions_symbols(env):
    """``gepetto_solvers`` is the façade; the solvers must reach through it.

    Asserts the seam rather than a symbol list, so adding a factor does not edit
    this test. The two entry points are named because the application layer is
    built on them.
    """
    assert env["symbols"], f"[{env['name']}] extension exposes no public symbols"
    assert not env["not_reexported"], (
        f"[{env['name']}] not re-exported by gepetto_solvers: "
        f"{sorted(env['not_reexported'])}"
    )
    for entry_point in ("CosseratRodDynamicsSolver", "HandSolver"):
        assert entry_point in env["symbols"], (
            f"[{env['name']}] {entry_point} missing; the extension built but is "
            "incomplete"
        )


# ---------------------------------------------------------------------------
# Linkage
# ---------------------------------------------------------------------------


@needs_loader_tool
def test_no_unresolved_libraries(env):
    """A 'not found' here is a broken install that imports fine until it doesn't.

    This is the guard on the ``activate.d/env_vars.sh`` LD_LIBRARY_PATH hook each
    setup script writes: drop it and libgtsam.so stops resolving.
    """
    if sys.platform == "darwin":
        pytest.skip("otool lists install names, not resolution; ldd-only check")

    out = subprocess.run(
        ["ldd", env["ext_path"]], capture_output=True, text=True, check=True
    ).stdout
    unresolved = [ln.strip() for ln in out.splitlines() if "not found" in ln]
    assert not unresolved, f"[{env['name']}] unresolved libraries: {unresolved}"


@needs_loader_tool
@pytest.mark.parametrize(
    "stem", ["libgtsam", "libopenvdb", "libtbb", "libboost_serialization"]
)
def test_core_dependency_comes_from_its_own_prefix(env, stem):
    """GTSAM/OpenVDB/TBB/Boost must be that env's copy, not the system's.

    CMakeLists.txt finds these through the interpreter prefix, so linking one and
    loading another (``/usr/lib/x86_64-linux-gnu/libboost_serialization.so``, say)
    is an ABI mismatch the loader is happy to perform. With several gepetto envs
    side by side it is also how one env ends up loading another's libgtsam.
    """
    prefix = Path(env["prefix"]).resolve()
    linked = _linked_libraries(Path(env["ext_path"]))
    matches = {so: p for so, p in linked.items() if so.startswith(stem)}
    assert matches, f"[{env['name']}] {stem} is not linked into the extension"

    for soname, path in matches.items():
        assert prefix in path.resolve().parents, (
            f"[{env['name']}] {soname} resolves to {path}, outside {prefix}"
        )


@needs_loader_tool
def test_exactly_one_boost_version_is_linked(env):
    """The regression this file was written for.

    conda-forge froze the ``boost`` metapackage at 1.85.0 with a hard
    ``libboost-python-devel ==1.85.0`` pin, while libpinocchio 4.0.0 needs >=1.88
    -- so ``boost`` and ``libboost-devel`` in one prefix is either an unsatisfiable
    solve (the loud failure, which is what the py10 setup log showed) or two Boosts
    on the loader path (the quiet one). GTSAM links Boost by unversioned soname, so
    the quiet case links and runs and corrupts memory across the serialization
    boundary.
    """
    linked = _linked_libraries(Path(env["ext_path"]))
    boost = [so for so in linked if so.startswith("libboost_")]
    versions = {
        m.group(1)
        for so in boost
        if (m := re.search(r"^libboost_\w+\.so\.(\d+\.\d+\.\d+)$", so))
    }
    # Asserted, not assumed: with no versioned soname matched, `len(versions) <= 1`
    # would hold vacuously and this test would silently stop checking anything.
    assert versions, (
        f"[{env['name']}] no versioned libboost soname among {boost}; the "
        "soname convention changed and this check needs updating"
    )
    assert len(versions) == 1, (
        f"[{env['name']}] extension links {len(versions)} Boost versions: "
        f"{sorted(versions)}; the prefix has both the frozen `boost` metapackage "
        "and `libboost-devel`"
    )


# ---------------------------------------------------------------------------
# The Eigen ABI pin, and the runtime stack, observed from inside the env
# ---------------------------------------------------------------------------


def test_pinocchio_and_extension_coexist(env):
    """Both loaded in one process, both still working -- the pin's actual purpose.

    A Pinocchio built against Eigen 5 alongside a GTSAM built against Eigen 3.4 is
    a compile-time static_assert for a .cpp including both, which is why the setup
    scripts pin the eigen-abi-3.4 ``libpinocchio`` build. This asserts the runtime
    half a user hits.
    """
    if env["modules"].get("pinocchio", "").startswith("FAIL"):
        pytest.skip(f"[{env['name']}] pinocchio not installed")

    coexist = env["coexist"]
    assert coexist["ok"], (
        f"[{env['name']}] pinocchio + gepetto_solvers in one process: "
        f"{coexist.get('error')}"
    )
    assert coexist["nq"] > 0


@pytest.mark.parametrize("name", ["numpy", "scipy", "trimesh"])
def test_runtime_dependency_imports(env, name):
    """pyproject's declared runtime deps, inside each env.

    Catches the mixed conda/pip state the setup scripts can leave behind: pip
    uninstalls conda's scipy to install its own wheel, and a later `conda install`
    in the same env can clobber the result.
    """
    version = env["modules"][name]
    assert not version.startswith("FAIL"), f"[{env['name']}] {name}: {version}"


@pytest.mark.parametrize("name", ["pyvista", "viser"])
def test_viz_extra_imports(env, name):
    """The ``[viz,web]`` extras the demo scripts and workbench open."""
    version = env["modules"][name]
    if version == "FAIL: ModuleNotFoundError":
        pytest.skip(f"[{env['name']}] {name} absent; install with .[viz,web]")
    assert not version.startswith("FAIL"), (
        f"[{env['name']}] {name} is installed but does not import: {version}"
    )
