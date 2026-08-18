"""Measure ``T_flange<-wrist`` from an Onshape assembly.

Given an Onshape assembly whose origin is the robot flange (the hand's attach
point) and which contains the exported hand STL as a part instance, this reads the
hand instance's occurrence transform -- Onshape's own answer to "where did you put
this part", exact to machine precision -- and composes it with the fixed CAD/solver
convention change in :mod:`mount` to get the wrist pose in flange coordinates.

    T_flange<-wrist  =  T_flange<-stl  @  T_stl<-wrist
                        ^ occurrence     ^ mount.py

The candidate conventions in :data:`mount.CANDIDATE_ROTATIONS` are scored against
the hand part's real bounding box, so a wrong axis convention is caught here rather
than shipped into a robot pose.

Credentials are read from the environment ONLY -- never flags (shell history),
never a file in this repo::

    ONSHAPE_ACCESS_KEY   API access key
    ONSHAPE_SECRET_KEY   API secret key
    ONSHAPE_URL          browser URL of the assembly tab

Keep those in a wrapper script outside the repo; see
``mount_onshape_fit.sh.example`` next to this file. Run from ``crest-sparse/``::

    python -m python.tests.tendon_hand.mount_onshape_fit --list
    python -m python.tests.tendon_hand.mount_onshape_fit --instance hand
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import requests
from requests.auth import HTTPBasicAuth

from . import mount
from .mount import DIGIT_NAMES

BASE = "https://cad.onshape.com"
# Onshape keeps older API versions live; try newest first and fall back rather
# than hard-coding a version that may be retired.
API_VERSIONS = ["v10", "v6", ""]

# Instance names to auto-match when --instance is not given.
DEFAULT_NAME_HINTS = ("hand", "palm", "finger")


class OnshapeError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Credentials / URL
# ---------------------------------------------------------------------------

def _require_env(name, what):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"missing environment variable {name} ({what}).\n"
            f"Set it in a wrapper script OUTSIDE this repo -- see "
            f"mount_onshape_fit.sh.example.")
    return value.strip()


def parse_document_url(url):
    """``(did, wvm, wvmid, eid)`` from an Onshape browser URL.

    Accepts the workspace form ``/documents/{did}/w/{wid}/e/{eid}`` and the
    version/microversion forms ``/v/{vid}/`` and ``/m/{mid}/``.
    """
    m = re.search(r"/documents/([0-9a-f]+)/([wvm])/([0-9a-f]+)/e/([0-9a-f]+)", url)
    if not m:
        raise SystemExit(
            "ONSHAPE_URL does not look like an Onshape element URL; expected\n"
            "  https://cad.onshape.com/documents/<did>/w/<wid>/e/<eid>\n"
            "(open the assembly tab in the browser and copy the address bar).")
    did, wvm, wvmid, eid = m.groups()
    return did, wvm, wvmid, eid


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class Onshape:
    """Minimal Onshape REST client (HTTP Basic, which Onshape documents as
    supported for local/internal use -- the HMAC signing scheme buys nothing here
    and adds clock-skew and nonce failure modes)."""

    def __init__(self, access_key, secret_key, timeout=60):
        self.auth = HTTPBasicAuth(access_key, secret_key)
        self.timeout = timeout
        self._version = None

    def get(self, path, **params):
        """GET ``/api/{version}{path}``, discovering a working API version once."""
        versions = [self._version] if self._version is not None else API_VERSIONS
        last = None
        for version in versions:
            prefix = f"{BASE}/api/{version}" if version else f"{BASE}/api"
            resp = requests.get(prefix + path, auth=self.auth, params=params,
                                headers={"Accept": "application/json"},
                                timeout=self.timeout)
            if resp.status_code == 404 and self._version is None:
                last = resp
                continue                      # maybe a retired API version
            if resp.status_code == 401:
                raise OnshapeError(
                    "401 Unauthorized -- check ONSHAPE_ACCESS_KEY / "
                    "ONSHAPE_SECRET_KEY (keys are per-account and can be revoked).")
            if resp.status_code == 403:
                raise OnshapeError(
                    "403 Forbidden -- the key's application does not have "
                    "read scope on this document, or the document is not shared "
                    "with the key's owner.")
            if not resp.ok:
                raise OnshapeError(
                    f"{resp.status_code} from {path}: {resp.text[:400]}")
            self._version = version
            return resp.json()
        raise OnshapeError(
            f"404 from {path} on every API version tried ({API_VERSIONS}); "
            f"last body: {last.text[:400] if last is not None else 'n/a'}")

    def assembly_definition(self, did, wvm, wvmid, eid):
        return self.get(f"/assemblies/d/{did}/{wvm}/{wvmid}/e/{eid}",
                        includeMateFeatures="true",
                        includeNonSolids="false",
                        includeMateConnectors="false")

    def part_bounding_box(self, did, wvm, wvmid, eid, partid):
        from urllib.parse import quote
        return self.get(
            f"/parts/d/{did}/{wvm}/{wvmid}/e/{eid}/partid/{quote(partid, safe='')}"
            f"/boundingboxes")


# ---------------------------------------------------------------------------
# Assembly parsing
# ---------------------------------------------------------------------------

def index_instances(assembly):
    """``{instance_id: instance}`` across the root assembly and all subassemblies.

    Occurrence paths are lists of instance ids that may descend into subassemblies,
    so resolving a path to readable names needs every instance, not just the root's.
    """
    index = {}
    for inst in assembly.get("rootAssembly", {}).get("instances", []):
        index[inst["id"]] = inst
    for sub in assembly.get("subAssemblies", []):
        for inst in sub.get("instances", []):
            index.setdefault(inst["id"], inst)
    return index


def occurrence_name(path, index):
    """Readable ``"outer / inner"`` name for an occurrence path."""
    return " / ".join(index.get(pid, {}).get("name", pid[:8]) for pid in path)


def occurrence_transform(occurrence):
    """The occurrence's 4x4 as a numpy array.

    Onshape reports it row-major with lengths in metres; it maps the instanced
    element's own coordinates into assembly (root) coordinates -- exactly the
    ``T_flange<-stl`` we want, given an assembly origin at the flange.
    """
    t = np.asarray(occurrence["transform"], float)
    if t.size != 16:
        raise OnshapeError(f"expected a 16-element transform, got {t.size}")
    return t.reshape(4, 4)


def validate_rigid(T, tol=1e-6):
    """``(scale, det, orthonormality_residual)``; raises if it is not a rigid motion.

    An STL imported with the wrong unit (mm read as inches, say) shows up as a
    uniform scale baked into the occurrence transform. Silently composing that
    would return a plausible-looking but wrong pose, so fail loudly with the number.
    """
    R = np.asarray(T, float)[:3, :3]
    scale = float(np.cbrt(abs(np.linalg.det(R))))
    det = float(np.linalg.det(R))
    resid = float(np.max(np.abs((R / scale).T @ (R / scale) - np.eye(3))))
    if abs(scale - 1.0) > tol:
        raise OnshapeError(
            f"occurrence transform carries a uniform scale of {scale:.9f} "
            f"(det={det:.9f}). The part was imported with a unit mismatch; fix the "
            f"import scale in Onshape rather than trusting this transform.")
    if resid > 1e-6:
        raise OnshapeError(
            f"occurrence rotation is not orthonormal (residual {resid:.2e}); the "
            f"instance is probably mirrored or non-uniformly scaled.")
    if det < 0:
        raise OnshapeError(
            f"occurrence rotation has det={det:.6f} < 0 -- the instance is "
            f"mirrored, which no rigid mount transform can represent.")
    return scale, det, resid


def pick_occurrence(assembly, index, wanted):
    """The one occurrence matching ``wanted`` (substring, case-insensitive).

    Only leaf part occurrences are considered -- a subassembly occurrence has its
    own transform but no part geometry to check against.
    """
    root = assembly.get("rootAssembly", {})
    candidates = []
    for occ in root.get("occurrences", []):
        path = occ.get("path", [])
        if not path:
            continue
        leaf = index.get(path[-1], {})
        if leaf.get("type") != "Part":
            continue
        name = occurrence_name(path, index)
        hints = (wanted.lower(),) if wanted else DEFAULT_NAME_HINTS
        if any(h in name.lower() for h in hints):
            candidates.append((name, occ, leaf))

    if not candidates:
        raise SystemExit(
            f"no part occurrence matched {wanted!r}. Run with --list to see the "
            f"instance names, then pass --instance <substring>.")
    if len(candidates) > 1:
        names = "\n  ".join(n for n, _, _ in candidates)
        raise SystemExit(
            f"{len(candidates)} part occurrences matched "
            f"{wanted or list(DEFAULT_NAME_HINTS)!r}:\n  {names}\n"
            f"Narrow it with --instance.")
    return candidates[0]


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

def containment_penalty(candidate, cad_points_hand, bbox_lo, bbox_hi):
    """``(penalty_m, n_outside)`` -- does the export flip belong, per the part itself?

    Maps the CAD-frame digit landmarks into the part frame through the candidate's
    ``R_hand<-stl`` and measures how far any of them fall OUTSIDE the part's own
    bounding box. Geometry cannot stick out of its own bounding box, so a candidate
    with real overhang is simply wrong.

    This deliberately uses the CAD placement chain rather than the solver's
    ``hand_base_offset``: the solver's mounting carries the known translation bug
    and an ``a_print`` conjugation, which blur the two flip hypotheses to within a
    couple of millimetres of each other -- close enough that noise decides. The CAD
    chain separates them by centimetres.
    """
    R_stl_from_hand = np.asarray(candidate.R_hand_from_stl, float).T
    penalty, outside = 0.0, 0
    for p in cad_points_hand.values():
        q = R_stl_from_hand @ p
        over = np.maximum(0.0, np.maximum(bbox_lo - q, q - bbox_hi))
        d = float(np.linalg.norm(over))
        penalty += d
        outside += d > 1e-9
    return penalty, outside


def score_candidate(candidate, cad_points_hand, bbox_lo, bbox_hi, dims):
    """``(yaw_deg, penalty_m, n_outside)`` for one convention.

    The two halves are tested against different evidence and neither can stand in
    for the other, so they are reported side by side rather than summed: the yaw
    sign comes from digit growth axes (offline, ~180 deg of separation), the export
    flip from bounding-box containment (needs the CAD measurement).
    """
    penalty, outside = containment_penalty(candidate, cad_points_hand,
                                           bbox_lo, bbox_hi)
    return mount.yaw_agreement_deg(candidate, dims), penalty, outside


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_matrix(T, indent="    "):
    return "\n".join(
        indent + "[" + "  ".join(f"{v:+11.6f}" for v in row) + "]" for row in T)


def report(T_flange_stl, R_used, used_name, dims, scores=None, bbox=None):
    T_flange_wrist = mount.compose_flange_from_wrist(T_flange_stl, R_used)
    T_wrist_flange = np.linalg.inv(T_flange_wrist)
    xyz, rpy = mount.as_xyz_rpy(T_flange_wrist)

    out = []
    add = out.append
    add("=" * 78)
    add("T_flange<-wrist   (wrist frame expressed in assembly/flange coordinates)")
    add("=" * 78)
    add(_fmt_matrix(T_flange_wrist))
    add("")
    add(f"  translation   {xyz[0]:+.6f}  {xyz[1]:+.6f}  {xyz[2]:+.6f}   m")
    add(f"                {xyz[0]*1e3:+8.3f}  {xyz[1]*1e3:+8.3f}  {xyz[2]*1e3:+8.3f}   mm")
    add(f"  rpy (ZYX)     {rpy[0]:+.6f}  {rpy[1]:+.6f}  {rpy[2]:+.6f}   rad")
    add(f"                {np.rad2deg(rpy[0]):+8.3f}  {np.rad2deg(rpy[1]):+8.3f}  "
        f"{np.rad2deg(rpy[2]):+8.3f}   deg")
    add(f"  convention    {used_name}")
    add("")
    add("paste-ready, same format/convention as solvers.DEFAULT_WRIST_XYZ/RPY:")
    add(f"    MOUNT_WRIST_XYZ = ({xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f})")
    add(f"    MOUNT_WRIST_RPY = ({rpy[0]:.6f}, {rpy[1]:.6f}, {rpy[2]:.6f})")
    add("")
    add("inverse, T_wrist<-flange (flange origin expressed in the wrist frame):")
    add(_fmt_matrix(T_wrist_flange))

    if scores:
        add("")
        add("=" * 78)
        add("candidate conventions, scored against the hand part bounding box")
        add("=" * 78)
        add("  yaw: mean CAD-vs-solver growth-axis disagreement (offline evidence)")
        add("  overhang: how far the CAD digits fall outside the real part's own")
        add("            bounding box -- geometry cannot do that, so nonzero = wrong")
        add("")
        add(f"  {'convention':40s} {'yaw (deg)':>10s} {'outside':>10s} "
            f"{'overhang (mm)':>14s}")
        for name, (yaw, penalty, outside), n_pts in scores:
            mark = "  <-- used" if name == used_name else ""
            add(f"  {name:40s} {yaw:10.2f} {outside:4d} / {n_pts:<3d} "
                f"{penalty*1e3:13.2f}{mark}")

    if bbox is not None:
        lo, hi = bbox
        add("")
        add("hand part bounding box, part frame (mm): "
            f"[{lo[0]*1e3:.1f},{hi[0]*1e3:.1f}] x [{lo[1]*1e3:.1f},{hi[1]*1e3:.1f}] "
            f"x [{lo[2]*1e3:.1f},{hi[2]*1e3:.1f}]")
        add("  The STL keeps its OpenSCAD coordinates on import, so the palm datum --")
        add("  the wrist origin -- should sit at 0 against one corner of this box,")
        add("  with the digits reaching ~160 mm along one axis. A box centred on 0")
        add("  instead means Onshape re-centred the import, and the 'the origins")
        add("  coincide' assumption behind this whole result is void.")

    # Landmarks to verify by hand in Onshape's Measure tool.
    pts = mount.digit_base_points_wrist(dims)
    disc = mount.mounting_discrepancy(dims)
    add("")
    add("=" * 78)
    add("verification: digit base landmarks in FLANGE coordinates (mm)")
    add("=" * 78)
    add("Measure these knuckle centres in Onshape and compare. The 'model err' "
        "column is")
    add("the known config.finger_base_offset() translation bug (see mount.py) -- "
        "expect")
    add("disagreement of about that size, worst at the thumb, and no more.")
    add(f"  {'digit':8s} {'x':>9s} {'y':>9s} {'z':>9s}   {'model err':>9s}")
    for name in DIGIT_NAMES:
        q = (T_flange_wrist @ np.append(pts[name], 1.0))[:3] * 1e3
        add(f"  {name:8s} {q[0]:9.2f} {q[1]:9.2f} {q[2]:9.2f}   "
            f"{disc[name][1]*1e3:9.2f}")

    g = mount.growth_axis_wrist(dims)
    g_flange = T_flange_wrist[:3, :3] @ g
    add("")
    add(f"mean digit growth axis in flange coords: "
        f"({g_flange[0]:+.4f}, {g_flange[1]:+.4f}, {g_flange[2]:+.4f})")
    add("  (sanity: should point away from the flange face, i.e. down the tool axis)")

    result = {
        "T_flange_from_wrist": T_flange_wrist.tolist(),
        "T_wrist_from_flange": T_wrist_flange.tolist(),
        "T_flange_from_stl": np.asarray(T_flange_stl, float).tolist(),
        "convention": used_name,
        "R_wrist_from_stl": np.asarray(R_used, float).tolist(),
        "wrist_xyz_m": list(map(float, xyz)),
        "wrist_rpy_zyx_rad": list(map(float, rpy)),
        "digit_base_landmarks_flange_m": {
            name: (T_flange_wrist @ np.append(pts[name], 1.0))[:3].tolist()
            for name in DIGIT_NAMES},
        "known_mounting_discrepancy_m": {
            name: disc[name][1] for name in DIGIT_NAMES},
    }
    return "\n".join(out), result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", default=None,
                    help="substring of the hand instance name in the assembly "
                         f"(default: match any of {list(DEFAULT_NAME_HINTS)})")
    ap.add_argument("--list", action="store_true",
                    help="list every occurrence with its translation and exit")
    ap.add_argument("--convention", default=None,
                    help="force one of mount.CANDIDATE_ROTATIONS instead of "
                         "scoring them (substring match)")
    ap.add_argument("--dump", metavar="PATH", default=None,
                    help="write the raw assembly JSON here for inspection")
    ap.add_argument("--json", metavar="PATH", default=None,
                    help="write the computed result as JSON")
    args = ap.parse_args(argv)

    access = _require_env("ONSHAPE_ACCESS_KEY", "Onshape API access key")
    secret = _require_env("ONSHAPE_SECRET_KEY", "Onshape API secret key")
    url = _require_env("ONSHAPE_URL", "assembly tab URL")

    did, wvm, wvmid, eid = parse_document_url(url)
    print(f"document {did[:8]}... element {eid[:8]}... ({wvm}/{wvmid[:8]}...)")

    client = Onshape(access, secret)
    assembly = client.assembly_definition(did, wvm, wvmid, eid)
    if args.dump:
        with open(args.dump, "w") as fh:
            json.dump(assembly, fh, indent=2)
        print(f"raw assembly JSON -> {args.dump}")

    index = index_instances(assembly)
    occurrences = assembly.get("rootAssembly", {}).get("occurrences", [])
    print(f"assembly has {len(occurrences)} occurrences, "
          f"{len(index)} instances\n")

    if args.list:
        print(f"  {'type':10s} {'translation (mm)':>26s}  name")
        for occ in occurrences:
            path = occ.get("path", [])
            if not path:
                continue
            leaf = index.get(path[-1], {})
            t = occurrence_transform(occ)[:3, 3] * 1e3
            print(f"  {leaf.get('type', '?'):10s} "
                  f"({t[0]:7.2f},{t[1]:7.2f},{t[2]:7.2f})  "
                  f"{occurrence_name(path, index)}")
        return 0

    name, occ, leaf = pick_occurrence(assembly, index, args.instance)
    T_flange_stl = occurrence_transform(occ)
    scale, det, resid = validate_rigid(T_flange_stl)
    print(f"hand instance: {name}")
    print(f"  occurrence transform is rigid (scale={scale:.9f}, det={det:+.9f}, "
          f"orthonormality residual={resid:.2e})")

    # Part bounding box, in the part's own frame -- the geometry the candidate
    # conventions are scored against.
    bbox = None
    try:
        bb = client.part_bounding_box(
            leaf.get("documentId", did),
            "m" if leaf.get("documentMicroversion") else wvm,
            leaf.get("documentMicroversion") or wvmid,
            leaf["elementId"], leaf["partId"])
        bbox = (np.array([bb["lowX"], bb["lowY"], bb["lowZ"]], float),
                np.array([bb["highX"], bb["highY"], bb["highZ"]], float))
        print(f"  part bounding box (mm): "
              f"{np.round(bbox[0]*1e3, 1)} .. {np.round(bbox[1]*1e3, 1)}")
    except (OnshapeError, KeyError) as exc:
        print(f"  [warn] could not read the part bounding box ({exc}); candidate "
              f"conventions cannot be scored, falling back to the derived one.")

    from .config import load_hand_dimensions
    dims = load_hand_dimensions()

    scores, R_used, used_name = None, mount.R_WRIST_FROM_STL, mount.DERIVED_CANDIDATE
    if args.convention:
        matches = [k for k in mount.CANDIDATE_ROTATIONS
                   if args.convention.lower() in k.lower()]
        if len(matches) != 1:
            raise SystemExit(
                f"--convention {args.convention!r} matched {len(matches)} "
                f"candidates; options are:\n  " +
                "\n  ".join(mount.CANDIDATE_ROTATIONS))
        used_name = matches[0]
        R_used = mount.CANDIDATE_ROTATIONS[used_name].R
        print(f"  convention forced to {used_name}")
    elif bbox is not None:
        pts = mount.cad_digit_points_hand(dims)
        # Rank on the yaw first (it is the sharper test by far), then containment.
        scores = sorted(
            ((k, score_candidate(c, pts, *bbox, dims), len(pts))
             for k, c in mount.CANDIDATE_ROTATIONS.items()),
            key=lambda row: (row[1][0], row[1][1]))
        used_name, best, _ = scores[0]
        R_used = mount.CANDIDATE_ROTATIONS[used_name].R
        if used_name != mount.DERIVED_CANDIDATE:
            print(f"\n  [WARN] the evidence prefers {used_name!r}, not the derived "
                  f"{mount.DERIVED_CANDIDATE!r}.\n"
                  f"         Trust the evidence, but check the assembly is what you "
                  f"think it is\n         before using this pose on the robot.")
        # Same yaw, and containment can't separate the flip -> genuinely ambiguous.
        ties = [row for row in scores[1:] if abs(row[1][0] - best[0]) < 1.0
                and abs(row[1][1] - best[1]) < 5e-3]
        if ties:
            print(f"\n  [WARN] {ties[0][0]!r} scores the same as the winner on both "
                  f"tests.\n         The part geometry does not discriminate them; "
                  f"verify with the landmark\n         table below before use.")

    print()
    text, result = report(T_flange_stl, R_used, used_name, dims, scores, bbox)
    print(text)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nresult -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
