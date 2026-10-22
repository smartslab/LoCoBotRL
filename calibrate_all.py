# -*- coding: utf-8 -*-
"""calibrate_all.py

One program that measures everything the clutter experiment needs, writes
the numbers into the running robot server, checks that a lone red block can
be grasped from a single camera frame, and prints the training command.

Run it on the Jetson while the robot server is up:

    python3 calibrate_all.py --host locobot.local            # full, ~15 min
    python3 calibrate_all.py --host locobot.local --quick    # daily, ~3 min

FULL mode (first time, or after changing the gripper, the table or the
camera mount). Steps 1-4 run on an EMPTY table, step 5 needs the red block:
  1. TABLE      fit a plane to the table top; find its height and edges
  2. FINGERTIPS send the closed jaw to several spots above the table and
                photograph it. Where the arm was told to go and where the
                camera sees the fingertips are both in the camera's frame,
                so the error between "where the camera says" and "where
                the arm goes" is read directly, in height and sideways,
                and corrected. Then it looks again to confirm.
  3. REACH      move the gripper outwards and record, for each gripper
                pitch, how far the arm can hold it at the heights needed
  4. WORKSPACE  table area that the arm can reach -> workspace box, push
                height, drop point, release point
  5. GRASP      real grasps of the red block, each from ONE frame, the same
                code path the training uses. After a success the robot puts
                the block down at another spot by itself and tries there
  6. RESULT     the server writes ~/locobot_calibration.json and loads it
                at every start; the training command is printed

QUICK mode (every day, when only the robot or the table may have been
bumped): steps 1 and 4, then two real grasps, then 6. The arm's own numbers
(reach, pitch, fingertip correction) are reused from the last full run.

You are asked to do something by hand at most three times: clear the table,
put the red block down, and (once) type the jaw opening you measured with a
ruler. Everything else is automatic.

The program stops and explains instead of guessing when
  * the table height it finds differs from the stored one by more than 1 cm
    (a floor was once taken for the table and the arm hit the table),
  * a measured correction is implausibly large, or the fingertips do not
    come back to the same place twice,
  * the grasps do not succeed. In that case NO training command is printed.
After a stop the server's values are put back as they were.
"""

from __future__ import print_function

import argparse
import datetime
import json
import os
import socket
import sys
import time

import numpy as np

import route2_perception as R

MIN_BUILD = "2026-10-14b"
SCAN_HEIGHTS = (0.07, 0.04)   # above the table, visited top down. The
# primitives need +0.04 (push approach) and +0.06..0.07 (pre-grasp,
# lift); +0.10 was dropped on Oct 22: it was never needed and every
# extra extended hold loads the elbow motor (joint_3 Overload).
GRASP_DEPTHS = (-0.012, -0.020, -0.025)   # GRASP_DZ ladder, shallow first
R_REF = 0.30
TABLE_MIN_Z, TABLE_MAX_Z = 0.05, 0.40   # a table top is between these
BLIND_RADIUS = 0.10     # perception drops anything this close to the parked
                        # gripper (Route2Perception.arm_radius)


class Stop(Exception):
    """Calibration cannot go on. The message says why and what to do."""


# ---------------- talking to the human ----------------

def ask(tag, text, kind="enter"):
    """Every request to the human goes through here."""
    print("\n>>> [%s] %s" % (tag, text))
    sys.stdout.flush()
    if kind == "enter":
        input("    press Enter when done ... ")
        return True
    if kind == "yesno":
        return input("    type y to accept, anything else to stop: "
                     ).strip().lower() in ("y", "yes")
    raw = input("    type the number and press Enter (just Enter = skip): "
                ).strip()
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def say(text=""):
    print(text)
    sys.stdout.flush()


def header(title):
    say("\n" + "=" * 72)
    say(title)
    say("=" * 72)


def mm(value):
    return "%+.1f mm" % (value * 1000.0)


# ---------------- talking to the robot server ----------------

class Robot(object):
    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=600)
        self.rfile = self.sock.makefile("r")

    def send(self, msg):
        self.sock.sendall((json.dumps(msg) + "\n").encode())
        line = self.rfile.readline()
        if not line:
            raise Stop("the robot server closed the connection. Look at "
                       "the server terminal on the robot.")
        resp = json.loads(line)
        if "error" in resp:
            raise Stop("the server rejected %r: %s"
                       % (msg.get("cmd"), resp["error"]))
        return resp

    def frame(self):
        return R.decode_frame(self.send({"cmd": "frame"}))

    def config(self):
        return self.send({"cmd": "get_config"})

    def set(self, values):
        values = dict((k, float(v)) for k, v in values.items())
        return self.send({"cmd": "set_config", "values": values})["config"]

    def home(self):
        return self.send({"cmd": "home"})

    def close(self):
        # the file made by makefile() keeps the connection open on its own
        for thing in (self.rfile, self.sock):
            try:
                thing.close()
            except Exception:
                pass


# ---------------- step 1: the table ----------------

def scene_cloud(robot, frames=2, stride=2):
    """Points of a few frames in the robot's base frame."""
    clouds = []
    for _ in range(frames):
        frame = robot.frame()
        pts, _ = R.backproject(frame["depth"], frame["intrinsics"],
                               frame["cam_to_base"], stride=stride)
        clouds.append(pts)
    pts = np.concatenate(clouds)
    radial = np.hypot(pts[:, 0], pts[:, 1])
    keep = ((pts[:, 0] > -0.10) & (pts[:, 0] < 0.80) &
            (pts[:, 1] > -0.90) & (pts[:, 1] < 0.30) &
            (pts[:, 2] > -0.15) & (pts[:, 2] < 0.60) & (radial < 0.80))
    return pts[keep]


class Grid(object):
    """A top-down occupancy grid of the table points, 1 cm cells."""

    CELL = 0.01

    def __init__(self, xy, min_count=3):
        self.x0 = float(np.floor(xy[:, 0].min() / self.CELL) * self.CELL)
        self.y0 = float(np.floor(xy[:, 1].min() / self.CELL) * self.CELL)
        ix = ((xy[:, 0] - self.x0) / self.CELL).astype(int)
        iy = ((xy[:, 1] - self.y0) / self.CELL).astype(int)
        counts = np.zeros((ix.max() + 1, iy.max() + 1), dtype=int)
        np.add.at(counts, (ix, iy), 1)
        self.occ = counts >= min_count

    def largest_component(self):
        """Keep only the biggest connected patch (4-neighbour)."""
        occ = self.occ
        label = np.zeros(occ.shape, dtype=int)
        best, best_size, current = 0, 0, 0
        for start in zip(*np.nonzero(occ)):
            if label[start]:
                continue
            current += 1
            stack, size = [start], 0
            label[start] = current
            while stack:
                i, j = stack.pop()
                size += 1
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = i + di, j + dj
                    if (0 <= a < occ.shape[0] and 0 <= b < occ.shape[1]
                            and occ[a, b] and not label[a, b]):
                        label[a, b] = current
                        stack.append((a, b))
            if size > best_size:
                best, best_size = current, size
        self.occ = label == best if best else occ
        return self

    @staticmethod
    def _shift_any(mask, radius):
        padded = np.pad(mask, radius, mode="constant")
        out = np.zeros_like(mask)
        n, m = mask.shape
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                out |= padded[radius + di:radius + di + n,
                              radius + dj:radius + dj + m]
        return out

    def close_holes(self, radius=4):
        """Fill the holes left by objects standing on the table."""
        grown = self._shift_any(np.pad(self.occ, radius, mode="constant"),
                                radius)
        shrunk = ~self._shift_any(~grown, radius)
        self.occ = shrunk[radius:-radius, radius:-radius]
        return self

    def cells(self):
        ii, jj = np.nonzero(self.occ)
        return np.stack([self.x0 + (ii + 0.5) * self.CELL,
                         self.y0 + (jj + 0.5) * self.CELL], axis=1)

    def area(self):
        return float(self.occ.sum()) * self.CELL ** 2

    def has(self, x, y):
        # the small shift keeps a point that sits exactly on a cell
        # border from falling into either cell by rounding accident
        i = int(np.floor((x - self.x0) / self.CELL + 1e-6))
        j = int(np.floor((y - self.y0) / self.CELL + 1e-6))
        return (0 <= i < self.occ.shape[0] and 0 <= j < self.occ.shape[1]
                and bool(self.occ[i, j]))

    def supported(self, x, y, margin):
        """On the table, and still on it `margin` away in every direction."""
        return all(self.has(x + dx, y + dy) for dx, dy in (
            (0, 0), (margin, 0), (-margin, 0), (0, margin), (0, -margin)))


def table_candidates(pts):
    """Flat surfaces in the cloud, each a possible table top."""
    z = pts[:, 2]
    edges = np.arange(z.min(), z.max() + 0.004, 0.004)
    if len(edges) < 4:
        return []
    hist, edges = np.histogram(z, bins=edges)
    smooth = np.convolve(hist, [1, 2, 1], mode="same")
    order = [k for k in np.argsort(smooth)[::-1]
             if smooth[k] >= smooth[max(k - 1, 0)]
             and smooth[k] >= smooth[min(k + 1, len(smooth) - 1)]]
    picked = []
    for k in order:
        zc = 0.5 * (edges[k] + edges[k + 1])
        if any(abs(zc - other) < 0.016 for other in picked):
            continue
        picked.append(zc)
        if len(picked) == 6:
            break
    out = []
    for zc in picked:
        slab = pts[np.abs(z - zc) < 0.012]
        if len(slab) < 300:
            continue
        grid = Grid(slab[:, :2]).largest_component()
        cells = grid.cells()
        near = float((np.hypot(cells[:, 0], cells[:, 1]) < 0.50).sum()
                     ) * Grid.CELL ** 2
        out.append({"z": float(np.median(slab[:, 2])), "points": len(slab),
                    "area": grid.area(), "near_area": near})
    return out


def fit_plane(pts, z_guess):
    """Least-squares plane z = a*x + b*y + c through the table points.
    The first pass only has a flat slice of a possibly tilted table; each
    further pass re-selects the points near the plane found so far."""
    sel = pts[np.abs(pts[:, 2] - z_guess) < 0.012]
    coef = np.array([0.0, 0.0, z_guess])
    for _ in range(5):
        if len(sel) < 200:
            raise Stop("too few table points for a plane fit (%d). Is the "
                       "camera looking at the table?" % len(sel))
        A = np.stack([sel[:, 0], sel[:, 1], np.ones(len(sel))], axis=1)
        coef = np.linalg.lstsq(A, sel[:, 2], rcond=None)[0]
        residual = pts[:, 2] - (pts[:, 0] * coef[0] + pts[:, 1] * coef[1]
                                + coef[2])
        sel = pts[np.abs(residual) < 0.008]
    A = np.stack([sel[:, 0], sel[:, 1], np.ones(len(sel))], axis=1)
    rms = float(np.sqrt(np.mean((A.dot(coef) - sel[:, 2]) ** 2)))
    if abs(coef[0]) > 0.15 or abs(coef[1]) > 0.15:
        raise Stop("the fitted table plane is tilted by more than 8 degrees "
                   "(slopes %.3f, %.3f). That is not a table top; check "
                   "what the camera sees." % (coef[0], coef[1]))
    return coef, sel, rms


def describe_view(robot, image_dir):
    """What the camera sees, in numbers and pictures, for when no table
    is found. Saves view_rgb.png and view_depth.png."""
    frame = R.decode_frame(robot.send({"cmd": "frame", "raw": True}))
    depth = frame["depth"]
    valid = depth > 0
    say("\nWHAT THE CAMERA SEES")
    say("  image %dx%d, %.0f%% of the pixels have a depth reading"
        % (depth.shape[1], depth.shape[0], 100.0 * valid.mean()))
    if valid.any():
        d = depth[valid] / 1000.0
        say("  distance to the camera: nearest %.2f m, median %.2f m, "
            "farthest %.2f m" % (d.min(), np.median(d), d.max()))
        close = (valid & (depth < 300)).mean() / max(valid.mean(), 1e-9)
        if close > 0.2:
            say("  %.0f%% of what it sees is closer than 30 cm: something "
                "(the arm?) is right in front of the lens" % (close * 100))
        pts, _ = R.backproject(depth, frame["intrinsics"],
                               frame["cam_to_base"], stride=4)
        say("  where those points are (robot frame): x %.2f..%.2f, "
            "y %.2f..%.2f, z %.2f..%.2f m"
            % tuple(v for axis in range(3) for v in np.percentile(
                pts[:, axis], [5, 95])))
        say("  (z is height above the FLOOR; the table top must show up "
            "here as a flat surface at its tape-measured height, within "
            "about 0.45 m of the robot centre)")
    cam = frame["cam_to_base"][:3, 3]
    look = frame["cam_to_base"][:3, 2]
    down_deg = float(np.degrees(np.arcsin(-float(np.clip(look[2], -1, 1)))))
    hit = (cam[2] / max(-look[2], 1e-6)) if look[2] < 0 else None
    say("  camera at (%.2f, %.2f, %.2f), looking towards (%.2f, %.2f, "
        "%.2f): %.0f deg below horizontal%s%s"
        % (tuple(cam) + tuple(look) + (
            down_deg,
            ("; its centre line meets the floor %.2f m ahead" % hit)
            if hit is not None else "",
            "  <- pointing UP, not at the table" if look[2] > -0.2 else "")))
    try:
        ee = robot.send({"cmd": "ee"})["ee"]
        say("  arm end effector at (%.2f, %.2f, %.2f)" % tuple(ee))
    except Exception:
        pass
    try:
        from PIL import Image
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)
        Image.fromarray(frame["rgb"]).save(
            os.path.join(image_dir, "view_rgb.png"))
        top = np.percentile(depth[valid], 99) if valid.any() else 1.0
        vis = (np.clip(depth / max(top, 1.0), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(vis).save(os.path.join(image_dir, "view_depth.png"))
        say("  pictures: %s/view_rgb.png and view_depth.png (black in the "
            "depth picture = no reading)" % image_dir)
    except Exception as error:
        say("  (could not save the pictures: %s)" % error)


def measure_table(robot, stored_z, image_dir="calib_images", hint=None):
    """Find the table plane. Returns a dict with the plane, the points on
    it and a top-down map of where the table is.

    hint: the table top's height above the floor in metres, from a tape
    measure (the robot's base frame has its origin at floor level). With a
    hint the floor cut-off moves to hint - 0.06 instead of the fixed
    TABLE_MIN_Z, so a very low table is not thrown away as "the floor",
    and the surface nearest the hint is taken."""
    # With a hint, anything more than 5 cm below the hinted height is the
    # floor (no lower clamp: a board lying on the floor is a valid table).
    floor_below = TABLE_MIN_Z if hint is None else hint - 0.05
    pts = scene_cloud(robot)
    if len(pts) < 2000:
        describe_view(robot, image_dir)
        raise Stop("the camera returns almost no points near the robot "
                   "(%d). See WHAT THE CAMERA SEES above." % len(pts))
    found = [c for c in table_candidates(pts) if c["area"] >= 0.03]
    found.sort(key=lambda c: -c["near_area"])
    say("flat surfaces in view (height in the camera frame):")
    for c in found:
        c["floor"] = c["z"] < floor_below
        say("    z=%.3f m   area %.2f m^2, of which %.2f m^2 within arm's "
            "reach%s" % (c["z"], c["area"], c["near_area"],
                         "   <- the floor, ignored" if c["floor"] else ""))
    # The robot's base frame has its origin at floor level, so a surface
    # lower than TABLE_MIN_Z is the floor, however big it looks. (The floor
    # is usually the BIGGEST surface in view; picking by size alone is how
    # the arm was once sent to the wrong height.)
    cands = [c for c in found if not c["floor"] and c["z"] <= TABLE_MAX_Z
             and c["near_area"] >= 0.03]
    if not cands:
        describe_view(robot, image_dir)
        raise Stop("no table top in view: there is no flat surface of at "
                   "least 0.03 m^2 within arm's reach between %.2f and "
                   "%.2f m height. See WHAT THE CAMERA SEES above."
                   % (floor_below, TABLE_MAX_Z))
    if hint is not None:
        # In floor mode the robot's own body and arm hide much of the near
        # surface from the camera, so a smaller visible patch is accepted;
        # the workspace is then usually given by hand with --box.
        cands = [c for c in found if not c["floor"] and c["z"] <= TABLE_MAX_Z
                 and c["near_area"] >= 0.01]
        near_hint = [c for c in cands if abs(c["z"] - hint) < 0.03]
        if not near_hint:
            describe_view(robot, image_dir)
            raise Stop("--table_hint says the table top is %.2f m above the "
                       "floor, but no flat surface within 3 cm of that is "
                       "in view (found: %s). Measure again from the FLOOR "
                       "to the table top, or point the camera lower."
                       % (hint, ", ".join("%.3f" % c["z"] for c in cands)))
        chosen = min(near_hint, key=lambda c: abs(c["z"] - hint))
    else:
        near_stored = [c for c in cands if abs(c["z"] - stored_z) < 0.012]
        chosen = (min(near_stored, key=lambda c: abs(c["z"] - stored_z))
                  if near_stored else cands[0])
    coef, inliers, rms = fit_plane(pts, chosen["z"])
    grid = Grid(inliers[:, :2]).largest_component().close_holes()
    centre = grid.cells().mean(axis=0)
    z_centre = float(coef[0] * centre[0] + coef[1] * centre[1] + coef[2])
    if abs(z_centre - stored_z) > 0.010:
        say("\nThe table height found now (%.3f) differs from the stored "
            "one (%.3f) by %.1f cm." % (z_centre, stored_z,
                                        abs(z_centre - stored_z) * 100))
        say("If the table or the robot was really changed, that is fine. "
            "If not, the camera may be looking at the floor or a shelf, "
            "and going on would send the arm to the wrong height.")
        if not ask(
                "CONFIRM-TABLE", "Is %.3f m the real table top? Check: the "
                "list above should show ONE big surface within arm's reach."
                % z_centre, kind="yesno"):
            raise Stop("stopped at the table height check. Nothing was "
                       "changed on the server.")
    return {"coef": coef, "rms": rms, "grid": grid, "inliers": len(inliers),
            "candidates": found, "z_centre": z_centre}


def plane_z(table, x, y):
    c = table["coef"]
    return float(c[0] * x + c[1] * y + c[2])


# ---------------- step 3: the workspace ----------------

def spread(points, first, count, min_gap):
    """Pick `count` points far from each other: start at `first`, then
    always take the point farthest from everything chosen so far."""
    chosen = [np.asarray(first, dtype=float)]
    points = np.asarray(points, dtype=float)
    while len(chosen) < count and len(points):
        dist = np.min([np.linalg.norm(points - c, axis=1) for c in chosen],
                      axis=0)
        if dist.max() < min_gap:
            break
        chosen.append(points[int(np.argmax(dist))])
    return [c.tolist() for c in chosen]


def fingertip_spots(table, home_xy, count=6):
    """Spots over the table for photographing the closed jaw: on the table
    (2 cm from its edge), from near (0.24 m) to far (0.35 m) so a height
    error that grows with distance shows up, and not under the parked
    gripper."""
    grid = table["grid"]
    cells = grid.cells()
    xs = np.arange(np.floor(cells[:, 0].min() * 100) / 100,
                   cells[:, 0].max() + 0.005, 0.01)
    ys = np.arange(np.floor(cells[:, 1].min() * 100) / 100,
                   cells[:, 1].max() + 0.005, 0.01)
    pts = np.array([[x, y] for x in xs for y in ys
                    if 0.24 <= np.hypot(x, y) <= 0.35
                    and grid.supported(x, y, 0.02)
                    and np.hypot(x - home_xy[0], y - home_xy[1]) > 0.13])
    if len(pts) < 30:
        raise Stop("there is almost no table between 24 and 35 cm from the "
                   "robot centre (%d cm^2 seen). Move the robot closer to "
                   "the table." % len(pts))
    first = pts[np.argmin(np.linalg.norm(pts - pts.mean(axis=0), axis=1))]
    return spread(pts, first, count, 0.04)


def plan_workspace(table, r_lo, r_max, home_xy, edge=0.01, box=None):
    """Table area the arm can reach -> everything the server and the
    training script need to know about where things are.

    box: (x_min, x_max, y_min, y_max) given by hand (--box). Then the
    camera's map of the surface is not consulted for the box itself,
    only for the surface height; used when the work surface is the floor
    or a board the camera cannot see in full."""
    grid = table["grid"]
    if box is not None:
        xs = np.arange(box[0], box[1] + 0.005, 0.01)
        ys = np.arange(box[2], box[3] + 0.005, 0.01)
        pts = np.array([[x, y] for x in xs for y in ys
                        if np.hypot(x, y) <= r_max])
    else:
        cells = grid.cells()
        xs = np.arange(np.floor(cells[:, 0].min() * 100) / 100,
                       cells[:, 0].max() + 0.005, 0.01)
        ys = np.arange(np.floor(cells[:, 1].min() * 100) / 100,
                       cells[:, 1].max() + 0.005, 0.01)
        pts = np.array([[x, y] for x in xs for y in ys
                        if np.hypot(x, y) <= r_max
                        and grid.supported(x, y, edge)])
    if len(pts) < 60:
        raise Stop("the table area the camera sees and the area the arm "
                   "can reach (radial <= %.2f m) barely overlap (%d cm^2). "
                   "Move the robot closer to the table." % (r_max, len(pts)))
    box = {"x_min": round(float(pts[:, 0].min()), 2),
           "x_max": round(float(pts[:, 0].max()), 2),
           "y_min": round(float(pts[:, 1].min()), 2),
           "y_max": round(float(pts[:, 1].max()), 2)}
    box_area = ((box["x_max"] - box["x_min"] + 0.01)
                * (box["y_max"] - box["y_min"] + 0.01))
    centre = pts.mean(axis=0)
    heights = np.array([plane_z(table, x, y) for x, y in pts])
    table_z = round(plane_z(table, centre[0], centre[1]), 3)
    radial = np.hypot(pts[:, 0], pts[:, 1])
    inside = ((pts[:, 0] >= box["x_min"] + 0.03)
              & (pts[:, 0] <= box["x_max"] - 0.03)
              & (pts[:, 1] >= box["y_min"] + 0.03)
              & (pts[:, 1] <= box["y_max"] - 0.03))
    seen = np.linalg.norm(pts - np.asarray(home_xy), axis=1) > (
        BLIND_RADIUS + 0.03)
    band = pts[inside & seen & (radial >= r_lo + 0.02)
               & (radial <= r_max - 0.03)]
    if len(band) < 20:
        raise Stop("after keeping 3 cm from the edges there is almost no "
                   "room left to put a block (%d cm^2). Move the robot "
                   "closer to the table." % len(band))
    anchor = band[np.argmin(np.linalg.norm(band - band.mean(axis=0),
                                           axis=1))]
    drop = band[np.argmax(np.linalg.norm(band - anchor, axis=1))]
    # places to test the grasp at, spread out and away from the drop point
    spots = spread(band[np.linalg.norm(band - drop, axis=1) > 0.07],
                   anchor, 5, 0.05)
    return {
        "table_z": table_z, "box": box, "anchor": anchor.tolist(),
        "drop": drop.tolist(), "spots": spots,
        "area_cm2": int(len(pts)),
        "box_fill": float(len(pts) * 1e-4 / box_area),
        "tilt_low": float(heights.min() - table_z),
        "tilt_high": float(heights.max() - table_z),
        "band": [r_lo + 0.02, r_max - 0.03], "r_lo": r_lo, "r_max": r_max,
        "drop_to_anchor": float(np.linalg.norm(drop - anchor)),
    }


def workspace_values(ws, release=None):
    release = ws["anchor"] if release is None else release
    z = ws["table_z"]
    return {
        "TABLE_Z": z,
        "WS_X_MIN": ws["box"]["x_min"], "WS_X_MAX": ws["box"]["x_max"],
        "WS_Y_MIN": ws["box"]["y_min"], "WS_Y_MAX": ws["box"]["y_max"],
        "WS_Z_MAX": z + 0.20, "SAFE_Z_MIN": z + 0.025,
        "HARD_Z_MIN": z - 0.005, "PUSH_EE_Z": z + 0.02,
        "TARGET_X": round(ws["anchor"][0], 3),
        "TARGET_Y": round(ws["anchor"][1], 3),
        "RELEASE_X": round(release[0], 3), "RELEASE_Y": round(release[1], 3),
        "DROP_X": round(ws["drop"][0], 3), "DROP_Y": round(ws["drop"][1], 3),
        "MAX_RADIAL": ws["r_max"],
        # the lower fingertip must clear the HIGHEST part of the table in
        # the workspace, not just its height at the centre
        "FINGER_TABLE_MARGIN": min(0.004 + max(ws["tilt_high"], 0.0), 0.015),
    }


def report_workspace(ws):
    b = ws["box"]
    say("table height (camera frame)   : %.3f m" % ws["table_z"])
    say("table height inside the box   : %s to %s around that value"
        % (mm(ws["tilt_low"]), mm(ws["tilt_high"])))
    say("workspace box                 : x %.2f..%.2f   y %.2f..%.2f"
        % (b["x_min"], b["x_max"], b["y_min"], b["y_max"]))
    say("reachable table area          : %d cm^2" % ws["area_cm2"])
    say("put blocks at radial distance : %.2f..%.2f m from the robot centre"
        % (ws["band"][0], ws["band"][1]))
    say("anchor / release point        : (%.2f, %.2f)" % tuple(ws["anchor"]))
    say("drop point for cleared blocks : (%.2f, %.2f), %.0f cm from the "
        "anchor" % (ws["drop"][0], ws["drop"][1],
                    ws["drop_to_anchor"] * 100))
    if ws["tilt_high"] - ws["tilt_low"] > 0.016:
        say("NOTE: the table height changes by %.0f mm across the box. "
            "Perception cuts the table at one height, so short blocks at "
            "the low end may lose points. Level the table or the robot if "
            "detections look wrong there."
            % ((ws["tilt_high"] - ws["tilt_low"]) * 1000))
    if ws["box_fill"] < 0.55:
        say("NOTE: only %.0f%% of the box is table the arm can reach (the "
            "table stands at an angle to the robot). Pushes aimed at the "
            "empty corners are refused or shortened by the server. "
            "Squaring the table to the robot gives more room."
            % (ws["box_fill"] * 100))
    if ws["drop_to_anchor"] < 0.10:
        say("NOTE: the drop point is only %.0f cm from the anchor; cleared "
            "blocks will land close to the target area."
            % (ws["drop_to_anchor"] * 100))


# ---------------- step 2: the reach of the arm ----------------

def make_route2(cfg, anchor, max_extent):
    pc = R.PerceptionConfig(
        ws_x=(cfg["WS_X_MIN"], cfg["WS_X_MAX"]),
        ws_y=(cfg["WS_Y_MIN"], cfg["WS_Y_MAX"]),
        table_z=cfg["TABLE_Z"], z_max=cfg["TABLE_Z"] + 0.20)
    return R.Route2Perception(
        R.GeometricBackend(pc), identity="color", target_label="red_object",
        anchor_xy=tuple(anchor), target_max_extent=max_extent)


def objects_on_table(robot, cfg, table, home_xyz):
    """What perception sees standing on the table. A patch of the table
    itself can show up as an 'object' where a tilted table rises above the
    single height perception cuts at; such a patch hugs the fitted plane
    and is not counted."""
    route2 = make_route2(cfg, (cfg["TARGET_X"], cfg["TARGET_Y"]), 0.30)
    summary = route2.start_episode(robot.frame(), ee_xyz=home_xyz)
    return [d for d in summary["detections"]
            if d["centroid"][2] - plane_z(table, d["centroid"][0],
                                          d["centroid"][1]) > 0.012]


def pack_array(array):
    import base64
    import zlib
    array = np.ascontiguousarray(array)
    return {"encoding": "zlib+base64", "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": base64.b64encode(zlib.compress(array.tobytes(), 1)
                                     ).decode("ascii")}


def shift_reduce(array, radius, reduce):
    """min / max over a (2r+1)^2 neighbourhood, with numpy shifts."""
    padded = np.pad(array, radius, mode="edge")
    h, w = array.shape
    out = array.copy()
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            out = reduce(out, padded[radius + di:radius + di + h,
                                     radius + dj:radius + dj + w])
    return out


def record_background(robot, cfg, table, home_xyz, args):
    """With the table EMPTY and the arm at home, everything the camera
    sees standing above the table is part of the robot (the arm, cables,
    the camera mount) or fixed in place. Record its depth on the server,
    which then masks it in every frame taken with the arm at home.

    Anything that rests ON the table is listed for the human first: it is
    most likely a block that was left behind."""
    for _ in range(6):
        robot.home()
        frames = [R.decode_frame(robot.send({"cmd": "frame", "raw": True}))
                  for _ in range(5)]
        stack = np.stack([f["depth"] for f in frames]).astype(np.float64)
        valid = np.all(stack > 0, axis=0)
        depth = np.where(valid, np.median(stack, axis=0), 0.0)
        frame = dict(frames[0], depth=depth.astype(np.uint16))
        # height of every pixel above the fitted table plane
        K, T = frame["intrinsics"], frame["cam_to_base"]
        h, w = depth.shape
        vv, uu = np.mgrid[0:h, 0:w]
        z = depth / 1000.0
        x = (uu - K[0, 2]) * z / K[0, 0]
        y = (vv - K[1, 2]) * z / K[1, 1]
        pts = np.stack([x, y, z, np.ones_like(z)], axis=-1).dot(T.T)
        c = table["coef"]
        height = pts[..., 2] - (c[0] * pts[..., 0] + c[1] * pts[..., 1]
                                + c[2])
        static = valid & (height > 0.008) & (height < 0.80)
        # what perception would make of it
        route2 = make_route2(cfg, (cfg["TARGET_X"], cfg["TARGET_Y"]), 0.30)
        found = [d for d in route2.start_episode(frame, ee_xyz=home_xyz)
                 ["detections"]
                 if d["centroid"][2] - plane_z(table, d["centroid"][0],
                                               d["centroid"][1]) > 0.012]
        # From above the camera sees only the top of things, so how high
        # something "floats" says nothing about whether it rests on the
        # table. Colour does: the arm and cables are dark or grey, the
        # task's blocks are coloured. Low and coloured = probably a block.
        suspects = []
        for d in found:
            d["top"] = (d["centroid"][2] + d["extent"][2] / 2.0
                        - plane_z(table, d["centroid"][0], d["centroid"][1]))
            coloured = d.get("sat", 0.0) >= R.GRAY_SAT_MIN
            if coloured and d["top"] < 0.10:
                suspects.append(d)
        photo = os.path.join(args.image_dir, "background.png")
        save_photo(photo, frame, np.argwhere(static), [])
        say("with the table empty and the arm at home, the camera sees %d "
            "pixel(s) standing above the table (painted yellow in %s)."
            % (int(static.sum()), photo))
        if found:
            say("perception makes these objects of them:")
            for d in found:
                say("    %-14s at (%.2f, %.2f), top %.0f cm above the "
                    "table%s"
                    % (d["label"], d["centroid"][0], d["centroid"][1],
                       d["top"] * 100,
                       "   <- LOOKS LIKE A BLOCK" if d in suspects else ""))
            text = ("Look at %s. Is everything listed part of the robot "
                    "(the arm in its home pose, cables, the camera mount) "
                    "or fixed in place? y = remember it as background and "
                    "ignore it from now on. Anything else = I took a real "
                    "object away, look again." % photo)
            if suspects:
                text = ("%d of them are coloured and low, which looks like "
                        "a block left behind. " % len(suspects)) + text
            if not ask("CONFIRM-BACKGROUND", text, kind="yesno"):
                continue
        # cover the noisy outline of the arm: 2 px wider, with the depth of
        # the nearest background pixel
        far = np.where(static, depth, 1e9)
        grown = shift_reduce(far, 2, np.minimum)
        background = np.where(grown < 1e8, grown, 0).astype(np.uint16)
        resp = robot.send({"cmd": "set_background",
                           "depth": pack_array(background)})
        say("background stored on the robot (%d pixels, %s). From now on it "
            "is ignored whenever the arm is at home."
            % (resp["pixels"], resp["file"]))
        # check at once that it works
        robot.home()
        test = robot.send({"cmd": "frame"})
        if test.get("at_home") is False:
            raise Stop("the background was stored, but the server says the "
                       "arm is not at home (joints %.2f rad away from the "
                       "recorded pose), so it is not applied. Send this "
                       "output." % test.get("home_joint_error", -1))
        left = objects_on_table(robot, cfg, table, home_xyz)
        say("check with the background applied: %d object(s) left"
            % len(left))
        return {"pixels": int(resp["pixels"]),
                "objects": [(d["label"], [round(v, 3) for v in
                                          d["centroid"]]) for d in found]}
    raise Stop("the background was never confirmed. Clear the table and "
               "run again.")


def check_background(robot, meta):
    """Quick mode: is the stored background still where it was? If the
    camera was turned or the home pose changed, it no longer is."""
    stored = robot.config().get("background") or {}
    if not stored.get("saved_at"):
        say("NOTE: no background is stored on the robot, so the arm at "
            "home may be seen as an object. Run a full calibration.")
        return None
    if not stored.get("pixels"):
        say("stored background: the camera saw nothing of the robot at "
            "home, so there is nothing to ignore")
        return 1.0
    robot.home()
    match = robot.frame().get("background_match")
    if match is None:
        say("NOTE: the stored background was not used (wrong image size, "
            "or the arm is not at home). Run a full calibration.")
        return None
    say("stored background (arm at home, cables): %.0f%% of it is where "
        "it was" % (match * 100))
    if match < 0.6:
        raise Stop("the stored background no longer matches what the "
                   "camera sees (%.0f%%). The camera was turned or the home "
                   "pose changed. Run a full calibration (without --quick)."
                   % (match * 100))
    return match


def wait_for_clear_table(robot, cfg, table, home_xyz):
    for _ in range(6):
        robot.home()
        meta = robot.send({"cmd": "frame"})
        found = objects_on_table(robot, cfg, table, home_xyz)
        if not found:
            return
        say("the camera still sees %d object(s) on the table:" % len(found))
        for d in found:
            say("    %s at (%.2f, %.2f), %.0f cm above the table"
                % (d["label"], d["centroid"][0], d["centroid"][1],
                   (d["centroid"][2] - cfg["TABLE_Z"]) * 100))
        if meta.get("at_home") is False:
            raise Stop("the stored background is NOT applied because the "
                       "server says the arm is not at home (joints %.2f rad "
                       "from the recorded pose; the limit is HOME_JOINT_TOL, "
                       "0.10 by default). Send this output."
                       % meta.get("home_joint_error", -1))
        ask("ACTION-CLEAR", "Take EVERYTHING off the table (the arm is "
            "about to move over it).")
    raise Stop("the table never looked empty. If it really is empty, the "
               "table height or the lighting is confusing perception; send "
               "the output above.")


def reach_scan(robot, cfg, direction, pitches):
    """For each pitch: the radial distances at which the gripper can be
    held at every height the primitives need. A refused point costs no
    motion; a feasible one is really moved to (from above)."""
    table_z = cfg["TABLE_Z"]

    def inside(xy):
        return (cfg["WS_X_MIN"] <= xy[0] <= cfg["WS_X_MAX"]
                and cfg["WS_Y_MIN"] <= xy[1] <= cfg["WS_Y_MAX"])

    def probe(radial, pitch):
        xy = direction * radial
        if not inside(xy):
            return None
        row = {}
        for h in SCAN_HEIGHTS:
            xyz = [float(xy[0]), float(xy[1]), float(table_z + h)]
            resp = robot.send({"cmd": "goto", "xyz": xyz, "pitch": pitch})
            ok = bool(resp.get("ok"))
            if ok:
                err = float(np.linalg.norm(np.asarray(resp["ee"]) - xyz))
                ok = err < 0.015
            row["%.2f" % h] = ok
        return row

    # Radial distances to try: a coarse ladder, limited to the part of the
    # line that lies inside the box (the near table edge can cut off the
    # first rungs). The gap around the limit is halved afterwards.
    coarse = [r for r in (0.24, 0.27, 0.30, 0.34, 0.38, 0.42)
              if inside(direction * r)]
    if len(coarse) < 2:
        raise Stop("the line from the robot to the middle of the table "
                   "crosses less than 6 cm of reachable table. The table "
                   "is too far away or almost out of the camera's view.")
    scan = {}
    for pitch in pitches:
        rows = {}
        usable, first_bad = None, None
        for radial in coarse:
            row = probe(radial, pitch)
            # Park between rungs: a far rung is held with the arm stretched
            # out, and holding that through several heights is what tripped
            # the elbow motor's overload protection on Oct 10.
            robot.home()
            if row is None:        # left the box on the far side
                first_bad = radial
                break
            rows["%.2f" % radial] = row
            if all(row.values()):
                usable = radial
            elif usable is not None:
                first_bad = radial
                break
        if usable is not None and first_bad is not None \
                and first_bad - usable > 0.03:
            middle = round((usable + first_bad) / 2.0, 2)
            row = probe(middle, pitch)
            if row is not None:
                rows["%.2f" % middle] = row
                if all(row.values()):
                    usable = middle
        scan["%.2f" % pitch] = {"rows": rows, "r_max": usable}
        robot.home()
        say("  pitch %.2f rad (%2.0f deg from straight down): %s"
            % (pitch, np.degrees(abs(np.pi / 2 - pitch)),
               "usable out to %.2f m" % usable if usable is not None
               else "not usable anywhere on this table"))
        for key in sorted(rows):
            say("      radial %s m: %s" % (key, "  ".join(
                "+%s:%s" % (h, "ok" if rows[key][h] else "--")
                for h in sorted(rows[key]))))
    return scan


def choose_pitch(scan, current_pitch, forced=None):
    reach = dict((float(k), v["r_max"]) for k, v in scan.items()
                 if v["r_max"] is not None)
    if not reach:
        raise Stop("the arm could not hold the gripper at any tested pitch "
                   "anywhere over the table. The table is too far or too "
                   "high for this arm. Move the robot closer.")
    if forced is not None:
        key = min(reach, key=lambda p: abs(p - forced))
        if abs(key - forced) > 0.011:
            raise Stop("--pitch %.2f was not usable in the scan." % forced)
        return key, reach[key], "forced with --pitch"
    best = max(reach.values())
    current = min(reach, key=lambda p: abs(p - current_pitch))
    if abs(current - current_pitch) < 0.011 and reach[current] >= best - 0.02:
        return current, reach[current], (
            "kept: it is the pitch in use and no other pitch reaches more "
            "than 2 cm farther")
    key = max((p for p in reach if reach[p] >= best - 1e-9))
    return key, reach[key], "it reaches farthest"


def inner_limit(robot, cfg, table, pitch, fallback):
    """How close to the robot the gripper can still be held."""
    cells = table["grid"].cells()
    ok = [c for c in cells if table["grid"].supported(c[0], c[1], 0.03)
          and cfg["WS_X_MIN"] <= c[0] <= cfg["WS_X_MAX"]
          and cfg["WS_Y_MIN"] <= c[1] <= cfg["WS_Y_MAX"]]
    if not ok:
        return fallback
    near = min(ok, key=lambda c: np.hypot(c[0], c[1]))
    radial = round(float(np.hypot(near[0], near[1])), 2)
    if radial >= fallback:
        return fallback
    for h in SCAN_HEIGHTS:
        resp = robot.send({"cmd": "goto", "pitch": pitch, "xyz": [
            float(near[0]), float(near[1]), float(cfg["TABLE_Z"] + h)]})
        if not resp.get("ok"):
            robot.home()
            return fallback
    robot.home()
    return radial


# ---------------- step 4: fingertips against the block ----------------

def hsv_arrays(rgb):
    x = np.asarray(rgb, dtype=float) / 255.0
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    mx, mn = x.max(axis=1), x.min(axis=1)
    d = mx - mn
    safe = np.where(d > 1e-9, d, 1.0)
    h = np.select(
        [d <= 1e-9, mx == r, mx == g],
        [0.0, ((g - b) / safe) % 6.0, (b - r) / safe + 2.0],
        default=(r - g) / safe + 4.0) * 60.0
    s = np.where(mx > 1e-9, d / np.where(mx > 1e-9, mx, 1.0), 0.0)
    return h, s, mx


def blue_mask(colors):
    h, s, v = hsv_arrays(colors)
    return (h > 190.0) & (h < 260.0) & (s > 0.30) & (v > 0.10)


def pixel_cloud(frame):
    """Every pixel -> base-frame point, colour and (v, u) pixel."""
    depth = frame["depth"]
    K, T = frame["intrinsics"], frame["cam_to_base"]
    h, w = depth.shape
    vv, uu = np.mgrid[0:h, 0:w]
    z = depth.astype(np.float64) / 1000.0
    valid = z > 0.05
    u, v, z = uu[valid], vv[valid], z[valid]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z, np.ones_like(z)], axis=1).dot(T.T)[:, :3]
    return pts, frame["rgb"][v, u].astype(np.float64), np.stack([v, u], 1)


def save_photo(path, frame, pix, marks):
    """The photo with the pixels taken as fingers painted yellow."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    img = frame["rgb"].copy()
    if pix is not None and len(pix):
        img[pix[:, 0], pix[:, 1]] = (
            0.35 * img[pix[:, 0], pix[:, 1]]
            + 0.65 * np.array([255, 230, 0])).astype(np.uint8)
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    T_inv = np.linalg.inv(frame["cam_to_base"])
    K = frame["intrinsics"]
    for k, (xyz, color, label) in enumerate(marks):
        p = T_inv.dot(np.append(np.asarray(xyz, dtype=float), 1.0))
        if p[2] <= 1e-6:
            continue
        u = K[0, 0] * p[0] / p[2] + K[0, 2]
        v = K[1, 1] * p[1] / p[2] + K[1, 2]
        draw.line([(u - 10, v), (u + 10, v)], fill=color, width=2)
        draw.line([(u, v - 10), (u, v + 10)], fill=color, width=2)
        draw.text((u + 12, v - 12 + 14 * k), label, fill=color)
    im.save(path)


def correction_at(cfg, xy):
    """Same formula as arm_correction() in the server."""
    xy = np.asarray(xy, dtype=float)[:2]
    radial = float(np.hypot(xy[0], xy[1]))
    r_hat = xy / max(radial, 1e-6)
    t_hat = np.array([-r_hat[1], r_hat[0]])
    dxy = (np.array([cfg["ARM_DX"], cfg["ARM_DY"]])
           + cfg["ARM_DR"] * r_hat + cfg["ARM_DT"] * t_hat)
    dz = cfg["ARM_Z_OFFSET"] + cfg["ARM_Z_SLOPE"] * (radial
                                                    - cfg["ARM_R_REF"])
    return dxy, float(dz)


def measure_fingertips(robot, cfg, table, xyz, pitch, args, tag):
    """Send the fingertips to xyz over the EMPTY table, close the jaw,
    photograph it and read where the fingertips really are.

    No block is needed for this: the point the arm was told to go to and
    the fingertips the camera sees are both in the camera's frame, so
    their difference is the whole error between "where the camera says"
    and "where the arm goes" (camera mounting, arm frame offset, joint
    play). The jaw is closed because, open, one finger hides behind the
    arm link from this camera.

    Returns a sample: where the fingertips were told to be (c), how far
    off they were (e_xy, e_z; + means too far in +x/+y, or too HIGH), and
    the correction that would have been exactly right there (t_xy, t_z)."""
    commanded = np.asarray(xyz, dtype=float)
    resp = robot.send({"cmd": "goto", "xyz": commanded.tolist(),
                       "pitch": pitch})
    if not resp.get("ok"):
        return {"ok": False, "refused": True, "reason": "; ".join(
            resp.get("motion_errors") or ["the arm refused the pose"])}
    time.sleep(args.settle)
    robot.send({"cmd": "gripper", "state": "close"})
    time.sleep(args.settle)
    frame = robot.frame()
    robot.send({"cmd": "gripper", "state": "open"})
    pts, colors, pix = pixel_cloud(frame)
    # Only look where fingers can be: near the commanded point, clear of
    # the table top (part of this table is blue), and below the blue cable
    # sleeves on the wrist, which sit 20 cm higher.
    floor = plane_z(table, commanded[0], commanded[1]) + 0.012
    keep = (np.linalg.norm(pts[:, :2] - commanded[:2], axis=1) < 0.12)
    keep &= (pts[:, 2] > floor) & (pts[:, 2] < commanded[2] + 0.10)
    pts, colors, pix = pts[keep], colors[keep], pix[keep]
    blue = blue_mask(colors)
    pts, pix = pts[blue], pix[blue]
    photo = os.path.join(args.image_dir, "%s.png" % tag)
    marks = [(commanded, (0, 255, 0), "told to be here")]
    clusters = (R.cluster_points(pts, voxel=0.008, min_points=10)
                if len(pts) >= 25 else [])
    if not clusters:
        save_photo(photo, frame, None, marks)
        return {"ok": False, "reason": "no blue finger pads found near the "
                "commanded point with the jaw closed (photo: %s)" % photo}
    main = pts[clusters[0]].mean(axis=0)
    parts = [idx for idx in clusters
             if np.linalg.norm(pts[idx].mean(axis=0)[:2] - main[:2]) < 0.035]
    idx = np.concatenate(parts)
    jaw = pts[idx]
    centre = jaw[:, :2].mean(axis=0)
    tip_z = float(np.percentile(jaw[:, 2], 3))
    save_photo(photo, frame, pix[idx], marks + [
        (np.append(centre, tip_z), (255, 230, 0), "fingers are here")])
    e_xy = centre - commanded[:2]
    e_z = tip_z - float(commanded[2])
    if np.linalg.norm(e_xy) > 0.06 or not (-0.036 < e_z < 0.06):
        return {"ok": False, "reason": "the blob taken as the fingers is "
                "%.0f mm sideways and %.0f mm in height away from where "
                "they were sent. That is too much to be a calibration "
                "error; something else blue is in the picture (photo: %s)"
                % (np.linalg.norm(e_xy) * 1000, e_z * 1000, photo)}
    applied_xy, applied_z = correction_at(cfg, commanded[:2])
    return {"ok": True, "c": commanded.tolist(),
            "e_xy": e_xy.tolist(), "e_z": e_z,
            "t_xy": (applied_xy - e_xy).tolist(), "t_z": applied_z - e_z,
            "n_points": int(len(jaw)), "photo": photo}


def look_at_spots(robot, cfg, table, spots, pitch, args, tag, samples,
                  repeat_first=False):
    """Measure the fingertips at each spot; append the usable samples."""
    used = 0
    for i, spot in enumerate(spots):
        xyz = [float(spot[0]), float(spot[1]), cfg["TABLE_Z"] + args.hover]
        for rep in range(2 if (repeat_first and i == 0) else 1):
            m = measure_fingertips(robot, cfg, table, xyz, pitch, args,
                                   "%s_spot%d_%d" % (tag, i + 1, rep + 1))
            where = "(%.2f, %.2f)" % (xyz[0], xyz[1])
            if not m["ok"]:
                say("  %s: not usable - %s" % (where, m["reason"]))
                break
            m["spot"] = [round(xyz[0], 3), round(xyz[1], 3)]
            samples.append(m)
            used += 1
            say("  %s: fingertips %s in height, %.1f mm sideways  "
                "(x %s, y %s)"
                % (where, mm(m["e_z"]), np.linalg.norm(m["e_xy"]) * 1000,
                   mm(m["e_xy"][0]), mm(m["e_xy"][1])))
            if rep == 0 and repeat_first and i == 0:
                robot.home()     # come back from somewhere else
    return used


def calibrate_fingertips(robot, cfg, table, home_xyz, pitch, args, start_cfg,
                         tag):
    """Photograph the closed jaw at several spots over the empty table,
    fit the simplest correction, apply it, and look again to confirm.

    Two different things can be wrong and they are judged separately:
      * a steady offset (the same error every time). Calibration removes
        it. Confirmed when the AVERAGE error of the second look is small.
      * scatter (a different error every time). Calibration cannot remove
        it; it is reported, and later compared with the room the jaw
        leaves around the block. Only hopeless scatter stops the run."""
    spots = fingertip_spots(table, home_xyz[:2])
    samples = []
    look_at_spots(robot, cfg, table, spots, pitch, args, tag, samples,
                  repeat_first=True)
    robot.home()
    n_spots = len(set(tuple(s["spot"]) for s in samples))
    if n_spots < 3:
        raise Stop("the fingertips could be measured at only %d of %d "
                   "spots; at least 3 are needed. Look at the photos in "
                   "%s/: the yellow patch must lie on the two blue finger "
                   "pads." % (n_spots, len(spots), args.image_dir))
    # look again at the nearest, the farthest and one spot in between
    by_radial = sorted(spots, key=lambda p: np.hypot(p[0], p[1]))
    again = [by_radial[0], by_radial[len(by_radial) // 2], by_radial[-1]]
    for attempt in range(3):
        fit = fit_corrections(samples)
        check_plausible(fit["values"], start_cfg)
        cfg = robot.set(fit["values"])
        say("correction: height offset %.3f m (%s); sideways %s"
            % (fit["values"]["ARM_Z_OFFSET"], fit["z_model"],
               fit["xy_model"]))
        say("looking again with the correction in place:")
        checks = []
        look_at_spots(robot, cfg, table, again, pitch, args,
                      "%s_check%d" % (tag, attempt + 1), checks)
        robot.home()
        if len(checks) < 2:
            raise Stop("the fingertips could not be photographed again "
                       "after the correction; see the photos in %s/."
                       % args.image_dir)
        samples.extend(checks)
        errors = np.array([c["e_xy"] + [c["e_z"]] for c in checks])
        steady = errors.mean(axis=0)
        steady_xy, steady_z = float(np.linalg.norm(steady[:2])), abs(
            float(steady[2]))
        worst = float(np.max(np.linalg.norm(errors, axis=1)))
        if steady_xy < 0.004 and steady_z < 0.004:
            say("steady offset removed: on average the fingertips are now "
                "%.1f mm (height) and %.1f mm (sideways) from where they "
                "are sent" % (steady_z * 1000, steady_xy * 1000))
            if worst > 0.008:
                say("NOTE: single looks still scatter up to %.0f mm. That "
                    "scatter is the arm itself (it does not stop at the "
                    "same place twice); calibration cannot remove it. The "
                    "grasp test decides whether the jaw is wide enough to "
                    "absorb it." % (worst * 1000))
            return cfg, samples, fit_corrections(samples)
    if worst > 0.012:
        raise Stop("the fingertips do not come back to the same place: "
                   "after three rounds of correction single looks are "
                   "still up to %.0f mm off. That is play in the arm "
                   "(loose joint, slipping servo), not an offset that "
                   "calibration can remove." % (worst * 1000))
    say("NOTE: the average error did not settle below 4 mm (now %.1f mm in "
        "height, %.1f mm sideways), but single looks stay within %.0f mm. "
        "Going on; the grasp test decides whether that is good enough."
        % (steady_z * 1000, steady_xy * 1000, worst * 1000))
    return cfg, samples, fit_corrections(samples)


def fit_corrections(samples):
    """The simplest correction that explains every sample."""
    n_places = len(set(tuple(s["spot"]) for s in samples))
    c = np.array([s["c"] for s in samples])
    t_xy = np.array([s["t_xy"] for s in samples])
    t_z = np.array([s["t_z"] for s in samples])
    radial = np.hypot(c[:, 0], c[:, 1])
    r_hat = c[:, :2] / radial[:, None]
    t_hat = np.stack([-r_hat[:, 1], r_hat[:, 0]], axis=1)

    def rms(rows):
        """Root mean square length of a set of 2-D error vectors."""
        return float(np.sqrt(np.mean(np.sum(rows ** 2, axis=1))))

    # height: one number, or a line in the radial distance
    z0, slope, z_model = float(t_z.mean()), 0.0, "one value everywhere"
    z_res = t_z - z0
    # (with only 3 spots a line fits almost anything, noise included)
    if n_places >= 4 and np.ptp(radial) >= 0.06:
        A = np.stack([radial - R_REF, np.ones(len(radial))], axis=1)
        sol = np.linalg.lstsq(A, t_z, rcond=None)[0]
        line_res = t_z - A.dot(sol)
        if (np.sqrt(np.mean(z_res ** 2)) - np.sqrt(np.mean(line_res ** 2))
                > 0.002 and abs(sol[0]) <= 0.5):
            slope, z0 = float(sol[0]), float(sol[1])
            z_res, z_model = line_res, "changes with distance from the robot"

    # sideways: a fixed shift, or a shift along/across the arm's direction
    const = t_xy.mean(axis=0)
    const_res = t_xy - const
    pr = float(np.sum(t_xy * r_hat, axis=1).mean())
    pt = float(np.sum(t_xy * t_hat, axis=1).mean())
    polar_res = t_xy - (pr * r_hat + pt * t_hat)
    if n_places >= 2 and rms(polar_res) < rms(const_res) - 0.0005:
        values = {"ARM_DX": 0.0, "ARM_DY": 0.0, "ARM_DR": pr, "ARM_DT": pt}
        xy_res, xy_model = polar_res, "follows the arm's direction"
    else:
        values = {"ARM_DX": float(const[0]), "ARM_DY": float(const[1]),
                  "ARM_DR": 0.0, "ARM_DT": 0.0}
        xy_res, xy_model = const_res, "same shift everywhere"
    values.update({"ARM_Z_OFFSET": z0, "ARM_Z_SLOPE": slope,
                   "ARM_R_REF": R_REF})
    return {"values": values, "z_model": z_model, "xy_model": xy_model,
            "xy_left": float(np.max(np.linalg.norm(xy_res, axis=1))),
            "z_left": float(np.max(np.abs(z_res))),
            "xy_typical": rms(xy_res),
            "z_typical": float(np.sqrt(np.mean(z_res ** 2))),
            "n_samples": len(samples), "n_places": n_places}


def check_plausible(values, start):
    """A correction that large is a wrong measurement, not a calibration."""
    if abs(values["ARM_Z_OFFSET"] - start["ARM_Z_OFFSET"]) > 0.05:
        raise Stop("the measured height correction is %.0f mm, more than "
                   "the 50 mm this program will apply on its own. Look at "
                   "the photos in the image folder: is the yellow patch on "
                   "the two finger pads?" % (
                       (values["ARM_Z_OFFSET"] - start["ARM_Z_OFFSET"])
                       * 1000))
    for name in ("ARM_DX", "ARM_DY", "ARM_DR", "ARM_DT"):
        if abs(values[name]) > 0.04:
            raise Stop("the measured sideways correction %s is %.0f mm, "
                       "more than the 40 mm this program will apply on its "
                       "own. Check the photos in the image folder."
                       % (name, values[name] * 1000))


# ---------------- step 5: real grasps from one frame ----------------

def one_frame_grasp(robot, route2, max_radial, min_radial):
    """Exactly what the training does for the planner grasp: reset, ONE
    frame, route 2, one server step, then the two success checks."""
    reset = robot.send({"cmd": "reset"})
    home_xyz = np.asarray(reset["obs"], dtype=float)[5:8]
    summary = route2.start_episode(robot.frame(), ee_xyz=home_xyz)
    out = {"n_objects": summary["n_objects"], "status": "no_target",
           "reset_errors": reset.get("motion_errors", [])}
    if not summary["target_found"]:
        return out
    target = summary["target"]
    radial = float(np.hypot(target["centroid"][0], target["centroid"][1]))
    out.update({"xy": [float(v) for v in target["centroid"][:2]],
                "radial": radial, "target": target,
                "footprint": [target.get("rect_long"),
                              target.get("rect_short")]})
    if radial > max_radial or radial < min_radial:
        out["status"] = "out_of_band"
        return out
    resp = robot.send({"cmd": "step", "action": [1.0, 0.0, 0.0, 0.0, 0.0],
                       "perception": {"target": target,
                                      "obstacles": summary["obstacles"]},
                       "grasp_role": "target"})
    info = resp.get("info", {})
    out.update({"raw": info.get("gripper_raw"),
                "plan": info.get("grasp_plan", {}),
                "motion_errors": info.get("motion_errors", [])})
    if not info.get("grasp_completed"):
        out["status"] = "reach"
        return out
    if not info.get("held"):
        out["status"] = "slipped" if info.get("gripper_raw") == 2 else "missed"
        return out
    robot.home()
    after = route2.target_on_table(
        robot.frame(),
        exclude_xyz=np.asarray(resp["obs"], dtype=float)[5:8])
    out["status"] = ("success" if not after["target_on_table"]
                     else "held_but_red_still_on_table")
    return out


STATUS_TEXT = {
    "success": "SUCCESS (held for the hold time, red gone from the table)",
    "no_target": "no red target found in the frame",
    "out_of_band": "the block is outside the reachable band",
    "reach": "the arm refused a pose (reach)",
    "missed": "the jaw closed on nothing",
    "slipped": "the jaw closed on the block but lost it during the hold",
    "held_but_red_still_on_table":
        "the gripper holds something but red is still on the table",
}


def describe_grasp(index, g):
    where = ("at (%.2f, %.2f), radial %.2f m"
             % (g["xy"][0], g["xy"][1], g["radial"]) if "xy" in g else "")
    say("  grasp %d %s: %s" % (index, where, STATUS_TEXT[g["status"]]))
    plan = g.get("plan") or {}
    if plan:
        say("           fingertip height difference %.0f mm, upper fingertip "
            "%.0f mm below the block top"
            % (plan.get("tip_height_difference", 0) * 1000,
               plan.get("upper_tip_below_top", 0) * 1000))
    for reason in g.get("motion_errors", []):
        say("           [motion] %s" % reason)


def place_request(ws, cfg, why):
    spot = ws["anchor"]
    ask("ACTION-BLOCK", "%s Put ONLY the red block on the table, about "
        "%.0f cm in front of the robot centre and %.0f cm to its right "
        "(any spot %.0f-%.0f cm from the robot centre is fine), its long "
        "side pointing at the robot. Keep hands away afterwards: the arm "
        "moves." % (why, spot[0] * 100, -spot[1] * 100,
                    ws["band"][0] * 100, ws["band"][1] * 100))


# ---------------- step 6: output ----------------

def training_commands(args, cfg, max_extent):
    stamp = datetime.datetime.now().strftime("%m%d")
    flags = ("--table_z %.3f --ws_x_min %.2f --ws_x_max %.2f "
             "--ws_y_min %.2f --ws_y_max %.2f --anchor \"%.2f,%.2f\" "
             "--max_radial %.2f"
             % (cfg["TABLE_Z"], cfg["WS_X_MIN"], cfg["WS_X_MAX"],
                cfg["WS_Y_MIN"], cfg["WS_Y_MAX"], cfg["TARGET_X"],
                cfg["TARGET_Y"], cfg["MAX_RADIAL"]))
    if max_extent > 0.0851:
        flags += " --target_max_extent %.2f" % max_extent
    common = ("--host %s --perception geometric --identity color "
              "--target_label red_object \\\n  %s" % (args.host, flags))
    if args.port != 9999:
        common = "--port %d %s" % (args.port, common)
    check = ("python3 train_two_pipeline_clutter.py \\\n  %s \\\n  "
             "--steps 12 --warmup 12 --log_dir logs_random_%s"
             % (common, stamp))
    train = ("python3 train_two_pipeline_clutter.py \\\n  %s \\\n  "
             "--steps 400 --warmup 30 --utd 20 --batch 256 \\\n  "
             "--log_dir logs_clutter_%s --save ckpt_clutter_%s.pt \\\n  "
             "--save_replay logs_clutter_%s/replay.npz --seed 0"
             % (common, stamp, stamp, stamp))
    other = flags.split(" --max_radial")[0]
    return check, train, other


# ---------------- the whole procedure ----------------

def run(args, robot):
    result = {"mode": "quick" if args.quick else "full", "ok": False,
              "started": datetime.datetime.now().strftime(
                  "%Y-%m-%d %H:%M:%S")}
    ping = robot.send({"cmd": "ping"})
    build = ping.get("build") or ""
    if build < MIN_BUILD:
        raise Stop(
            "the server on the robot is build %r; this program needs %s or "
            "newer (it sets the server's numbers at run time). Copy the new "
            "grasp_env_server_rgbd.py to the robot and restart it:\n"
            "    scp grasp_env_server_rgbd.py locobot@locobot.local:~/\n"
            "    (robot)  pkill -f grasp_env_server ; "
            "source ~/server_env.sh && python ~/grasp_env_server_rgbd.py"
            % (build or "unknown", MIN_BUILD))
    state = robot.config()
    cfg, meta = state["config"], dict(state.get("meta") or {})
    start_cfg = dict(cfg)
    robot.start_cfg = start_cfg      # main() puts these back after a Stop
    say("server build %s, calibration file %s (%s)"
        % (build, state.get("calib_file"),
           "loaded" if state.get("calib_loaded") else "none yet"))
    if args.quick and not meta.get("full_done"):
        raise Stop("--quick reuses the arm's numbers from a full run, and "
                   "the server has none stored. Run once without --quick.")
    if not os.path.isdir(args.image_dir):
        os.makedirs(args.image_dir)

    # ----- jaw opening: the one number the camera cannot measure -----
    jaw_open = args.jaw_open
    if jaw_open is None and not meta.get("jaw_open_measured") \
            and not args.quick:
        value = ask("RULER-JAW", "Measure with a ruler: gripper fully open, "
                    "distance between the INNER faces of the two fingers, "
                    "in millimetres. (From the camera one open finger hides "
                    "behind the arm, so this cannot be measured "
                    "automatically. It is asked only once.)", kind="number")
        if value is not None:
            jaw_open = value / 1000.0
    if jaw_open is not None:
        if not 0.02 <= jaw_open <= 0.12:
            raise Stop("a jaw opening of %.0f mm is not believable (expected "
                       "20-120 mm; --jaw_open is in metres)."
                       % (jaw_open * 1000))
        cfg = robot.set({"GRIPPER_OPENING": jaw_open})
        meta["jaw_open_measured"] = True
    jaw_known = bool(meta.get("jaw_open_measured"))

    # The reach scan checks heights up to 10 cm above the table. Keep the
    # heights the primitives use inside that, for a block up to 4 cm tall.
    limits = {"PREGRASP_DZ": 0.04, "LIFT_DZ": 0.05, "RELEASE_DZ": 0.05,
              "PUSH_APPROACH_DZ": 0.05}
    lower = dict((k, v) for k, v in limits.items() if cfg[k] > v + 1e-9)
    if lower:
        say("these heights are above what the reach scan checks and are "
            "lowered: %s" % ", ".join(
                "%s %.2f -> %.2f" % (k, cfg[k], v)
                for k, v in sorted(lower.items())))
        cfg = robot.set(lower)

    # ================= 1. TABLE =================
    header("1. TABLE")
    if not args.quick:
        ask("ACTION-CLEAR", "Take EVERYTHING off the table. The arm will "
            "move over the whole table top with the gripper empty.")
    robot.home()
    home_xyz = np.asarray(robot.send({"cmd": "ee"})["ee"], dtype=float)
    table = measure_table(robot, cfg["TABLE_Z"], args.image_dir,
                          hint=args.table_hint)
    box = None
    if args.box:
        box = tuple(float(v) for v in args.box.split(","))
        if len(box) != 4 or box[0] + 0.05 > box[1] or box[2] + 0.05 > box[3]:
            raise Stop("--box must be x_min,x_max,y_min,y_max with at least "
                       "5 cm in each direction")
        say("work area given by hand: x %.2f..%.2f, y %.2f..%.2f"
            % box)
    say("plane fit: %d points, %.1f mm scatter, slope %.1f mm per 10 cm "
        "forward, %.1f mm per 10 cm sideways"
        % (table["inliers"], table["rms"] * 1000, table["coef"][0] * 100,
           table["coef"][1] * 100))
    result["table"] = {"coef": table["coef"].tolist(), "rms": table["rms"],
                       "candidates": table["candidates"]}

    samples, fit = [], None
    if args.quick:
        result["background_match"] = check_background(robot, meta)
        pitch = cfg["GRASP_PITCH"]
        r_max, r_lo = float(meta["r_max"]), float(meta["r_lo"])
        ws = plan_workspace(table, r_lo, r_max, home_xyz[:2], box=box)
        cfg = robot.set(workspace_values(ws))
    else:
        # A generous box first, so the arm may move anywhere over the table
        # while it is being measured. It is replaced in step 4.
        wide = plan_workspace(table, 0.20, 0.46, home_xyz[:2], box=box)
        cfg = robot.set(workspace_values(wide))
        header("1b. BACKGROUND  (what the camera sees of the robot itself)")
        result["background"] = record_background(robot, cfg, table,
                                                 home_xyz, args)
        meta["background_at"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M")
        wait_for_clear_table(robot, cfg, table, home_xyz)

        # ================= 2. FINGERTIPS =================
        header("2. FINGERTIPS  (where the arm really goes)")
        pitch = cfg["GRASP_PITCH"]
        say("closed jaw photographed %.0f cm above the empty table, at "
            "pitch %.2f:" % (args.hover * 100, pitch))
        cfg, samples, fit = calibrate_fingertips(
            robot, cfg, table, home_xyz, pitch, args, start_cfg, "tips")

        # ================= 3. REACH =================
        header("3. REACH OF THE ARM")
        centre = np.asarray(wide["anchor"], dtype=float)
        direction = centre / np.linalg.norm(centre)
        if (args.table_hint is not None and args.table_hint < 0.05
                and args.pitches == "1.0"):
            # reaching down to floor level favours a more vertical
            # gripper; let the scan compare, unless pitches were given
            args.pitches = "1.0,1.3,1.5"
        if args.skip_reach:
            r_max, why = float(args.max_radial), "given with --max_radial"
            scan = {}
            say("reach scan skipped (--skip_reach): using %.2f m from "
                "--max_radial, pitch %.2f kept" % (r_max, pitch))
            new_pitch = pitch
        else:
            say("moving the empty gripper along the line from the robot "
                "to (%.2f, %.2f); heights are above the table. The arm "
                "parks between points. If the server prints 'Overload', "
                "power-cycle the arm and run again with --skip_reach "
                "--max_radial 0.34." % tuple(centre))
            pitches = [float(v) for v in args.pitches.split(",")]
            if not any(abs(p - pitch) < 0.011 for p in pitches):
                pitches.insert(0, pitch)
            scan = reach_scan(robot, cfg, direction, pitches)
            new_pitch, r_max, why = choose_pitch(scan, pitch, args.pitch)
            say("\nchosen pitch %.2f rad, usable out to %.2f m  (%s)"
                % (new_pitch, r_max, why))
        if abs(new_pitch - pitch) > 0.011:
            # The fingertip height depends on the pitch, so the numbers of
            # step 2 do not carry over. Measure again, then scan this one
            # pitch again with correct heights.
            pitch = new_pitch
            cfg = robot.set({"GRASP_PITCH": pitch})
            say("\nthe pitch changed, so the fingertips are measured again "
                "at pitch %.2f:" % pitch)
            cfg, samples, fit = calibrate_fingertips(
                robot, cfg, table, home_xyz, pitch, args, start_cfg, "tips2")
            again = reach_scan(robot, cfg, direction, [pitch])
            scan.update(dict(("%s again" % k, v) for k, v in again.items()))
            r_max = again["%.2f" % pitch]["r_max"]
            if r_max is None:
                raise Stop("pitch %.2f was usable in the first scan but "
                           "not after the fingertip correction. Run again "
                           "with --pitch %.2f to keep the pitch in use."
                           % (pitch, start_cfg["GRASP_PITCH"]))
        r_lo = inner_limit(robot, cfg, table, pitch, 0.24)
        say("closest usable distance: %.2f m" % r_lo)
        result["reach"] = {"scan": scan, "pitch": pitch, "r_max": r_max,
                           "r_lo": r_lo, "why": why}
        ws = plan_workspace(table, r_lo, r_max, home_xyz[:2], box=box)
        cfg = robot.set(workspace_values(ws))

    # ================= 4. WORKSPACE =================
    header("4. WORKSPACE")
    report_workspace(ws)
    home_gap = float(np.hypot(
        max(ws["box"]["x_min"] - home_xyz[0], 0,
            home_xyz[0] - ws["box"]["x_max"]),
        max(ws["box"]["y_min"] - home_xyz[1], 0,
            home_xyz[1] - ws["box"]["y_max"])))
    blind_note = None
    if home_gap < BLIND_RADIUS:
        blind_note = (
            "perception ignores anything within %.0f cm of the parked "
            "gripper, which rests at (%.2f, %.2f). The part of the box "
            "nearest to it is a blind spot: a block pushed there reads as "
            "'gone'." % (BLIND_RADIUS * 100, home_xyz[0], home_xyz[1]))
        say("NOTE: " + blind_note)
    spots = ws["spots"]
    if not args.quick:
        # The workspace is final now. Look at the fingertips once more at
        # the very spots the grasps will be tested at (the far ones were
        # not part of step 2), and fold what is seen into the correction.
        say("\nfingertips at the %d test spots of the final workspace:"
            % len(spots))
        before = len(samples)
        look_at_spots(robot, cfg, table, spots, pitch, args, "final",
                      samples)
        robot.home()
        if len(samples) > before:
            fit = fit_corrections(samples)
            check_plausible(fit["values"], start_cfg)
            cfg = robot.set(fit["values"])
        say("what the correction does not explain, over %d looks at %d "
            "spots: typically %.1f mm sideways and %.1f mm in height, at "
            "worst %.1f mm and %.1f mm"
            % (fit["n_samples"], fit["n_places"], fit["xy_typical"] * 1000,
               fit["z_typical"] * 1000, fit["xy_left"] * 1000,
               fit["z_left"] * 1000))
        result["fit"] = dict(fit, samples=samples)
    result["workspace"] = ws

    # ================= 5. GRASP =================
    header("5. GRASP FROM ONE FRAME")
    grasps = []
    max_extent = 0.13
    route2 = make_route2(cfg, ws["anchor"], max_extent)
    depth_i = GRASP_DEPTHS.index(min(GRASP_DEPTHS, key=lambda d: abs(
        d - cfg["GRASP_DZ"]))) if args.quick else 0
    n_test = args.trials if args.trials is not None else (
        2 if args.quick else 5)
    failure = None
    reach_fail = []
    block_dims = None
    released = [0]

    def next_release():
        """Where the NEXT reset puts a held block: the next test spot."""
        released[0] += 1
        spot = spots[released[0] % len(spots)]
        return robot.set({"RELEASE_X": round(spot[0], 3),
                          "RELEASE_Y": round(spot[1], 3)})

    def find_block(why):
        """Home, one frame; ask for the block until it is usable."""
        for _ in range(6):
            reset = robot.send({"cmd": "reset"})
            ee = np.asarray(reset["obs"], dtype=float)[5:8]
            summary = route2.start_episode(robot.frame(), ee_xyz=ee)
            if summary["target_found"]:
                c = summary["target"]["centroid"]
                radial = float(np.hypot(c[0], c[1]))
                # patches of a tilted table do not count as objects
                standing = [d for d in summary["detections"]
                            if d["centroid"][2] - plane_z(
                                table, d["centroid"][0], d["centroid"][1])
                            > 0.012]
                if len(standing) > 1:
                    why = ("The camera sees %d objects; only the red block "
                           "may be on the table." % len(standing))
                elif not (ws["band"][0] - 0.01 <= radial
                          <= ws["band"][1] + 0.01):
                    why = ("The red block is %.0f cm from the robot centre, "
                           "outside the %.0f-%.0f cm band the arm reaches "
                           "well." % (radial * 100, ws["band"][0] * 100,
                                      ws["band"][1] * 100))
                else:
                    return summary
            place_request(ws, cfg, why)
            why = "The red block was not found where it can be used."
        raise Stop("the red block never showed up as a usable target. Check "
                   "lighting and that it is the only object on the table.")

    summary = find_block("The table part is done." if not args.quick
                         else "The grasp check needs the red block.")
    target = summary["target"]
    block_dims = (float(target.get("rect_long") or 0.03),
                  float(target.get("rect_short") or 0.03))
    say("red block as the camera sees it: %.0f x %.0f mm footprint, at "
        "(%.2f, %.2f)" % (block_dims[0] * 1000, block_dims[1] * 1000,
                          target["centroid"][0], target["centroid"][1]))

    def attempt(phase):
        cfg_now = robot.set({"GRASP_DZ": GRASP_DEPTHS[depth_i]})
        g = one_frame_grasp(robot, route2, r_max + 0.01, r_lo - 0.01)
        g["phase"], g["grasp_dz"] = phase, GRASP_DEPTHS[depth_i]
        grasps.append(g)
        describe_grasp(len(grasps), g)
        if g["status"] == "reach":
            reach_fail.append(g["radial"])
        return cfg_now, g

    if not args.quick:
        # How deep to close. Start shallow (the careful direction) and go
        # deeper only when the jaw closes on nothing.
        say("\nfinding the grasp depth (shallow first):")
        held = False
        for _ in range(len(GRASP_DEPTHS)):
            cfg = next_release()
            cfg, g = attempt("depth")
            if g["status"] == "success":
                held = True
                break
            if g["status"] in ("missed", "slipped") \
                    and depth_i + 1 < len(GRASP_DEPTHS):
                depth_i += 1
                say("           next try closes %.0f mm below the block top"
                    % (-GRASP_DEPTHS[depth_i] * 1000))
            elif g["status"] in ("no_target", "out_of_band"):
                find_block("The failed grasp moved the block away.")
            elif g["status"] == "reach":
                break
        if not held:
            failure = ("the block could not be grasped at any depth, with "
                       "the fingertips already corrected")
        elif depth_i == 0:
            depth_i = 1          # 12 mm was only the careful first try

    if failure is None:
        say("\nTEST: %d grasps, each from ONE frame, at different spots"
            % n_test)
        tested, lost = 0, 0
        while tested < n_test:
            cfg = next_release()
            cfg, g = attempt("test")
            if g["status"] in ("no_target", "out_of_band"):
                # the grasp before knocked the block away; that grasp is
                # already counted as failed, this one is not a grasp
                g["phase"] = "lost"
                lost += 1
                if lost > 3:
                    failure = "the red block keeps ending up out of reach"
                    break
                find_block("The block is not where it can be grasped.")
                continue
            tested += 1

    # put the block back at the anchor and park the arm
    cfg = robot.set({"RELEASE_X": round(ws["anchor"][0], 3),
                     "RELEASE_Y": round(ws["anchor"][1], 3)})
    robot.send({"cmd": "reset"})

    tests = [g for g in grasps if g.get("phase") == "test"]
    wins = sum(1 for g in tests if g["status"] == "success")
    result.update({"grasps": [dict((k, v) for k, v in g.items()
                                   if k != "target") for g in grasps],
                   "test_success": wins, "test_total": len(tests)})
    if failure is None and not tests:
        failure = "no test grasp was run"
    if failure is None and wins < int(np.ceil(0.8 * len(tests))):
        failure = "only %d of %d test grasps succeeded" % (wins, len(tests))
    if failure is None and fit is not None and block_dims is not None:
        # Five grasps are a small sample. If the arm's typical scatter
        # (two of them, to cover most attempts) is more than the room the
        # jaw leaves on each side, 4 of 5 was luck and training would see
        # many misses.
        room = 0.5 * (cfg["GRIPPER_OPENING"] - block_dims[1])
        if 2.0 * fit["xy_typical"] > room:
            failure = ("the arm's typical sideways scatter (%.1f mm, so "
                       "%.1f mm for most attempts) is more than the %.1f mm "
                       "of room the jaw leaves on each side; %d of %d test "
                       "grasps was luck"
                       % (fit["xy_typical"] * 1000,
                          2000 * fit["xy_typical"], room * 1000, wins,
                          len(tests)))

    if reach_fail and failure is None:
        # the far end was refused in a real grasp: pull the limit in
        r_max = round(min(reach_fail) - 0.02, 2)
        say("\nA real grasp was refused at radial %.2f m; the reach limit "
            "is reduced to %.2f m." % (min(reach_fail), r_max))
        ws = plan_workspace(table, r_lo, r_max, home_xyz[:2], box=box)
        cfg = robot.set(workspace_values(ws))
        result["workspace"] = ws

    # ================= 6. RESULT =================
    header("6. RESULT")
    diagonal = float(np.hypot(block_dims[0], block_dims[1]))
    max_extent = max(0.085, float(np.ceil((diagonal + 0.012) * 100) / 100.0))
    if failure is not None:
        # the depth found by trial is not trustworthy if the test failed
        cfg = robot.set({"GRASP_DZ": start_cfg["GRASP_DZ"]})
    meta.update({
        "calibrated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mode": result["mode"],
        "status": "ok" if failure is None else "grasp_failed",
        "r_max": r_max, "r_lo": r_lo, "target_max_extent": max_extent,
        "test": "%d/%d" % (wins, len(tests))})
    if not args.quick:
        meta["full_done"] = failure is None
    saved = robot.send({"cmd": "save_config", "meta": meta})
    robot.start_cfg = None           # stored on purpose; nothing to undo
    result.update({"config": cfg, "meta": meta, "failure": failure})

    changed = [(k, start_cfg[k], cfg[k]) for k in sorted(cfg)
               if abs(cfg[k] - start_cfg[k]) > 5e-4]
    say("server values changed in this run (%d):" % len(changed))
    for name, old, new in changed:
        say("    %-20s %8.3f -> %8.3f" % (name, old, new))
    say("test grasps: %d of %d succeeded" % (wins, len(tests)))

    # does this gripper leave enough room?
    tilt = abs(np.pi / 2 - cfg["GRASP_PITCH"])
    worst_diff = cfg["GRIPPER_OPENING"] * np.sin(tilt)
    room = 0.5 * (cfg["GRIPPER_OPENING"] - block_dims[1])
    left = fit["xy_left"] if fit else None
    say("\nGRIPPER ROOM%s"
        % ("" if jaw_known else "  (the jaw opening is an ESTIMATE; "
           "measure it and rerun with --jaw_open)"))
    say("    jaw opening %.0f mm, block %.0f mm across -> %.1f mm of room "
        "on each side" % (cfg["GRIPPER_OPENING"] * 1000,
                          block_dims[1] * 1000, room * 1000))
    if left is not None:
        say("    sideways error left after calibration: up to %.1f mm -> %s"
            % (left * 1000, "enough room" if left < room - 0.002 else
               "NOT enough room: a finger can land on the block"))
    say("    at pitch %.2f one fingertip is up to %.0f mm higher than the "
        "other when the jaw has to close along the line to the robot, and "
        "level when it closes across that line"
        % (cfg["GRASP_PITCH"], worst_diff * 1000))
    result["gripper_room"] = {"room": room, "xy_left": left,
                              "tip_difference_worst": worst_diff,
                              "jaw_known": jaw_known}

    if failure is not None:
        say("\nNOT READY FOR TRAINING: %s." % failure)
        say("Table, workspace, reach and the fingertip correction are "
            "stored (each was checked on its own). The grasp depth was put "
            "back to its old value. --quick stays locked until a full run "
            "passes.")
        counts = {}
        for g in grasps:
            if g.get("phase") in ("test", "depth"):
                counts[g["status"]] = counts.get(g["status"], 0) + 1
        for status, n in sorted(counts.items()):
            say("    %d x %s" % (n, STATUS_TEXT[status]))
        if counts.get("missed", 0) >= max(counts.get("slipped", 0), 1):
            say("Mostly 'closed on nothing': the fingers do not end up "
                "around the block. Compare the two GRIPPER ROOM lines "
                "above. If the room per side is smaller than the error "
                "left, more calibration will not help: use a narrower "
                "block or a wider jaw, or add a look-and-correct step "
                "before each descent.")
        elif counts.get("slipped"):
            say("Mostly 'lost during the hold': the fingers reach the "
                "block but do not keep it. Look at the finger pads (slip) "
                "and at how deep the upper fingertip gets (lines above).")
        say("Send the full output and the photos in %s/." % args.image_dir)
        return result

    result["ok"] = True
    check, train, other = training_commands(args, cfg, max_extent)
    result["commands"] = {"check": check, "train": train, "flags": other}
    say("\nREADY. The server already uses these values and has saved them "
        "to %s. They are loaded at every server start and win over "
        "server_env.sh." % saved.get("file"))
    if blind_note:
        say("Reminder: " + blind_note)
    if max_extent > 0.0851:
        say("The red block is %.0f mm long, so the commands carry "
            "--target_max_extent %.2f (without it the block is taken for "
            "a pile and never becomes the target)."
            % (block_dims[0] * 1000, max_extent))
    say("\nA) Set up the full scene (red block + obstacles, all %.0f-%.0f "
        "cm from the robot centre), then a short random run to watch that "
        "every motion is safe:\n" % (ws["band"][0] * 100,
                                     ws["band"][1] * 100))
    say(check)
    say("\nB) Training:\n")
    say(train)
    say("\nC) The same numbers for the other tools (grasp_once.py, "
        "verify_perception.py, diag_grasp.py):\n")
    say("  " + other)
    say("")
    return result


def undo(robot):
    """After a stop: arm home, and the server's numbers as they were when
    this program started, so a half-finished calibration is never used."""
    try:
        robot.home()
    except Exception:
        pass
    start = getattr(robot, "start_cfg", None)
    if start:
        try:
            robot.set(start)
            say("The server's values were put back to what they were "
                "before this run. Nothing was saved.")
        except Exception as error:
            say("WARNING: could not put the server's values back (%s). "
                "Restart the server on the robot before doing anything "
                "else." % error)


def to_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure table, reach and gripper; configure the robot "
                    "server; verify one-frame grasps; print the training "
                    "command.")
    parser.add_argument("--host", default="locobot.local")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--quick", action="store_true",
                        help="daily check: table + workspace + 2 grasps, "
                             "reusing the arm's numbers from the last full "
                             "run")
    parser.add_argument("--jaw_open", type=float, default=None,
                        help="inner opening of the fully open jaw, METRES, "
                             "measured with a ruler (asked for once if not "
                             "given)")
    parser.add_argument("--pitches", default="1.0",
                        help="gripper pitches to scan (1.57 = straight "
                             "down). Default: only the pitch in use; add "
                             "1.2,1.4 to compare, at the price of more "
                             "extended holds for the elbow motor")
    parser.add_argument("--table_hint", type=float, default=None,
                        help="table top height above the FLOOR in metres, "
                             "from a tape measure. Needed for a very low "
                             "table (below %.2f m), which would otherwise "
                             "be mistaken for the floor" % TABLE_MIN_Z)
    parser.add_argument("--box", default=None,
                        help="work area by hand, metres in the robot frame: "
                             "x_min,x_max,y_min,y_max (x forward, y left). "
                             "For a floor-level board the camera cannot "
                             "see in full. Measure from the robot's centre "
                             "with a tape")
    parser.add_argument("--skip_reach", action="store_true",
                        help="do not scan the arm's reach (step 3); use "
                             "--max_radial instead. For a motor that "
                             "overloads during the scan")
    parser.add_argument("--max_radial", type=float, default=0.34,
                        help="with --skip_reach: farthest distance from "
                             "the robot centre the gripper may be sent to "
                             "(m). 0.34 is a safe value for this arm at "
                             "pitch 1.0; the scan measured 0.38 on the old "
                             "table")
    parser.add_argument("--pitch", type=float, default=None,
                        help="use this pitch instead of the automatic "
                             "choice")
    parser.add_argument("--trials", type=int, default=None,
                        help="test grasps (default 5, or 2 with --quick)")
    parser.add_argument("--hover", type=float, default=0.05,
                        help="fingertip height above the table while "
                             "photographing the closed jaw (m)")
    parser.add_argument("--settle", type=float, default=1.0,
                        help="seconds to wait before a photo")
    parser.add_argument("--image_dir", default="calib_images")
    parser.add_argument("--out", default="calibration_result.json")
    args = parser.parse_args(argv)

    robot = Robot(args.host, args.port)
    result = {"ok": False}
    code = 1
    try:
        result = run(args, robot)
        code = 0 if result.get("ok") else 2
    except Stop as stop:
        say("\nSTOPPED: %s" % stop)
        result["stopped"] = str(stop)
        undo(robot)
    except KeyboardInterrupt:
        say("\ninterrupted")
        undo(robot)
        say("Because the run was cut off in the middle of a command, "
            "restart the server on the robot before the next run.")
    finally:
        try:
            with open(args.out, "w") as handle:
                json.dump(result, handle, indent=1, default=to_json)
            say("details written to %s" % args.out)
        except Exception as error:
            say("(could not write %s: %s)" % (args.out, error))
        robot.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
