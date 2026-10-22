# -*- coding: utf-8 -*-
"""measure_workspace.py

Measure the working area for the RIGHT-SIDE table layout. Run on the
training machine (Jetson) while the env server is up. The robot does not
move.

What it does:
  1. grabs one full frame from the server (real intrinsics + TF),
  2. finds the table height from the strongest 1 cm z bin,
  3. detects the blocks at each candidate height with the colour backend,
  4. prints paste-ready environment variables for the robot server and
     paste-ready flags for the training script,
  5. prints a ruler check so the TF can be verified against the real world.

Run it again after ANY of: moving the robot, moving the table, changing
the camera pan/tilt, or changing the block layout. TABLE_Z only changes
when the table or the robot's ride height changes; the workspace box and
anchor change whenever the robot or the blocks move.

Usage:
    python3 measure_workspace.py --host locobot.local
"""

import argparse
import json
import socket

import numpy as np

import route2_perception as R

ARM_REACH_MAX = 0.48   # beyond this the 5-DOF arm cannot grasp reliably
ARM_REACH_MIN = 0.18


def fetch_frame(host, port, go_home=False):
    sock = socket.create_connection((host, port), timeout=120)
    rfile = sock.makefile("r")

    def send(msg):
        sock.sendall((json.dumps(msg) + "\n").encode())
        return json.loads(rfile.readline())

    assert send({"cmd": "ping"}).get("ok"), "server ping failed"
    if go_home:
        print("moving the arm to the episode home first (--home)")
        send({"cmd": "home"})
    ee = send({"cmd": "ee"}).get("ee")
    frame = R.decode_frame(send({"cmd": "frame"}))
    sock.close()
    return frame, ee


HOVER_ABOVE = 0.09     # a resting block's centroid is never this far
                       # above the table; anything higher is hovering
                       # (the wrist, a cable) and not a scene object
ARM_RADIUS = 0.12      # tighter than the training-side 0.18 on purpose:
                       # here a wrong exclusion silently skews the printed
                       # workspace, and the hover rule above already does
                       # the main work of removing the arm


def split_scene_and_arm(dets, ee, table_z):
    """Separate real table objects from the arm / hovering junk, so the
    suggested workspace is built from the scene only. The hover rule runs
    first because height is what actually distinguishes the wrist blob; the
    EE-distance rule only mops up low arm parts very close to the gripper."""
    scene, dropped = [], []
    for d in dets:
        reason = None
        if d["centroid"][2] > table_z + HOVER_ABOVE:
            reason = "hovering %.2f m above the table (arm or cable)" % (
                d["centroid"][2] - table_z)
        elif ee is not None and np.linalg.norm(
                np.asarray(d["centroid"][:2])
                - np.asarray(ee, dtype=float)[:2]) < ARM_RADIUS:
            reason = "at the end effector (the arm)"
        (dropped if reason else scene).append((d, reason))
    return [d for d, _ in scene], dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="locobot.local")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--margin", type=float, default=0.12,
                        help="workspace margin around the blocks, metres")
    parser.add_argument("--home", action="store_true",
                        help="move the arm to the episode home first, so it "
                             "is out of the view (the only motion this "
                             "script can cause)")
    args = parser.parse_args()

    frame, ee = fetch_frame(args.host, args.port, go_home=args.home)
    if ee is not None:
        print("end effector at [%.3f, %.3f, %.3f]" % tuple(ee))
    pts, _ = R.backproject(frame["depth"], frame["intrinsics"],
                           frame["cam_to_base"], stride=2, rgb=frame["rgb"])
    keep = ((pts[:, 0] > -0.2) & (pts[:, 0] < 0.9) &
            (pts[:, 1] > -1.0) & (pts[:, 1] < 0.0) &
            (pts[:, 2] > -0.1) & (pts[:, 2] < 0.6))
    pts = pts[keep]
    print("right-side points: %d" % len(pts))
    if len(pts) < 1000:
        print("Too few points on the right side. Is the camera aimed at "
              "the table? Send this output back.")
        return

    bins = np.arange(pts[:, 2].min(), pts[:, 2].max() + 0.01, 0.01)
    hist, edges = np.histogram(pts[:, 2], bins=bins)
    peaks = np.argsort(hist)[::-1][:4]
    print("\ncandidate table heights:")
    for k in peaks:
        inl = pts[np.abs(pts[:, 2] - edges[k]) < 0.012]
        print("  z=%.3f  %6d pts  x %.2f..%.2f  y %.2f..%.2f"
              % (edges[k], hist[k],
                 np.percentile(inl[:, 0], 2), np.percentile(inl[:, 0], 98),
                 np.percentile(inl[:, 1], 2), np.percentile(inl[:, 1], 98)))

    best = None
    for k in peaks:
        z0 = float(edges[k])
        cfg = R.PerceptionConfig(ws_x=(-0.1, 0.8), ws_y=(-0.95, -0.05),
                                 table_z=z0, z_max=z0 + 0.25, ws_margin=0.0)
        dets = R.GeometricBackend(cfg).detect(frame)
        labels = sorted(d["label"] for d in dets)
        print("\n-- table_z=%.3f -> %d objects: %s" % (z0, len(dets), labels))
        too_far = False
        for d in dets:
            reach = float(np.hypot(d["centroid"][0], d["centroid"][1]))
            note = ""
            if reach > ARM_REACH_MAX:
                note, too_far = "  <-- TOO FAR for the arm", True
            elif reach < ARM_REACH_MIN:
                note = "  <-- too close"
            print("   %-13s [%.3f, %.3f, %.3f]  reach=%.2f m%s"
                  % (d["label"], d["centroid"][0], d["centroid"][1],
                     d["centroid"][2], reach, note))
        # A height qualifies as soon as it shows coloured objects at all;
        # target identification is the outlier rule, not a fixed colour.
        if best is None and len(R.colored(dets)) >= 1:
            best = (z0, dets, too_far)

    if best is None:
        print("\nNo coloured object was found at any "
              "height. Check lighting and the "
              "scene, then send this output back.")
        return

    z0, dets, too_far = best
    dets, dropped = split_scene_and_arm(dets, ee, z0)
    for d, reason in dropped:
        print("\nexcluded from the suggestion: %s at [%.2f, %.2f, %.2f] - %s"
              % (d["label"], d["centroid"][0], d["centroid"][1],
                 d["centroid"][2], reason))
    if not R.colored(dets):
        print("\nAfter excluding the arm, no coloured object is left. Is "
              "the gripper parked on top of the scene? Move the arm "
              "(--home) and run again.")
        return
    if len(dets) < 3:
        print("\nNOTE: only %d object(s) detected. Fine for a measurement "
              "run; the full scene needs 1 red + 2 green blocks." % len(dets))
    if too_far:
        print("\nWARNING: at least one block is beyond the arm's reach "
              "(%.2f m). Move the ROBOT closer to the table (or the blocks "
              "closer to the robot), then run this script again. The "
              "numbers below describe the CURRENT, partly unreachable "
              "layout." % ARM_REACH_MAX)

    xs = [d["centroid"][0] for d in dets]
    ys = [d["centroid"][1] for d in dets]
    outlier = R.hue_outlier(dets)
    if outlier is not None:
        red = dets[outlier]
        print("\ntarget by colour-outlier rule: %s (hue %.0f deg) at "
              "[%.2f, %.2f] - the one block whose colour differs from all "
              "the others." % (red["label"], red.get("hue", -1),
                              red["centroid"][0], red["centroid"][1]))
    else:
        cols = R.colored(dets)
        seeded = [d for d in cols if R.detection_hue_match(
            d, R.LABEL_HUES["red_object"], 35.0) is not None]
        red = (min(seeded, key=lambda d: R.hue_distance(
                   d["hue"], R.LABEL_HUES["red_object"]))
               if seeded else max(cols, key=lambda d: d["n_points"]))
        print("\nnote: fewer than 3 coloured objects, so the outlier rule "
              "cannot run yet; using %s as the anchor reference. "
              "Calibration numbers below are valid either way. Place the "
              "full scene (target + 2 same-coloured obstacles) before "
              "verify_perception."
              % ("the reddish block" if seeded
                 else "the biggest coloured object"))
    m = args.margin
    print("\n================ SUGGESTED SETTINGS ================")
    print("robot terminal (before restarting the server):")
    print("  export TABLE_Z=%.3f" % z0)
    print("  export WS_X_MIN=%.2f WS_X_MAX=%.2f" % (min(xs) - m, max(xs) + m))
    print("  export WS_Y_MIN=%.2f WS_Y_MAX=%.2f" % (min(ys) - m, max(ys) + m))
    print("  export WS_Z_MAX=%.3f" % (z0 + 0.20))
    print("  export TARGET_X=%.2f TARGET_Y=%.2f"
          % (red["centroid"][0], red["centroid"][1]))
    print("Jetson training flags:")
    print("  --table_z %.3f --ws_x_min %.2f --ws_x_max %.2f"
          " --ws_y_min %.2f --ws_y_max %.2f --anchor \"%.2f,%.2f\""
          % (z0, min(xs) - m, max(xs) + m, min(ys) - m, max(ys) + m,
             red["centroid"][0], red["centroid"][1]))
    print("\nRULER CHECK: the target block should be %.2f m forward(+x) and "
          "%.2f m to the RIGHT of the robot centre. Measure it; the error "
          "must be under 2 cm before anything else runs."
          % (red["centroid"][0], -red["centroid"][1]))


if __name__ == "__main__":
    main()
