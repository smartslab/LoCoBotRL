# -*- coding: utf-8 -*-
"""
grasp_env_server.py

Runs ON THE ROBOT inside the pyrobot Python 3.6 env:
    load_pyrobot_env
    python ~/grasp_env_server.py

Requires in another terminal:
    roslaunch locobot_control main.launch use_arm:=true use_camera:=true

Sequential clutter grasping only. There is no reach/pretraining mode.

At reset, the point-cloud cluster nearest TARGET_X/TARGET_Y is designated as
the target block. Every policy decision executes one complete manipulation
primitive and then re-observes the scene:
  push  : select an obstacle and push it in a chosen XY direction.
  grasp : attempt a scripted top-down grasp of the target, but only when the
          measured target clearance is large enough.

Protocol: newline-delimited JSON over TCP (port 9999).
  {"cmd":"ping"}                          -> {"ok": true}
  {"cmd":"reset"}                         -> state + synchronized RGB-D
  {"cmd":"step","action":[5 floats],
   "perception": optional dict}           -> {"obs":[...], "reward":r,
                                              "done":bool, "info":{...}}
      When "perception" is present, the server overwrites its internal
      target/obstacle state with the client-supplied values BEFORE running
      the primitive. The client (training machine) computes these from the
      full frame with YOLO segmentation. Format:
        {"target": {"centroid":[x,y,z], "extent":[ex,ey,ez]} or null,
         "obstacles": [{"centroid":[...], "extent":[...]}, ...]}
      In this external mode the server does NOT re-run its own point-cloud
      perception after the primitive, and the returned reward contains only
      the terms the server can compute (push cost, motion failures, gripper
      result). The client adds the perception-based terms (clearance gain,
      obstacle removal, target-lost penalty, success confirmation) itself.
  {"cmd":"ee"}                            -> current EE position (debug)
  {"cmd":"goto","xyz":[x,y,z]}            -> move EE there, pitch down (debug)
  {"cmd":"goto_pregrasp"}                 -> move EE to the pre-grasp pose
  {"cmd":"grasp_test"}                    -> run the scripted grasp once
  {"cmd":"pc_stats"}                      -> capture point cloud, return
                                             object centroid/extent stats
  {"cmd":"rgbd_stats"}                    -> inspect camera streams without
                                             moving the robot
  {"cmd":"frame"}                         -> full-resolution synchronized
                                             RGB + aligned depth, plus the
                                             color intrinsics and the 4x4
                                             base_link<-camera transform.
                                             The training machine runs YOLO
                                             segmentation on this frame and
                                             back-projects the masks to get
                                             object centroids in the base
                                             frame. Does not move the robot.
  {"cmd":"capture_home"}                   -> make the current 5 arm joint
                                             angles the episode home
  {"cmd":"home"}                           -> move to the episode home
  {"cmd":"pc_save"}                       -> capture and dump raw + cropped
                                             cloud to ~/pc_dumps/ as .npy
  {"cmd":"get_config"}                    -> every calibratable value
  {"cmd":"set_config","values":{...}}     -> change values at run time
                                             (range-checked, all or none)
  {"cmd":"save_config","meta":{...}}      -> write them to CALIB_FILE; the
                                             file is loaded at every start
                                             and wins over server_env.sh
  {"cmd":"frame","raw":true}              -> the same frame without the
                                             static-background mask
  {"cmd":"set_background","depth":{...}}  -> store the depth of the static
                                             background (arm at home,
                                             cables), see BACKGROUND_FILE
  {"cmd":"close"}                         -> shut the server down

Observation (45-D):
  joints (5) + EE pos (3) + gripper flag (1)
  + target centroid (3) + target extent (3) + visible flag (1)
  + target clearance (1)
  + four obstacle slots, each centroid (3) + extent (3) + mask (1).
The point cloud is refreshed after every push or grasp attempt.
Point cloud processing is numpy only: crop to the workspace box, drop
table points by z threshold, centroid + axis-aligned extent of the rest.

Visual observation:
  The server uses direct rospy.Subscriber callbacks for the color and depth
  Image topics, pairs their latest frames by timestamp, center-crops and
  downsamples each stream, and sends RGB uint8 + depth uint16-mm arrays as
  zlib/base64 packets under response["rgbd"].  The depth stream is the aligned
  depth (/camera/aligned_depth_to_color/image_raw), reprojected into the color
  optical frame so its pixels line up with the RGB image; point-cloud
  reconstruction therefore uses the color camera_info topic and the depth
  message's frame_id, which is the color optical frame.

Action (5-D), all values in [-1,1]:
  action[0] < 0  -> push; action[1] selects one obstacle;
                    action[2:4] is the XY push direction and action[4]
                    controls push distance.
  action[0] >= 0 -> grasp the target; other fields are ignored.
The server executes safe high-level primitives; the policy never directly
commands the gripper or raw joint deltas.

This server is a DUMB EXECUTOR:
  * It always executes the chosen primitive. There is no graspable gate; a
    grasp with no visible target is a blind attempt that simply fails.
  * It computes NO reward. Every step returns reward = 0.0 for protocol
    compatibility only. The training client computes the reward from the
    gripper result plus its own route-2 (YOLO/DINOv2) visual check
    ("gripper held AND target gone from the table") and decides episode
    termination. The server's ``done`` flag reflects only the HORIZON budget.
  * The target/obstacle coordinates used by the primitives come from the
    client's route-2 perception, supplied per step:
        {"cmd":"step","action":[5 floats],"perception":{...}}
    where perception = {"target":{"centroid":[x,y,z],"extent":[x,y,z]},
                        "obstacles":[{"centroid":..,"extent":..}, ...]}.
    With no perception field the server falls back to its own point cloud.
"""

from __future__ import print_function

import base64
import json
import os
import socket
import threading
import time
import traceback
import zlib

import numpy as np
from pyrobot import Robot

# ---------------- network ----------------
HOST = "0.0.0.0"
PORT = 9999

# ---------------- synchronized ROS RGB-D stream ----------------
# These defaults match the actual topics this RealSense publishes (checked with
# `rostopic list | grep camera`).
#   RGB   : /camera/color/image_raw
#   depth : /camera/aligned_depth_to_color/image_raw
# The aligned depth stream is reprojected into the COLOR camera frame, so its
# pixels line up one-to-one with the RGB image. Because of this alignment,
# point-cloud reconstruction must use the COLOR intrinsics
# (/camera/color/camera_info), NOT the raw depth intrinsics, and the transform
# to base_link uses the depth message frame_id, which for the aligned stream is
# the color optical frame. See DEPTH_INFO_TOPIC below.
# Override at launch if your setup differs, e.g.
#   DEPTH_TOPIC=/camera/depth/image_rect_raw python ~/grasp_env_server_rgbd.py
RGB_TOPIC = os.environ.get("RGB_TOPIC", "/camera/color/image_raw")
DEPTH_TOPIC = os.environ.get(
    "DEPTH_TOPIC", "/camera/aligned_depth_to_color/image_raw")
# Intrinsics for the stream that DEPTH_TOPIC carries. The aligned depth stream
# lives in the color frame, so its correct intrinsics are the color ones.
DEPTH_INFO_TOPIC = os.environ.get(
    "DEPTH_INFO_TOPIC", "/camera/color/camera_info")
RGBD_SIZE = int(os.environ.get("RGBD_SIZE", "84"))
RGBD_SYNC_SLOP = float(os.environ.get("RGBD_SYNC_SLOP", "0.08"))
RGBD_TIMEOUT = float(os.environ.get("RGBD_TIMEOUT", "10.0"))


def env_flag(name, default=False):
    value = os.environ.get(name, "1" if default else "0")
    return value.strip().lower() in ("1", "true", "yes", "on")


def parse_joint_vector(text):
    """Parse five comma-separated LoCoBot arm joint angles in radians."""
    if not text:
        return None
    values = np.asarray(
        [float(value.strip()) for value in text.split(",")], dtype=float)
    if values.size != 5 or not np.all(np.isfinite(values)):
        raise ValueError(
            "ARM_HOME_JOINTS must contain exactly 5 finite angles in radians")
    return values

# ---------------- safety ---------------------------------------------------
# If the values in your existing reach_env_server.py differ, copy those over.
# SAFE_Z_MIN / HARD_Z_MIN are derived from TABLE_Z further down, so a new
# support surface only needs the TABLE_Z environment variable and the z
# safety floors follow it. With the historical TABLE_Z=0.225 the derived
# values are exactly the old hand-set ones (0.25 / 0.22). Both can still be
# overridden individually through the environment.
MAX_DELTA = 0.08      # rad per joint per step

# Conservative joint box (rad). Port from your existing server if different.
J_MIN = np.array([-1.5, -1.2, -1.2, -1.6, -1.5])
J_MAX = np.array([1.5, 1.2, 1.4, 1.6, 1.5])

# ---------------- episode home pose ---------------------------------------
# Default behavior: after roslaunch is running and the motors have been
# re-torqued, start this server while the arm is in the desired rotated/tucked
# pose. The current encoder readings are captured once and every reset returns
# to that pose instead of calling PyRobot's fixed go_home().
#
# For a repeatable pose across restarts, copy the five angles printed at server
# startup and launch with, for example:
#   CAPTURE_START_AS_HOME=0 \
#   ARM_HOME_JOINTS="-1.45,-0.20,0.55,0.90,0.00" python ...
CAPTURE_START_AS_HOME = env_flag("CAPTURE_START_AS_HOME", True)
ARM_HOME_JOINTS_TEXT = os.environ.get("ARM_HOME_JOINTS", "").strip()
ARM_HOME_PLAN = env_flag("ARM_HOME_PLAN", True)

# ---------------- task geometry (MEASURE THESE ON SITE) ----------------
# At reset, the sane point-cloud cluster nearest this marked location becomes
# the target. Keep the target block near this anchor when arranging a scene.
TARGET_X = float(os.environ.get("TARGET_X", "0.40"))
TARGET_Y = float(os.environ.get("TARGET_Y", "0.03"))
TARGET_TOP_Z = float(os.environ.get("TARGET_TOP_Z", "0.31"))
TARGET_TRACK_MAX = 0.10
# Largest xy size the target may have and still count as one block. Part of
# the stored calibration (TARGET_MAX_EXTENT), so the 9 cm bar is set once.
TARGET_MAX_XY_EXTENT = float(os.environ.get("TARGET_MAX_EXTENT", "0.085"))
MAX_OBSTACLES = 4
# Radius of the "no obstacle point allowed" circle drawn at the target's
# xy centroid. 0.06 m = target half-width (~0.015) + gripper half-opening
# (~0.022) + finger thickness (~0.010) + a 0.013 safety margin.
GRASP_CLEARANCE = float(os.environ.get("GRASP_CLEARANCE", "0.06"))

# Heights of the scripted primitives, all relative, all tunable through the
# environment. This 5-DOF arm cannot hold a near-vertical gripper at every
# height: pointing the gripper down puts the wrist ~15 cm above the
# fingertips, so a high hover with a vertical pitch runs out of arm before
# the fingertips get there. When the table is raised, every absolute hover
# height rises with it and can leave the feasible envelope; that is exactly
# what happened when TABLE_Z went from 0.225 to 0.266 and all pre-grasp
# (top + 0.08) and push-approach (0.34) poses were refused by IK at radial
# distances that grasp fine when lower. Use reach_scan.py to measure the
# feasible (radial, height) envelope, then set these.
PREGRASP_DZ = float(os.environ.get("PREGRASP_DZ", "0.08"))
GRASP_DZ = float(os.environ.get("GRASP_DZ", "-0.03"))
LIFT_DZ = float(os.environ.get("LIFT_DZ", "0.10"))

GRASP_PITCH = float(os.environ.get("GRASP_PITCH", "1.4"))
GRASP_ROLL = float(os.environ.get("GRASP_ROLL", "0.0"))
# Finger alignment. The client sends each object's principal axis ("yaw",
# radians in the base frame, from a PCA of its points). For a parallel-jaw
# grasp the fingers must close ACROSS the short side, i.e. perpendicular to
# that axis. The roll needed for that depends on the gripper convention:
#     roll = GRASP_ROLL_SIGN * (closing_dir - arm_azimuth) + GRASP_ROLL_BASE
# with closing_dir = yaw + pi/2 and arm_azimuth = atan2(y, x) of the target.
# The sign and base are found once on the robot with grasp_once.py (a
# block placed radially, then tangentially; both must be grasped). Mode
# "off" keeps the fixed GRASP_ROLL.
GRASP_ROLL_MODE = os.environ.get("GRASP_ROLL_MODE", "auto")
GRASP_ROLL_SIGN = float(os.environ.get("GRASP_ROLL_SIGN", "1.0"))
# pi/2 is required at pitch=1.0. roll and the fingertip height pair are
# coupled by the forward tilt: BASE=pi/2 gives tangential closing (both
# fingers at the same world height) for a radial bar; BASE=0 gives radial
# closing with a 24 mm height difference, and the lower finger goes into
# the table. The Oct 2 success and the Oct 20 crash both confirm this.
GRASP_ROLL_BASE = float(os.environ.get("GRASP_ROLL_BASE", str(np.pi / 2.0)))
# The policy owns the final roll. action[5] is an offset in
# [-GRASP_ROLL_RANGE, +GRASP_ROLL_RANGE] added to the geometric default
# (perpendicular to the target's long axis). The default is a starting
# point, not a constraint: in clutter the best approach angle is often not
# the perpendicular one, and roll is a real degree of freedom of the
# gripper, so it belongs in the action space and not in a hand-tuned
# constant. Zero range pins the roll to the geometric default.
GRASP_ROLL_RANGE = float(os.environ.get("GRASP_ROLL_RANGE", "0.785"))
# Systematic offset between where perception says the target is and where
# the fingers actually have to go. Calibrate ONCE with grasp_tune.py: hover
# over an isolated block, find the dx/dy that puts the fingers either side
# of it, put them here. This is a constant of the setup (camera extrinsics,
# tool centre point), not something the policy should have to rediscover on
# every episode at 15 seconds a try.
GRASP_XY_OFFSET = np.array([
    float(os.environ.get("GRASP_DX", "0.0")),
    float(os.environ.get("GRASP_DY", "0.0"))])
# The policy may also shift the grasp point itself: action[6], action[7]
# are scaled to +-GRASP_XY_RANGE metres and added to the perceived centroid.
# Without these dimensions a policy cannot correct an aiming error at all,
# no matter how long it trains - there is simply no action for "move a
# centimetre left". With them, a failed grasp can teach the policy where
# the fingers actually have to go, which is the point of learning the
# grasp on the real robot instead of calibrating it.
GRASP_XY_RANGE = float(os.environ.get("GRASP_XY_RANGE", "0.03"))
# Residual correction: after a goto, the arm's own FK is read back and, if
# it stopped short of the commanded point by more than this, the command
# is re-issued once with the residual added. Measured on the robot: the
# controller reliably stops 19-21 mm short at radial >= 0.30, which is one
# finger's width on a 3 cm block.
GOTO_RESIDUAL_TOL = float(os.environ.get("GOTO_RESIDUAL_TOL", "0.008"))
# Where a held object is put back on the table at reset, so a successful
# episode resets itself without a human replacing the block. Defaults to
# the anchor. RELEASE_DZ is the fingertip height above the table when the
# gripper opens (a short drop keeps the block upright).
RELEASE_X = float(os.environ.get("RELEASE_X", str(TARGET_X)))
RELEASE_Y = float(os.environ.get("RELEASE_Y", str(TARGET_Y)))
RELEASE_DZ = float(os.environ.get("RELEASE_DZ", "0.05"))

PUSH_MIN_DISTANCE = float(os.environ.get("PUSH_MIN_DISTANCE", "0.025"))
# Largest push. 10 cm was enough to scatter a pile of 3 cm blocks in one
# shove; 6 cm keeps a push a nudge.
PUSH_MAX_DISTANCE = float(os.environ.get("PUSH_MAX_DISTANCE", "0.06"))
# The sweep is executed as short segments with a brief pause between them,
# which caps the speed at contact and lets a block slide instead of fly.
PUSH_SEGMENT = float(os.environ.get("PUSH_SEGMENT", "0.02"))
PUSH_SEGMENT_PAUSE = float(os.environ.get("PUSH_SEGMENT_PAUSE", "0.25"))
PUSH_APPROACH_DZ = float(os.environ.get("PUSH_APPROACH_DZ", "0.07"))
# How far behind the object's face the closed fingers are put down.
PUSH_TOOL_OFFSET = float(os.environ.get("PUSH_TOOL_OFFSET", "0.025"))
# 1 (default): a push moves the object by the commanded distance. 0: the old
# stroke, which went through the object first. See scripted_push.
PUSH_FROM_FACE = env_flag("PUSH_FROM_FACE", True)
# ---- gentle push (build 2026-10-09b) ----
# This arm is position controlled. PyRobot has no speed or force setting
# for it ("Velocity control for locobot not supported yet"), so a push
# cannot be made weaker by turning a force down. What CAN be controlled is
# how the fingertips are moved while they touch something:
#   * the jaw is closed before the arm comes down;
#   * from 1 cm before the object's face the fingertips advance in steps of
#     PUSH_STEP, with a stop of PUSH_STEP_PAUSE after each. A servo that
#     is asked for 4 mm never gets up to speed, and an object that is
#     struck is driven for 4 mm at most before the arm stands still again;
#   * no "you stopped short, go further" re-command while pushing. That
#     re-command is right in free air and wrong against an object: it
#     answers resistance by pushing harder;
#   * if the fingertips stop advancing for PUSH_STALL_STEPS steps in a row
#     the push ends there (something does not give way) and the arm lifts.
# These replace PUSH_SEGMENT / PUSH_SEGMENT_PAUSE, which are read no more.
PUSH_STEP = float(os.environ.get("PUSH_STEP", "0.004"))
PUSH_STEP_PAUSE = float(os.environ.get("PUSH_STEP_PAUSE", "0.10"))
PUSH_STALL_STEPS = int(os.environ.get("PUSH_STALL_STEPS", "3"))
# Oct 29g: the jaw is closed once, before the first push, and then STAYS
# closed: no open between pushes, at home or at an episode reset. On Oct
# 29 the gripper motor (ID 7) tripped "Overload error" three times, each
# time 4.4 s after an OPEN command, the last time with the arm already at
# home and nothing near the fingers. Closing never tripped it, and a
# closed jaw rests at zero power (/gripper/state = 3). So during push-only
# training (--allow_pick 0) the jaw is simply never opened. A grasp
# primitive still opens it as before. JAW_STAYS_CLOSED=0 brings back the
# 29f behaviour (open at home after every push).
JAW_STAYS_CLOSED = env_flag("JAW_STAYS_CLOSED", True)
# Half thickness of the closed jaw plus a margin: the fingers are not put
# down closer than this to any OTHER object.
PUSH_FINGER_CLEARANCE = float(os.environ.get("PUSH_FINGER_CLEARANCE", "0.012"))
# PUSH_EE_Z (fingertip sweep height) is derived from TABLE_Z further down:
# it used to be the absolute 0.27, tuned for a 0.225 table (4.5 cm gap). A
# raised table silently turned that into a 4 mm gap and the push scraped
# the surface.
PUSH_TARGET_MARGIN = 0.025
# Refuse a push that would drive an obstacle into the target. OFF by
# default (Oct 28): pushing an obstacle into the red block is a bad move
# for the TASK, not a danger to the robot, so it is left to the reward
# (the client charges "hit_target" and the lost clearance) and the policy
# learns it. The refusals that remain are the ones that protect the robot:
# out of reach, fingers landing on another object, object pushed off the
# table edge or into the inner limit. PROTECT_TARGET=1 brings it back.
PROTECT_TARGET = env_flag("PROTECT_TARGET", False)
# Table contact during a push: the fingertips (from the arm's own encoders)
# are this much higher than commanded -> the table is holding them up.
# The stroke stops and the arm lifts. Oct 28: a stroke at 1 cm scraped the
# table from one side to the other and an overloaded motor shut the robot
# down.
PUSH_CONTACT_TOL = float(os.environ.get("PUSH_CONTACT_TOL", "0.008"))
# The stroke stops when the fingertips (arm's encoders) come closer than
# this to the table top.
PUSH_TABLE_GAP = float(os.environ.get("PUSH_TABLE_GAP", "0.004"))
# A push may START this far beyond MAX_RADIAL (the IK decides the rest).
PUSH_START_SLACK = float(os.environ.get("PUSH_START_SLACK", "0.03"))
# Largest upward correction the stroke may add to hold its height.
PUSH_MAX_LIFT = float(os.environ.get("PUSH_MAX_LIFT", "0.03"))

# ---------------- point cloud (base frame, metres) ----------------
# Manual camera is the default. The server then never commands pan/tilt; use
# the /locobot/torque_enable service to release and re-lock the camera joints.
# Set CAMERA_AUTO_POSE=1 only when a repeatable commanded pose is desired.
CAMERA_AUTO_POSE = env_flag("CAMERA_AUTO_POSE", False)
CAM_PAN = float(os.environ.get("CAM_PAN", "-1.5708"))  # optional 90 deg right
CAM_TILT = float(os.environ.get("CAM_TILT", "0.70"))
TABLE_Z = float(os.environ.get("TABLE_Z", "0.225"))

# ---------------- arm frame vs camera frame ----------------------------
# Everything in this server is reasoned in the CAMERA/TF frame: TABLE_Z,
# the workspace box, the z safety floors, every perceived centroid. The
# arm's own kinematics live in a frame whose z does NOT agree with that -
# measured on the robot, the arm believes the table is about 0.10 m higher
# than the camera says. Commanding camera-frame heights straight to the
# arm therefore drove the gripper below the table top and shoved the table.
#
# ARM_Z_OFFSET is the one number that reconciles them:
#     z_arm = z_camera + ARM_Z_OFFSET
# It is applied in exactly two places, on the way out in _goto_xyz and on
# the way in in _ee_pos, so nothing else in the code has to know two frames
# exist. Calibrate it with z_calibrate.py (hover at a commanded height,
# measure the fingertip-to-table gap with a ruler, solve).
SERVER_BUILD = "2026-10-29g"
ARM_Z_OFFSET = float(os.environ.get("ARM_Z_OFFSET", "0.0"))
# The same reconciliation, extended. calibrate_all.py photographs the
# fingertips against a block at several places on the table and fits the
# simplest correction that explains what it sees:
#     z_arm  = z_camera  + ARM_Z_OFFSET + ARM_Z_SLOPE * (radial - ARM_R_REF)
#     xy_arm = xy_camera + [ARM_DX, ARM_DY] + ARM_DR * r_hat + ARM_DT * t_hat
# r_hat points away from the robot at that xy, t_hat is r_hat turned 90 deg
# to the left. A constant shift (camera mounting) shows up in ARM_DX/DY, a
# reach error of the arm in ARM_DR. All default to 0, which is the old
# behaviour. They are applied in the same two places as ARM_Z_OFFSET, so
# EVERY motion (grasp, push, release) is corrected, not only the grasp.
ARM_Z_SLOPE = float(os.environ.get("ARM_Z_SLOPE", "0.0"))
ARM_R_REF = float(os.environ.get("ARM_R_REF", "0.30"))
ARM_DX = float(os.environ.get("ARM_DX", "0.0"))
ARM_DY = float(os.environ.get("ARM_DY", "0.0"))
ARM_DR = float(os.environ.get("ARM_DR", "0.0"))
ARM_DT = float(os.environ.get("ARM_DT", "0.0"))

# ---------------- tool centre point ------------------------------------
# The arm reports and accepts the position of its WRIST, not of the
# fingertips. The gripper hangs off the wrist along the tool axis, so with
# a pitch p (p = pi/2 is straight down) the fingertips sit
#     L*cos(p) further out radially   and   L*sin(p) lower
# than the wrist. Only the vertical part was ever compensated, by folding
# it into ARM_Z_OFFSET; the horizontal part was not, and at pitch 1.0 that
# is L*cos(1.0) = 7 cm of lateral error - the gripper hovered a hand's
# width to the side of the block on every single attempt.
#
# With GRIPPER_LENGTH set, every xyz in this server means the FINGERTIP
# position and the wrist target is computed from it, so any pitch works.
# L was derived from the calibration hover: the measured vertical error at
# pitch 1.0 was 0.110 m = L*sin(1.0), giving L = 0.131 m. ARM_Z_OFFSET then
# goes back to the pure frame offset (0.040 by tape measure), because the
# gripper's own drop is no longer hidden inside it.
GRIPPER_LENGTH = float(os.environ.get("GRIPPER_LENGTH", "0.0"))
# +1: the fingertips point away from the base (the usual reach-out pose).
# -1 flips it. Check with one hover: if the correction doubles the error
# instead of removing it, flip the sign.
GRIPPER_TCP_SIGN = float(os.environ.get("GRIPPER_TCP_SIGN", "1.0"))
# IK solver. PyRobot ships an ANALYTICAL solver for this 5-DOF arm and its
# own LoCoBot grasping example uses it (numerical=False, pitch=pi/2). This
# server used the numerical solver, which starts from the current joints
# and can fail to converge on poses that are perfectly reachable - a
# plausible cause of many of the "IK refused" lines seen so far. Default
# is now the analytical solver; set IK_NUMERICAL=1 to go back.
IK_NUMERICAL = os.environ.get("IK_NUMERICAL", "0") == "1"

# fingertip height during a push sweep: 4.5 cm above the surface, as before
PUSH_EE_Z = float(os.environ.get("PUSH_EE_Z", str(TABLE_Z + 0.045)))
# z safety floors, tied to the support surface (see the safety section above)
SAFE_Z_MIN = float(os.environ.get("SAFE_Z_MIN", str(TABLE_Z + 0.025)))
HARD_Z_MIN = float(os.environ.get("HARD_Z_MIN", str(TABLE_Z - 0.005)))
                        # support surface height: the PLATFORM top,
                       # measured from the point cloud dump (2026-08-07).
PC_Z_CUT = 0.012       # drop points within this height above the FITTED
                       # table plane. The plane fit absorbs the ~1.5cm tilt
                       # from the camera TF error, so the margin can be small
                       # enough to keep short blocks.
CLUSTER_VOXEL = 0.02   # voxel size for clustering; two objects closer than
                       # this may merge into one cluster
CLUSTER_MIN_PTS = 25   # discard clusters smaller than this
# base_link follows ROS convention: +x forward, +y left, so a table on the
# robot's right normally has negative y. All bounds are launch-time settings.
WS_X = (float(os.environ.get("WS_X_MIN", "0.28")),
        float(os.environ.get("WS_X_MAX", "0.60")))
WS_Y = (float(os.environ.get("WS_Y_MIN", "-0.25")),
        float(os.environ.get("WS_Y_MAX", "0.25")))
WS_Z_MAX = float(os.environ.get("WS_Z_MAX", "0.42"))
PC_MIN_POINTS = 30     # fewer than this after cropping -> no object seen

# Gripper state value that means "object held". PyRobot's mapping varies
# by version; run {"cmd":"grasp_test"} once with the can and once without,
# note the raw values printed, and set this accordingly.
HELD_STATE = 2

# ---------------- episode/reward ----------------
HORIZON = int(os.environ.get("HORIZON", "6"))   # clearing decisions per episode
# Success needs the object to stay in the gripper for this long after the
# lift; a slipping grasp reads held for a moment and then drops.
HOLD_SECONDS = float(os.environ.get("HOLD_SECONDS", "2.0"))
# Where a cleared obstacle is put down by the grasp-and-place primitive.
# Must be on the table, inside the reachable band, and away from the
# target area. Set it per scene.
# Reach limit of the arm for the push primitive at the configured pitch
# (from reach_scan.py). A push whose start is beyond it is refused; a push
# whose END is beyond it is shortened to end at the limit instead of being
# refused outright, so "push a bit further" still moves the object.
MAX_RADIAL = float(os.environ.get("MAX_RADIAL", "0.38"))
# Inner reach limit: nothing is pushed closer to the robot than this, and the
# target is not grasped closer than this. Objects at the near table edge are
# where the arm hit the table on its way home (Oct 2026). Measured by
# calibrate_all.py (closest usable distance) and stored with the rest.
MIN_RADIAL = float(os.environ.get("MIN_RADIAL", "0.22"))
# Before every move to home the fingertips first go straight up to this
# height above the table, so the joint-space move home starts clear of the
# table edge.
RETRACT_DZ = float(os.environ.get("RETRACT_DZ", "0.10"))
# Jaw opening of the gripper when fully open. Used to reason about a
# TILTED grasp: with the gripper pitched away from vertical, the two
# fingertips are not at the same height. If the jaw closes in the tilt
# plane, their height difference is OPENING * sin(tilt); at pitch 1.0 that
# is 4.5 cm * sin(33 deg) = 2.5 cm on a 3 cm block, so the upper finger
# skims the top face while the lower one is at the table. The planner
# therefore (a) prefers to close TANGENTIALLY, where both fingertips are at
# the same height, and (b) lowers the grasp so that the UPPER fingertip
# still reaches the block, without driving the lower one into the table.
GRIPPER_OPENING = float(os.environ.get("GRIPPER_OPENING", "0.045"))
# Lowest the LOWER fingertip may go, above the table.
FINGER_TABLE_MARGIN = float(os.environ.get("FINGER_TABLE_MARGIN", "0.004"))
DROP_X = float(os.environ.get("DROP_X", "0.20"))
DROP_Y = float(os.environ.get("DROP_Y", "-0.30"))
SETTLE_TIME = 0.7      # seconds to wait after a joint command
PUSH_COST = 0.20
CLEARANCE_GAIN_SCALE = 8.0
OBSTACLE_REMOVED_BONUS = 0.5
GRASP_SUCCESS_BONUS = 10.0
GRASP_FAILURE_PENALTY = 2.0
BLOCKED_GRASP_PENALTY = 0.75


# ---------------- runtime configuration -----------------------------------
# calibrate_all.py (on the Jetson) measures the table, the reach of the arm
# and the gripper, sends the numbers here with {"cmd": "set_config"} and has
# them written to CALIB_FILE with {"cmd": "save_config"}. At the next start
# the file is loaded again, so nobody has to edit server_env.sh by hand.
#
# ORDER OF PRECEDENCE: built-in default < server_env.sh < CALIB_FILE.
# The startup lines "[calib] NAME: a -> b" show every value the file changed.
# To ignore the file once: USE_CALIB_FILE=0 python ~/grasp_env_server_rgbd.py
CALIB_FILE = os.path.expanduser(
    os.environ.get("CALIB_FILE", "~/locobot_calibration.json"))
USE_CALIB_FILE = env_flag("USE_CALIB_FILE", True)
CALIB_META = {}
CALIB_SAVED_AT = None

# name -> (lowest, highest) accepted value. A value outside its range is
# refused, so a wrong measurement cannot send the arm somewhere absurd.
CONFIG_LIMITS = {
    "TABLE_Z": (-0.20, 0.60),
    "WS_X_MIN": (-0.30, 0.80), "WS_X_MAX": (-0.30, 0.80),
    "WS_Y_MIN": (-0.90, 0.90), "WS_Y_MAX": (-0.90, 0.90),
    "WS_Z_MAX": (-0.10, 0.90),
    "SAFE_Z_MIN": (-0.20, 0.70), "HARD_Z_MIN": (-0.20, 0.70),
    "TARGET_X": (-0.30, 0.80), "TARGET_Y": (-0.90, 0.90),
    "RELEASE_X": (-0.30, 0.80), "RELEASE_Y": (-0.90, 0.90),
    "RELEASE_DZ": (0.02, 0.12),
    "DROP_X": (-0.30, 0.80), "DROP_Y": (-0.90, 0.90),
    "ARM_Z_OFFSET": (0.0, 0.30), "ARM_Z_SLOPE": (-0.60, 0.60),
    "ARM_R_REF": (0.10, 0.50),
    "ARM_DX": (-0.05, 0.05), "ARM_DY": (-0.05, 0.05),
    "ARM_DR": (-0.05, 0.05), "ARM_DT": (-0.05, 0.05),
    "GRASP_DX": (-0.05, 0.05), "GRASP_DY": (-0.05, 0.05),
    "GRASP_PITCH": (0.60, 1.60),
    "PREGRASP_DZ": (0.015, 0.12), "GRASP_DZ": (-0.05, 0.0),
    "LIFT_DZ": (0.02, 0.15),
    "PUSH_EE_Z": (-0.20, 0.70), "PUSH_APPROACH_DZ": (0.02, 0.12),
    "MAX_RADIAL": (0.20, 0.50), "MIN_RADIAL": (0.10, 0.40),
    "RETRACT_DZ": (0.04, 0.20),
    "GRASP_CLEARANCE": (0.02, 0.15), "TARGET_MAX_EXTENT": (0.03, 0.20),
    "GRIPPER_OPENING": (0.02, 0.12),
    "FINGER_TABLE_MARGIN": (0.002, 0.03),
    "HOLD_SECONDS": (0.0, 10.0),
}
# Heights that are defined relative to the table. When TABLE_Z changes and
# one of these is not sent along, it moves by the same amount.
TABLE_RELATIVE = ("WS_Z_MAX", "SAFE_Z_MIN", "HARD_Z_MIN", "PUSH_EE_Z")


def current_config():
    """Every calibratable value as a flat dict."""
    return {
        "TABLE_Z": TABLE_Z,
        "WS_X_MIN": WS_X[0], "WS_X_MAX": WS_X[1],
        "WS_Y_MIN": WS_Y[0], "WS_Y_MAX": WS_Y[1], "WS_Z_MAX": WS_Z_MAX,
        "SAFE_Z_MIN": SAFE_Z_MIN, "HARD_Z_MIN": HARD_Z_MIN,
        "TARGET_X": TARGET_X, "TARGET_Y": TARGET_Y,
        "RELEASE_X": RELEASE_X, "RELEASE_Y": RELEASE_Y,
        "RELEASE_DZ": RELEASE_DZ,
        "DROP_X": DROP_X, "DROP_Y": DROP_Y,
        "ARM_Z_OFFSET": ARM_Z_OFFSET, "ARM_Z_SLOPE": ARM_Z_SLOPE,
        "ARM_R_REF": ARM_R_REF,
        "ARM_DX": ARM_DX, "ARM_DY": ARM_DY,
        "ARM_DR": ARM_DR, "ARM_DT": ARM_DT,
        "GRASP_DX": float(GRASP_XY_OFFSET[0]),
        "GRASP_DY": float(GRASP_XY_OFFSET[1]),
        "GRASP_PITCH": GRASP_PITCH,
        "PREGRASP_DZ": PREGRASP_DZ, "GRASP_DZ": GRASP_DZ, "LIFT_DZ": LIFT_DZ,
        "PUSH_EE_Z": PUSH_EE_Z, "PUSH_APPROACH_DZ": PUSH_APPROACH_DZ,
        "MAX_RADIAL": MAX_RADIAL, "MIN_RADIAL": MIN_RADIAL,
        "RETRACT_DZ": RETRACT_DZ,
        "GRASP_CLEARANCE": GRASP_CLEARANCE,
        "TARGET_MAX_EXTENT": TARGET_MAX_XY_EXTENT,
        "GRIPPER_OPENING": GRIPPER_OPENING,
        "FINGER_TABLE_MARGIN": FINGER_TABLE_MARGIN,
        "HOLD_SECONDS": HOLD_SECONDS,
    }


def apply_config(values, quiet=False):
    """Check and apply new values. All of them or none: one bad value
    refuses the whole request. Returns {name: [old, new]} of what changed."""
    global TABLE_Z, WS_X, WS_Y, WS_Z_MAX, SAFE_Z_MIN, HARD_Z_MIN
    global TARGET_X, TARGET_Y, RELEASE_X, RELEASE_Y, RELEASE_DZ
    global DROP_X, DROP_Y, ARM_Z_OFFSET, ARM_Z_SLOPE, ARM_R_REF
    global ARM_DX, ARM_DY, ARM_DR, ARM_DT, GRASP_XY_OFFSET, GRASP_PITCH
    global PREGRASP_DZ, GRASP_DZ, LIFT_DZ, PUSH_EE_Z, PUSH_APPROACH_DZ
    global MAX_RADIAL, GRIPPER_OPENING, FINGER_TABLE_MARGIN, HOLD_SECONDS
    global MIN_RADIAL, RETRACT_DZ, GRASP_CLEARANCE, TARGET_MAX_XY_EXTENT
    before = current_config()
    after = dict(before)
    for name, value in values.items():
        if name not in CONFIG_LIMITS:
            raise ValueError("unknown config name: %s" % name)
        value = float(value)
        low, high = CONFIG_LIMITS[name]
        if not np.isfinite(value) or value < low or value > high:
            raise ValueError("%s=%r is outside the accepted range [%s, %s]"
                             % (name, value, low, high))
        after[name] = value
    shift = after["TABLE_Z"] - before["TABLE_Z"]
    if abs(shift) > 1e-9:
        for name in TABLE_RELATIVE:
            if name not in values:
                after[name] = before[name] + shift
    if not (after["WS_X_MIN"] + 0.05 <= after["WS_X_MAX"]
            and after["WS_Y_MIN"] + 0.05 <= after["WS_Y_MAX"]):
        raise ValueError("workspace box is empty or thinner than 5 cm")
    if after["MIN_RADIAL"] + 0.05 > after["MAX_RADIAL"]:
        raise ValueError("MIN_RADIAL must be at least 5 cm below MAX_RADIAL")
    if after["HARD_Z_MIN"] < after["TABLE_Z"] - 0.01:
        raise ValueError("HARD_Z_MIN may not be more than 1 cm below TABLE_Z")
    TABLE_Z = after["TABLE_Z"]
    WS_X = (after["WS_X_MIN"], after["WS_X_MAX"])
    WS_Y = (after["WS_Y_MIN"], after["WS_Y_MAX"])
    WS_Z_MAX = after["WS_Z_MAX"]
    SAFE_Z_MIN, HARD_Z_MIN = after["SAFE_Z_MIN"], after["HARD_Z_MIN"]
    TARGET_X, TARGET_Y = after["TARGET_X"], after["TARGET_Y"]
    RELEASE_X, RELEASE_Y = after["RELEASE_X"], after["RELEASE_Y"]
    RELEASE_DZ = after["RELEASE_DZ"]
    DROP_X, DROP_Y = after["DROP_X"], after["DROP_Y"]
    ARM_Z_OFFSET, ARM_Z_SLOPE = after["ARM_Z_OFFSET"], after["ARM_Z_SLOPE"]
    ARM_R_REF = after["ARM_R_REF"]
    ARM_DX, ARM_DY = after["ARM_DX"], after["ARM_DY"]
    ARM_DR, ARM_DT = after["ARM_DR"], after["ARM_DT"]
    GRASP_XY_OFFSET = np.array([after["GRASP_DX"], after["GRASP_DY"]])
    GRASP_PITCH = after["GRASP_PITCH"]
    PREGRASP_DZ, GRASP_DZ = after["PREGRASP_DZ"], after["GRASP_DZ"]
    LIFT_DZ = after["LIFT_DZ"]
    PUSH_EE_Z = after["PUSH_EE_Z"]
    PUSH_APPROACH_DZ = after["PUSH_APPROACH_DZ"]
    MAX_RADIAL = after["MAX_RADIAL"]
    MIN_RADIAL, RETRACT_DZ = after["MIN_RADIAL"], after["RETRACT_DZ"]
    GRASP_CLEARANCE = after["GRASP_CLEARANCE"]
    TARGET_MAX_XY_EXTENT = after["TARGET_MAX_EXTENT"]
    GRIPPER_OPENING = after["GRIPPER_OPENING"]
    FINGER_TABLE_MARGIN = after["FINGER_TABLE_MARGIN"]
    HOLD_SECONDS = after["HOLD_SECONDS"]
    changed = {}
    for name in sorted(after):
        if abs(after[name] - before[name]) > 1e-9:
            changed[name] = [before[name], after[name]]
            if not quiet:
                print("[calib] %s: %.4f -> %.4f"
                      % (name, before[name], after[name]))
    return changed


def save_calibration_file(meta=None):
    """Write the current values (and the calibration notes) to CALIB_FILE."""
    global CALIB_META, CALIB_SAVED_AT
    if meta is not None:
        CALIB_META = dict(meta)
    CALIB_SAVED_AT = time.strftime("%Y-%m-%d %H:%M:%S")
    payload = {"config": current_config(), "meta": CALIB_META,
               "server_build": SERVER_BUILD,
               "saved_at": CALIB_SAVED_AT}
    temp = CALIB_FILE + ".tmp"
    with open(temp, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    os.rename(temp, CALIB_FILE)
    print("[calib] saved %d values to %s" % (len(payload["config"]),
                                             CALIB_FILE))
    return CALIB_FILE


def load_calibration_file():
    """Apply CALIB_FILE on top of the environment values, if it exists."""
    global CALIB_META, CALIB_SAVED_AT
    if not USE_CALIB_FILE:
        print("[calib] USE_CALIB_FILE=0: the calibration file is ignored")
        return False
    if not os.path.isfile(CALIB_FILE):
        print("[calib] no calibration file at %s; using server_env.sh "
              "values. Run calibrate_all.py on the Jetson." % CALIB_FILE)
        return False
    try:
        with open(CALIB_FILE) as handle:
            payload = json.load(handle)
        known = dict((k, v) for k, v in payload.get("config", {}).items()
                     if k in CONFIG_LIMITS)
        print("[calib] loading %s (saved %s)"
              % (CALIB_FILE, payload.get("saved_at", "?")))
        apply_config(known)
        CALIB_META = dict(payload.get("meta") or {})
        CALIB_SAVED_AT = payload.get("saved_at")
        return True
    except Exception as error:
        print("[calib] WARNING: could not use %s (%s). Falling back to the "
              "server_env.sh values." % (CALIB_FILE, error))
        return False


def arm_correction(xy):
    """(dxy, dz) to ADD to a camera-frame point to get the arm-frame
    target that puts the tool there. See the notes at ARM_Z_OFFSET."""
    xy = np.asarray(xy, dtype=float)[:2]
    radial = float(np.hypot(xy[0], xy[1]))
    if radial > 1e-6:
        r_hat = xy / radial
    else:
        r_hat = np.array([1.0, 0.0])
    t_hat = np.array([-r_hat[1], r_hat[0]])
    dxy = np.array([ARM_DX, ARM_DY]) + ARM_DR * r_hat + ARM_DT * t_hat
    dz = ARM_Z_OFFSET + ARM_Z_SLOPE * (radial - ARM_R_REF)
    return dxy, float(dz)


CALIB_LOADED = load_calibration_file()


# ---------------- static background (the arm at home, cables) -------------
# At home the arm itself is in the camera's view, a few cm above the table.
# Perception cannot tell it from an object, so it showed up as a grey
# obstacle. calibrate_all.py photographs the EMPTY table with the arm at
# home and sends the depth of everything standing above the table (arm,
# cables, mounts) here. From then on, whenever the arm is at home, every
# frame comes with a mask of the pixels that still show that background,
# and route2_perception drops those pixels. Anything that is IN FRONT of
# the background (closer to the camera by more than BACKGROUND_TOL) is kept,
# so a block is never hidden by this. When the arm is anywhere else the
# mask is not applied, because the background is not where it was.
BACKGROUND_FILE = os.path.expanduser(
    os.environ.get("BACKGROUND_FILE", "~/locobot_background.npz"))
BACKGROUND_TOL = float(os.environ.get("BACKGROUND_TOL", "0.015"))
# The depth camera's error grows with distance (about 1-2 % of the range
# for a D435). A wall 1 m away flickers by more than 1.5 cm, so the
# tolerance is the larger of BACKGROUND_TOL and this share of the distance.
BACKGROUND_REL_TOL = float(os.environ.get("BACKGROUND_REL_TOL", "0.035"))
# How far (rad, any joint) the arm may be from the pose it had when the
# background was recorded and still count as "at home".
HOME_JOINT_TOL = float(os.environ.get("HOME_JOINT_TOL", "0.10"))
BACKGROUND = {"depth": None, "saved_at": None, "joints": None}


def decode_packed(packet):
    raw = zlib.decompress(base64.b64decode(packet["data"]))
    arr = np.frombuffer(raw, dtype=np.dtype(packet["dtype"]))
    return arr.reshape(packet["shape"]).copy()


def load_background():
    if not os.path.isfile(BACKGROUND_FILE):
        print("[background] none stored yet (%s); calibrate_all.py "
              "records it" % BACKGROUND_FILE)
        return
    try:
        data = np.load(BACKGROUND_FILE)
        BACKGROUND["depth"] = np.asarray(data["depth"], dtype=np.uint16)
        BACKGROUND["saved_at"] = str(data["saved_at"])
        if "joints" in data.files:
            BACKGROUND["joints"] = np.asarray(data["joints"], dtype=float)
        print("[background] loaded %d pixels from %s (saved %s)"
              % (int((BACKGROUND["depth"] > 0).sum()), BACKGROUND_FILE,
                 BACKGROUND["saved_at"]))
    except Exception as error:
        print("[background] WARNING: could not load %s (%s)"
              % (BACKGROUND_FILE, error))


load_background()


class RGBDStream(object):
    """Thread-safe RGB/depth stream built from direct rospy subscribers."""

    def __init__(self, rgb_topic, depth_topic):
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.rgb_sample = None
        self.depth_sample = None

        # Keep these objects alive for the lifetime of the environment.
        self.rgb_sub = rospy.Subscriber(
            rgb_topic, Image, self._rgb_callback,
            queue_size=1, buff_size=2 ** 24)
        self.depth_sub = rospy.Subscriber(
            depth_topic, Image, self._depth_callback,
            queue_size=1, buff_size=2 ** 24)
        print("[rgbd] rospy subscribers rgb=%s depth=%s max_dt=%.3fs"
              % (rgb_topic, depth_topic, RGBD_SYNC_SLOP))

    def _rgb_callback(self, rgb_msg):
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
            with self.changed:
                self.rgb_sample = {
                    "rgb": rgb,
                    "rgb_stamp": rgb_msg.header.stamp.to_sec(),
                }
                self.changed.notify_all()
        except Exception:
            traceback.print_exc()

    def _depth_callback(self, depth_msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough")
            depth = np.asarray(depth)

            # Canonical wire/storage representation is uint16 millimetres.
            if np.issubdtype(depth.dtype, np.floating):
                depth = np.where(np.isfinite(depth), depth, 0.0)
                depth = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
            else:
                depth = np.clip(depth, 0, 65535).astype(np.uint16,
                                                               copy=False)

            with self.changed:
                self.depth_sample = {
                    "depth": np.ascontiguousarray(depth),
                    "depth_stamp": depth_msg.header.stamp.to_sec(),
                    "depth_frame": depth_msg.header.frame_id,
                }
                self.changed.notify_all()
        except Exception:
            traceback.print_exc()

    def latest(self, timeout=RGBD_TIMEOUT):
        """Return the newest rospy frames whose timestamps are close enough."""
        deadline = time.time() + timeout
        with self.changed:
            while True:
                if self.rgb_sample is not None and self.depth_sample is not None:
                    rgb_stamp = self.rgb_sample["rgb_stamp"]
                    depth_stamp = self.depth_sample["depth_stamp"]
                    if abs(rgb_stamp - depth_stamp) <= RGBD_SYNC_SLOP:
                        return {
                            "rgb": self.rgb_sample["rgb"].copy(),
                            "depth": self.depth_sample["depth"].copy(),
                            "rgb_stamp": rgb_stamp,
                            "depth_stamp": depth_stamp,
                            "depth_frame": self.depth_sample["depth_frame"],
                        }

                remaining = deadline - time.time()
                if remaining <= 0:
                    rgb_stamp = (None if self.rgb_sample is None else
                                 self.rgb_sample["rgb_stamp"])
                    depth_stamp = (None if self.depth_sample is None else
                                   self.depth_sample["depth_stamp"])
                    # A None stamp means the callback never fired, which almost
                    # always means the subscribed topic name is wrong. List the
                    # camera topics that actually exist so the correct name is
                    # visible in the error itself.
                    hint = ""
                    if rgb_stamp is None or depth_stamp is None:
                        try:
                            import rospy
                            published = [
                                name for name, _ in
                                rospy.get_published_topics()
                                if "camera" in name or "image" in name]
                            hint = ("; camera/image topics currently "
                                    "published: %s" % (sorted(published),))
                        except Exception:
                            hint = ""
                    raise RuntimeError(
                        "timed out waiting for rospy RGB-D pair on %s and %s; "
                        "last stamps rgb=%s depth=%s%s"
                        % (RGB_TOPIC, DEPTH_TOPIC, rgb_stamp, depth_stamp,
                           hint))
                self.changed.wait(min(remaining, 0.1))



class GraspEnv(object):

    def __init__(self):
        print("[env] initializing PyRobot ...")
        self.bot = Robot("locobot")
        self.arm = self.bot.arm
        self.gripper = self.bot.gripper
        configured_home = parse_joint_vector(ARM_HOME_JOINTS_TEXT)
        if configured_home is not None:
            self.home_joints = configured_home.copy()
            self.home_source = "ARM_HOME_JOINTS"
        elif CAPTURE_START_AS_HOME:
            self.home_joints = self._joints().astype(float, copy=True)
            if self.home_joints.size != 5:
                raise RuntimeError(
                    "expected 5 LoCoBot arm joints, got %d"
                    % self.home_joints.size)
            self.home_source = "server startup pose"
        else:
            self.home_joints = None
            self.home_source = "PyRobot default go_home"
        print("[env] ARM_Z_OFFSET=%.3f  GRIPPER_LENGTH=%.3f (sign %+.0f)  "
              "TABLE_Z=%.3f camera frame  IK=%s"
              % (ARM_Z_OFFSET, GRIPPER_LENGTH, GRIPPER_TCP_SIGN, TABLE_Z,
                 "numerical" if IK_NUMERICAL else "analytical"))
        print("[env] server build %s (file %s)"
              % (SERVER_BUILD, os.path.abspath(__file__)))
        print("[env] calibration file: %s  (%s)"
              % (CALIB_FILE,
                 "loaded, %s" % CALIB_META.get("calibrated_at", "no date")
                 if CALIB_LOADED else "NOT loaded"))
        print("[env] gentle push: closed jaw, steps of %.0f mm with %.2f s "
              "stops, stroke %.1f-%.1f cm from the object's face, stall "
              "stop after %d steps"
              % (PUSH_STEP * 1000, PUSH_STEP_PAUSE, PUSH_MIN_DISTANCE * 100,
                 PUSH_MAX_DISTANCE * 100, PUSH_STALL_STEPS))
        print("[env] arm correction: ARM_Z_SLOPE=%.3f (ref %.2f)  "
              "ARM_DX/DY=%.4f/%.4f  ARM_DR/DT=%.4f/%.4f"
              % (ARM_Z_SLOPE, ARM_R_REF, ARM_DX, ARM_DY, ARM_DR, ARM_DT))
        print("[env] GRASP_PITCH=%.2f  PREGRASP_DZ=%.3f  GRASP_DZ=%.3f  "
              "LIFT_DZ=%.3f  PUSH_EE_Z=%.3f  PUSH_APPROACH_DZ=%.3f"
              % (GRASP_PITCH, PREGRASP_DZ, GRASP_DZ, LIFT_DZ, PUSH_EE_Z,
                 PUSH_APPROACH_DZ))
        if GRIPPER_LENGTH <= 0:
            print("[env] WARNING: GRIPPER_LENGTH is 0, so xyz means the "
                  "WRIST, not the fingertips. At a pitch away from "
                  "vertical the fingers land several centimetres to the "
                  "side of the commanded point.")
        print("[env] workspace WS_X=%s WS_Y=%s WS_Z_MAX=%.3f  "
              "HARD_Z_MIN=%.3f SAFE_Z_MIN=%.3f  PUSH_EE_Z=%.3f  "
              "TARGET=(%.2f, %.2f)"
              % (list(WS_X), list(WS_Y), WS_Z_MAX, HARD_Z_MIN, SAFE_Z_MIN,
                 PUSH_EE_Z, TARGET_X, TARGET_Y))
        if abs(WS_X[0] - 0.28) < 1e-6 and abs(WS_Y[0] + 0.25) < 1e-6:
            print("[env] NOTE: the workspace box is the built-in default "
                  "(front-of-robot). If the table is elsewhere, the "
                  "environment variables were not applied - start with "
                  "'source ~/robot_env.sh && python ...'")
        print("[env] episode home source=%s joints=%s"
              % (self.home_source,
                 None if self.home_joints is None else
                 np.round(self.home_joints, 6).tolist()))
        self.rgbd = RGBDStream(RGB_TOPIC, DEPTH_TOPIC)
        self.fk_ok = self._check_fk()
        print("[env] FK available: %s" % self.fk_ok)
        if CAMERA_AUTO_POSE:
            self.set_camera(CAM_PAN, CAM_TILT)
        else:
            print("[env] manual camera mode: startup will not command pan/tilt")
        print("[env] workspace x=%s y=%s target=(%.3f, %.3f)"
              % (WS_X, WS_Y, TARGET_X, TARGET_Y))
        self.t = 0
        self.gripper_flag = 0.0   # 0 open, 1 closed
        self.holding = False      # a scripted grasp reported held
        self.motion_log = []      # failure reasons of the current command
        self.last_pitch = GRASP_PITCH
        self.target_centroid = np.array(
            [TARGET_X, TARGET_Y, TARGET_TOP_Z - 0.03])
        self.target_extent = np.zeros(3)
        self.target_yaw = None
        self.target_rect = (None, None)
        self.target_visible = False
        self.target_clearance = 0.0
        self.obstacles = []
        self.pushables = None     # client-supplied push candidates
        self.last_grasp = {}      # phase-by-phase record of the last grasp
        self.last_push = {}       # how the last push went

    # ---------- kinematics helpers ----------

    def _check_fk(self):
        try:
            j = self.arm.get_joint_angles()
            p = self._fk(j)
            return p is not None
        except Exception:
            return False

    def _fk(self, joints):
        try:
            pos, _ = self.arm.compute_fk_position(
                np.asarray(joints), "gripper_link")
            return np.asarray(pos).reshape(-1)
        except Exception:
            return None

    def _ee_pos(self):
        """End-effector position, converted into the camera frame."""
        pose = self.arm.pose_ee
        xyz = np.asarray(pose[0], dtype=float).reshape(-1).copy()
        # Undo arm_correction. The correction depends on the camera-frame
        # xy we are solving for, so take two passes; the second one is
        # exact to well under a millimetre for corrections of a few cm.
        cam_xy = xyz[:2] - arm_correction(xyz[:2])[0]
        dxy, dz = arm_correction(cam_xy)
        xyz[:2] -= dxy
        xyz[2] -= dz
        return xyz

    def _joints(self):
        return np.asarray(self.arm.get_joint_angles()).reshape(-1)

    def capture_current_home(self):
        """Use the current encoder readings as the home for future resets."""
        joints = self._joints().astype(float, copy=True)
        if joints.size != 5 or not np.all(np.isfinite(joints)):
            raise RuntimeError("could not read five finite arm joint angles")
        self.home_joints = joints
        self.home_source = "capture_home command"
        print("[env] captured episode home joints=%s"
              % np.round(self.home_joints, 6).tolist())
        return self.home_joints.copy()

    def _retract(self):
        """Lift the fingertips straight up before going home.

        The move home is a joint-space move: every joint turns at once
        toward its home angle. Started from a low pose near the robot (an
        object at the near table edge), that sweep took the gripper or the
        forearm through the table edge and the arm's overload protection
        shut it down. Going straight up first starts the sweep clear of the
        table. If the arm is already high, or nowhere near the table, this
        does nothing."""
        try:
            tip = self._fingertip_from_wrist(self.last_pitch)
        except Exception:
            return
        if not np.all(np.isfinite(tip)):
            return
        safe_z = TABLE_Z + RETRACT_DZ
        if tip[2] >= safe_z - 0.01 or self._at_home():
            return
        radial = float(np.hypot(tip[0], tip[1]))
        # straight up first; if the IK refuses that close to the robot, go
        # up a little less, then up and slightly outward
        tries = [(tip[0], tip[1], safe_z),
                 (tip[0], tip[1], max(tip[2] + 0.04, TABLE_Z + 0.05))]
        if radial > 1e-3:
            out = (radial + 0.03) / radial
            tries.append((tip[0] * out, tip[1] * out, safe_z))
        for xyz in tries:
            if self._goto_xyz(list(xyz), pitch=self.last_pitch,
                              correct=False, quiet=True, any_xy=True):
                return
        self._log_motion(
            "retract: could not lift the gripper at %s before going home; "
            "going home directly" % np.round(tip, 3).tolist())

    def go_episode_home(self):
        """Move to the configured/captured home; never recapture implicitly.
        The gripper is lifted clear of the table first (see _retract)."""
        self._retract()
        if self.home_joints is None:
            result = self.arm.go_home()
        else:
            result = self.arm.set_joint_positions(
                self.home_joints.tolist(), plan=ARM_HOME_PLAN)
        time.sleep(SETTLE_TIME)
        # Some PyRobot versions return None after a successful command.
        ok = True if result is None else bool(result)
        # Oct 29f: the jaw left closed by a push is opened HERE, at home,
        # where the fingers are clear of the table and the blocks. Not
        # when holding a grasped object.
        if (ok and not JAW_STAYS_CLOSED and not self.holding
                and self.gripper_flag >= 0.5):
            self._open_gripper()
        return ok

    # ---------- camera pan/tilt ----------

    def set_camera(self, pan, tilt):
        """Move the camera pan/tilt mount to a fixed pose. Called once at
        startup so every session sees the workspace from the same angle."""
        try:
            self.bot.camera.set_pan_tilt(pan, tilt, wait=True)
            print("[env] camera set: pan=%.2f tilt=%.2f" % (pan, tilt))
            return True
        except Exception:
            traceback.print_exc()
            print("[env] WARNING: could not set camera pan/tilt. "
                  "Set it manually with rostopic pub.")
            return False

    # ---------- point cloud ----------

    def _get_depth_intrinsics(self):
        """Return the 3x3 intrinsic matrix that matches the DEPTH_TOPIC stream.
        DEPTH_TOPIC is the aligned depth stream, which lives in the color
        camera frame, so DEPTH_INFO_TOPIC points at the color camera_info and
        these are the correct intrinsics for reprojecting the aligned depth."""
        import rospy
        from sensor_msgs.msg import CameraInfo
        msg = rospy.wait_for_message(
            DEPTH_INFO_TOPIC, CameraInfo, timeout=5.0)
        return np.asarray(msg.K, dtype=np.float64).reshape(3, 3)

    def _get_cam_to_base(self, camera_frame):
        """Return the 4x4 transform base_link <- camera_frame."""
        import tf as ros_tf
        if not hasattr(self, "_tf_listener"):
            self._tf_listener = ros_tf.TransformListener()
            time.sleep(1.0)  # let the buffer fill once
        import rospy
        camera_frame = camera_frame or "camera_color_optical_frame"
        self._tf_listener.waitForTransform(
            "/base_link", camera_frame,
            rospy.Time(0), rospy.Duration(3.0))
        trans, rot = self._tf_listener.lookupTransform(
            "/base_link", camera_frame, rospy.Time(0))
        T = np.eye(4)
        T[:3, :3] = ros_tf.transformations.quaternion_matrix(rot)[:3, :3]
        T[:3, 3] = trans
        return T

    def _capture_pcd(self):
        """Point cloud in the BASE frame, computed directly from the aligned
        depth image + color intrinsics + the depth message's TF. The aligned
        depth stream is in the color optical frame, so the color intrinsics and
        the message frame_id are the consistent choice. This bypasses PyRobot's
        fixed-resolution point-cloud grid. Returns Nx3 array or None."""
        try:
            sample = self.rgbd.latest()
            depth = sample["depth"]
            K = self._get_depth_intrinsics()
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            # RGBDStream canonicalizes depth as uint16 millimetres.
            z = depth.astype(np.float32) / 1000.0

            stride = 4  # ~57k points at 1280x720, enough and fast
            z = z[::stride, ::stride]
            h, w = z.shape
            us = (np.arange(w) * stride).astype(np.float32)
            vs = (np.arange(h) * stride).astype(np.float32)
            uu, vv = np.meshgrid(us, vs)

            valid = (z > 0.1) & (z < 1.5)
            zc = z[valid]
            xc = (uu[valid] - cx) * zc / fx
            yc = (vv[valid] - cy) * zc / fy
            pts_cam = np.stack([xc, yc, zc], axis=1)

            T = self._get_cam_to_base(sample["depth_frame"])
            pts = pts_cam.dot(T[:3, :3].T) + T[:3, 3]
            return pts
        except Exception:
            traceback.print_exc()
            return None

    def _remove_table(self, roi):
        """Fit the table plane on low points, keep points above it.
        Falls back to a fixed z threshold if the fit looks wrong."""
        cand = roi[roi[:, 2] < TABLE_Z + 0.05]
        if len(cand) >= 200:
            try:
                A = np.stack([cand[:, 0], cand[:, 1],
                              np.ones(len(cand))], axis=1)
                coef = np.linalg.lstsq(A, cand[:, 2], rcond=None)[0]
                slope_ok = abs(coef[0]) < 0.15 and abs(coef[1]) < 0.15
                plane_z = roi[:, 0] * coef[0] + roi[:, 1] * coef[1] + coef[2]
                level_ok = abs(np.median(plane_z) - TABLE_Z) < 0.05
                if slope_ok and level_ok:
                    return roi[(roi[:, 2] - plane_z) > PC_Z_CUT]
            except Exception:
                traceback.print_exc()
        return roi[roi[:, 2] > TABLE_Z + PC_Z_CUT]

    def _cluster(self, pts):
        """Voxel connected-components clustering, numpy only.
        Returns clusters sorted by point count, largest first."""
        if len(pts) == 0:
            return []
        keys = np.floor(pts / CLUSTER_VOXEL).astype(np.int64)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        n = len(uniq)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        index = {}
        for i in range(n):
            index[tuple(uniq[i])] = i
        offsets = [(a, b, c)
                   for a in (-1, 0, 1) for b in (-1, 0, 1)
                   for c in (-1, 0, 1) if (a, b, c) > (0, 0, 0)]
        for i in range(n):
            v = uniq[i]
            for off in offsets:
                j = index.get((v[0] + off[0], v[1] + off[1], v[2] + off[2]))
                if j is not None:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
        vox_label = np.array([find(i) for i in range(n)])
        labels = vox_label[inv]
        clusters = []
        for lab in np.unique(labels):
            p = pts[labels == lab]
            if len(p) < CLUSTER_MIN_PTS:
                continue
            # keep a downsampled 2D point cloud for the point-based
            # clearance check (same shape as the client's route 2)
            xy = p[:, :2]
            if len(xy) > 60:
                step = int(np.ceil(len(xy) / 60.0))
                xy = xy[::step][:60]
            clusters.append({
                "n": int(len(p)),
                "centroid": p.mean(axis=0),
                "extent": p.max(axis=0) - p.min(axis=0),
                "points_xy": xy})
        clusters.sort(key=lambda c: -c["n"])
        return clusters

    def _process_pcd(self, pts):
        """Crop to workspace, remove the fitted table plane, cluster.
        Returns (clusters, n_object_points)."""
        if pts is None or len(pts) == 0:
            return [], 0
        m = ((pts[:, 0] > WS_X[0]) & (pts[:, 0] < WS_X[1]) &
             (pts[:, 1] > WS_Y[0]) & (pts[:, 1] < WS_Y[1]) &
             (pts[:, 2] > TABLE_Z - 0.04) & (pts[:, 2] < WS_Z_MAX))
        roi = pts[m]
        if len(roi) < PC_MIN_POINTS:
            return [], len(roi)
        obj = self._remove_table(roi)
        return self._cluster(obj), len(obj)

    @staticmethod
    def _sane(cl):
        return all(0.01 < ei < 0.20 for ei in cl["extent"])

    @staticmethod
    def _xy_radius(cluster):
        return 0.5 * float(np.linalg.norm(cluster["extent"][:2]))

    def _clearance_to_obstacles(self, target, obstacles):
        """Shortest distance from any obstacle POINT to the target's xy
        centroid, in metres. 0.20 when nothing is near. The grasp is
        graspable only when this is >= GRASP_CLEARANCE. See the client's
        route 2 for the same rule.

        An obstacle without a point cloud (never happens with the current
        clients but kept for backward compatibility) falls back to the
        centroid distance."""
        if target is None:
            return 0.0
        if not obstacles:
            return 0.20
        center = np.asarray(target["centroid"][:2], dtype=float)
        min_dist = 0.20
        for obstacle in obstacles:
            pts = obstacle.get("points_xy")
            if pts is not None and len(pts) > 0:
                arr = np.asarray(pts, dtype=float)
                dist = float(np.min(np.linalg.norm(arr - center, axis=1)))
            else:
                dist = float(np.linalg.norm(
                    np.asarray(obstacle["centroid"][:2], dtype=float)
                    - center))
            if dist < min_dist:
                min_dist = dist
        return float(np.clip(min_dist, 0.0, 0.20))

    def update_perception(self, reset_target=False):
        """Refresh target/obstacles after every high-level decision.

        At reset, target identity is the cluster nearest the configured anchor.
        Later, nearest-neighbor tracking uses the previous target centroid.
        """
        clusters, n = self._process_pcd(self._capture_pcd())
        sane = [cluster for cluster in clusters if self._sane(cluster)]
        target = None
        target_index = None
        if sane:
            reference = (np.array([TARGET_X, TARGET_Y]) if reset_target
                         else self.target_centroid[:2])
            distances = [
                np.linalg.norm(cluster["centroid"][:2] - reference)
                for cluster in sane]
            candidate_index = int(np.argmin(distances))
            if reset_target or distances[candidate_index] <= TARGET_TRACK_MAX:
                target_index = candidate_index
                target = sane[candidate_index]

        if target is not None:
            c, e = target["centroid"], target["extent"]
            self.target_centroid = c.copy()
            self.target_extent = e.copy()
            self.target_visible = True
        else:
            self.target_visible = False

        self.obstacles = [
            cluster for i, cluster in enumerate(sane) if i != target_index]
        self.obstacles.sort(key=lambda obstacle: np.linalg.norm(
            obstacle["centroid"][:2] - self.target_centroid[:2]))
        self.obstacles = self.obstacles[:MAX_OBSTACLES]
        self.target_clearance = self._clearance_to_obstacles(
            target, self.obstacles)
        print("[pc] points=%d sane=%d target=%s obstacles=%d clearance=%.3f"
              % (n, len(sane),
                 None if target is None else
                 np.round(target["centroid"], 3).tolist(),
                 len(self.obstacles), self.target_clearance))
        return n

    def current_pregrasp(self):
        top = float(np.clip(
            self.target_centroid[2] + self.target_extent[2] / 2.0,
            TABLE_Z + 0.03, 0.38))
        return np.array([self.target_centroid[0], self.target_centroid[1],
                         top + PREGRASP_DZ])

    def target_graspable(self):
        target_isolated = bool(
            np.max(self.target_extent[:2]) <= TARGET_MAX_XY_EXTENT)
        return bool(self.target_visible and target_isolated and
                    self.target_clearance >= GRASP_CLEARANCE)

    def pc_stats(self):
        pts = self._capture_pcd()
        clusters, n = self._process_pcd(pts)
        out = []
        for cl in clusters[:6]:
            out.append({"n": cl["n"],
                        "centroid": np.round(cl["centroid"], 3).tolist(),
                        "extent": np.round(cl["extent"], 3).tolist(),
                        "sane": self._sane(cl)})
        return {"n_raw": 0 if pts is None else int(len(pts)),
                "n_object": int(n),
                "n_clusters": len(clusters),
                "clusters": out,
                "note": "reset target = sane cluster nearest TARGET_X/Y"}

    def pc_save(self):
        import os
        d = os.path.expanduser("~/pc_dumps")
        if not os.path.isdir(d):
            os.makedirs(d)
        pts = self._capture_pcd()
        stamp = time.strftime("%H%M%S")
        if pts is not None:
            np.save("%s/raw_%s.npy" % (d, stamp), pts)
        clusters, n = self._process_pcd(pts)
        return {"saved": pts is not None, "dir": d, "n_object": int(n),
                "n_clusters": len(clusters)}

    # ---------- observation ----------

    @staticmethod
    def _center_crop_resize(array, size):
        """Center-crop to a square and resize with nearest-neighbor numpy."""
        h, w = array.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = array[y0:y0 + side, x0:x0 + side]
        iy = np.linspace(0, side - 1, size).round().astype(np.int64)
        ix = np.linspace(0, side - 1, size).round().astype(np.int64)
        return np.ascontiguousarray(crop[iy[:, None], ix[None, :]])

    @staticmethod
    def _pack_array(array):
        """Losslessly encode a numpy array for newline-delimited JSON."""
        array = np.ascontiguousarray(array)
        compressed = zlib.compress(array.tobytes(), 1)
        return {
            "encoding": "zlib+base64",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": base64.b64encode(compressed).decode("ascii"),
        }

    def _rgbd_obs(self):
        sample = self.rgbd.latest()
        rgb = self._center_crop_resize(sample["rgb"], RGBD_SIZE)
        depth = self._center_crop_resize(sample["depth"], RGBD_SIZE)
        return {
            "rgb": self._pack_array(rgb),
            "depth": self._pack_array(depth),
            "rgb_stamp": sample["rgb_stamp"],
            "depth_stamp": sample["depth_stamp"],
            "depth_unit": "millimetres",
        }

    def rgbd_stats(self):
        """Return safe camera diagnostics without moving the robot."""
        sample = self.rgbd.latest()
        rgb = sample["rgb"]
        depth = sample["depth"]
        valid = depth > 0
        valid_depth = depth[valid]
        if valid_depth.size:
            depth_mm = {
                "min": int(valid_depth.min()),
                "median": float(np.median(valid_depth)),
                "max": int(valid_depth.max()),
            }
        else:
            depth_mm = {"min": None, "median": None, "max": None}
        return {
            "rgb_shape": list(rgb.shape),
            "rgb_dtype": str(rgb.dtype),
            "depth_shape": list(depth.shape),
            "depth_dtype": str(depth.dtype),
            "depth_frame": sample["depth_frame"],
            "valid_depth_fraction": float(valid.mean()),
            "depth_mm": depth_mm,
            "rgb_stamp": sample["rgb_stamp"],
            "depth_stamp": sample["depth_stamp"],
            "stamp_delta_sec": abs(
                sample["rgb_stamp"] - sample["depth_stamp"]),
        }

    def apply_external_perception(self, payload):
        """Overwrite target/obstacle state with client-supplied perception.

        The training client computes object centroids from the full frame
        (YOLO segmentation + depth back-projection) and sends them with each
        step. The primitives (grasp position, push targeting, graspable gate)
        then run on those YOLO centroids instead of the internal point-cloud
        clusters. Obstacle ORDER is preserved as sent, so the policy's slot
        index and the push selector agree between client and server."""
        target = payload.get("target")
        if target is not None:
            self.target_centroid = np.asarray(
                target["centroid"], dtype=float)
            self.target_extent = np.asarray(target["extent"], dtype=float)
            self.target_yaw = (float(target["yaw"])
                               if target.get("yaw") is not None else None)
            self.target_rect = (target.get("rect_long"),
                                target.get("rect_short"))
            self.target_visible = True
        else:
            self.target_visible = False
        obstacles = []
        for entry in (payload.get("obstacles") or [])[:MAX_OBSTACLES]:
            pts = entry.get("points_xy")
            obstacles.append({
                "centroid": np.asarray(entry["centroid"], dtype=float),
                "extent": np.asarray(entry["extent"], dtype=float),
                "n": int(entry.get("n", 0)),
                "points_xy": (np.asarray(pts, dtype=float) if pts else
                              np.zeros((0, 2)))})
        self.obstacles = obstacles
        target_cluster = ({"centroid": self.target_centroid,
                           "extent": self.target_extent}
                          if self.target_visible else None)
        self.target_clearance = self._clearance_to_obstacles(
            target_cluster, self.obstacles)
        print("[external] target=%s obstacles=%d clearance=%.3f"
              % (None if not self.target_visible else
                 np.round(self.target_centroid, 3).tolist(),
                 len(self.obstacles), self.target_clearance))
        # Push candidates. When the client sends "pushables" (target first
        # when it is visible, then the obstacles), the push selector indexes
        # that list, so pushing the target out of a pile is a legal action.
        # The target is never in the pick-and-place list.
        self.pushables = payload.get("pushables")
    def _home_error(self):
        """Largest joint difference (rad) from the pose the arm had when
        the background was recorded (or, before that, from the commanded
        home). A real arm does not come back to exactly the commanded
        angles, so the recorded pose is the better reference."""
        if BACKGROUND["joints"] is not None:
            reference = BACKGROUND["joints"]
        elif self.home_joints is not None:
            reference = np.asarray(self.home_joints, dtype=float)
        else:
            reference = np.zeros(5)
        try:
            return float(np.max(np.abs(self._joints() - reference)))
        except Exception:
            return float("inf")

    def _at_home(self):
        return self._home_error() < HOME_JOINT_TOL

    def full_frame(self, raw=False):
        """Full-resolution synchronized RGB + aligned depth for client-side
        YOLO perception on the training machine. Also ships the color
        intrinsics and the base_link<-camera transform so the client can
        back-project the depth image into the base frame. Never moves the
        robot; it returns whatever the camera sees right now, so the client
        should call it right after reset/step, when the arm is out of the
        way."""
        sample = self.rgbd.latest()
        K = self._get_depth_intrinsics()
        T = self._get_cam_to_base(sample.get("depth_frame"))
        resp = {
            "rgb": self._pack_array(sample["rgb"]),
            "depth": self._pack_array(sample["depth"]),
            "intrinsics": np.round(K, 6).tolist(),
            "cam_to_base": np.round(T, 6).tolist(),
            "rgb_stamp": sample["rgb_stamp"],
            "depth_stamp": sample["depth_stamp"],
            "depth_unit": "millimetres",
        }
        background = BACKGROUND["depth"]
        depth = sample["depth"]
        if (background is not None and background.shape == depth.shape
                and background.any()):
            error = self._home_error()
            at_home = error < HOME_JOINT_TOL
            resp["at_home"] = at_home
            resp["home_joint_error"] = round(error, 4)
            if at_home:
                static = background > 0
                valid = depth > 0
                d = depth.astype(np.int32)
                b = background.astype(np.int32)
                tol = np.maximum(int(round(BACKGROUND_TOL * 1000)),
                                 (BACKGROUND_REL_TOL * b).astype(np.int32))
                seen = static & valid
                match = seen & (np.abs(d - b) < tol)
                resp["background_match"] = float(
                    match.sum()) / max(1, int(seen.sum()))
                if not raw:
                    # drop what is not clearly in front of the background
                    cover = seen & (d >= b - tol)
                    resp["static_mask"] = self._pack_array(
                        cover.astype(np.uint8))
                    resp["static_pixels"] = int(cover.sum())
        elif background is not None:
            resp["background_note"] = (
                "stored background is %s but the depth image is %s; not "
                "used. Run calibrate_all.py again."
                % (list(background.shape), list(depth.shape)))
        return resp

    def _obs(self):
        j = self._joints()
        ee = self._ee_pos()
        values = [j, ee, [self.gripper_flag],
                  self.target_centroid, self.target_extent,
                  [float(self.target_visible), self.target_clearance]]
        for index in range(MAX_OBSTACLES):
            if index < len(self.obstacles):
                obstacle = self.obstacles[index]
                values.extend([
                    obstacle["centroid"], obstacle["extent"], [1.0]])
            else:
                values.append(np.zeros(7))
        obs = np.concatenate(values)
        if obs.size != 45:
            raise RuntimeError("internal observation size is %d, expected 45"
                               % obs.size)
        return obs.tolist()

    # ---------- gripper ----------

    def _read_gripper_state(self):
        """Raw gripper state. Mapping differs across PyRobot versions;
        verify on site with grasp_test. Returns -1 if unreadable."""
        try:
            return int(self.gripper.get_gripper_state())
        except Exception:
            return -1

    def _open_gripper(self):
        try:
            self.gripper.open()
            self.gripper_flag = 0.0
            time.sleep(1.0)
        except Exception:
            traceback.print_exc()

    def _close_gripper(self):
        try:
            self.gripper.close()
            self.gripper_flag = 1.0
            time.sleep(1.5)
        except Exception:
            traceback.print_exc()

    # ---------- scripted grasp ----------

    def _log_motion(self, reason):
        """Record a motion failure. The list travels back to the training
        client inside every response, so the reason a primitive did nothing
        shows up in the client's terminal and CSV instead of dying silently
        in this terminal. IK failures used to be completely silent here,
        which cost days of guessing."""
        print("[motion] %s" % reason)
        self.motion_log.append(reason)

    @staticmethod
    def _wrist_for_fingertip(xyz, pitch):
        """Wrist position (arm frame) that puts the FINGERTIPS at xyz
        (camera frame). See the tool-centre-point notes at the top."""
        wrist = np.asarray(xyz, dtype=float).copy()
        if GRIPPER_LENGTH > 0:
            azimuth = float(np.arctan2(xyz[1], xyz[0]))
            radial = np.array([np.cos(azimuth), np.sin(azimuth)])
            wrist[:2] -= (GRIPPER_TCP_SIGN * GRIPPER_LENGTH
                          * np.cos(pitch) * radial)
            wrist[2] += GRIPPER_LENGTH * np.sin(pitch)
        dxy, dz = arm_correction(xyz)
        wrist[:2] += dxy
        wrist[2] += dz
        return wrist

    def _fingertip_from_wrist(self, pitch):
        """Where the fingertips are now, in the camera frame."""
        wrist = self._ee_pos()
        if GRIPPER_LENGTH <= 0:
            return wrist
        tip = wrist.copy()
        azimuth = float(np.arctan2(wrist[1], wrist[0]))
        radial = np.array([np.cos(azimuth), np.sin(azimuth)])
        tip[:2] += (GRIPPER_TCP_SIGN * GRIPPER_LENGTH
                    * np.cos(pitch) * radial)
        tip[2] -= GRIPPER_LENGTH * np.sin(pitch)
        return tip

    def _goto_xyz(self, xyz, wait=True, pitch=None, roll=None,
                  settle=None, correct=True, quiet=False, any_xy=False):
        """Move the FINGERTIPS to xyz. Returns True on success.
        pitch/roll override GRASP_PITCH/GRASP_ROLL for this one call.
        settle overrides SETTLE_TIME; correct=False skips the residual
        re-command (used while pushing); quiet=True leaves the refusal
        message to the caller, who knows what the step was for."""
        xyz = np.asarray(xyz, dtype=float)
        pitch = GRASP_PITCH if pitch is None else float(pitch)
        roll = GRASP_ROLL if roll is None else float(roll)
        settle = SETTLE_TIME if settle is None else float(settle)
        self.last_pitch = pitch
        outside = (xyz[0] < WS_X[0] or xyz[0] > WS_X[1]
                   or xyz[1] < WS_Y[0] or xyz[1] > WS_Y[1])
        if xyz[2] < HARD_Z_MIN or (outside and not any_xy):
            self._log_motion(
                "out-of-workspace xyz=%s (WS_X=%s WS_Y=%s HARD_Z_MIN=%.3f)"
                % (np.round(xyz, 3).tolist(), list(WS_X), list(WS_Y),
                   HARD_Z_MIN))
            return False
        try:
            arm_xyz = self._wrist_for_fingertip(xyz, pitch)
            ok = self.arm.set_ee_pose_pitch_roll(
                position=arm_xyz,
                pitch=pitch, roll=roll,
                plan=False, wait=wait, numerical=IK_NUMERICAL)
            time.sleep(settle)
            if ok and correct and GOTO_RESIDUAL_TOL > 0:
                # The controller stops short by a steady ~2 cm at the far
                # end of the workspace. Read the arm's own FK and re-issue
                # the command once with the residual folded in.
                residual = xyz - self._fingertip_from_wrist(pitch)
                if float(np.linalg.norm(residual)) > GOTO_RESIDUAL_TOL:
                    corrected = arm_xyz + residual
                    ok2 = self.arm.set_ee_pose_pitch_roll(
                        position=corrected, pitch=pitch, roll=roll,
                        plan=False, wait=wait, numerical=IK_NUMERICAL)
                    time.sleep(SETTLE_TIME)
                    after = xyz - self._fingertip_from_wrist(pitch)
                    print("[motion] residual %.0f mm -> %.0f mm after "
                          "correction (retry ok=%s)"
                          % (np.linalg.norm(residual) * 1000,
                             np.linalg.norm(after) * 1000, bool(ok2)))
            if not ok and not quiet:
                self._log_motion(
                    "IK/controller refused xyz=%s pitch=%.2f (radial "
                    "%.2f m, height %.2f m above TABLE_Z; a near-vertical "
                    "pitch is feasible only inside a limited radial/height "
                    "envelope - measure it with reach_scan.py)"
                    % (np.round(xyz, 3).tolist(), pitch,
                       float(np.hypot(xyz[0], xyz[1])), xyz[2] - TABLE_Z))
            return bool(ok)
        except Exception:
            traceback.print_exc()
            self._log_motion("exception during goto xyz=%s"
                             % np.round(xyz, 3).tolist())
            return False

    def _closing_candidates(self, yaw, rect=(None, None)):
        """World-frame closing directions that put the fingers across a
        face. A near-square footprint offers both face normals; an
        elongated one only the direction across its short side."""
        if yaw is None:
            return [None]
        across_short = yaw + np.pi / 2.0
        long_side, short_side = rect
        if (long_side is not None and short_side is not None
                and short_side > 1e-3 and long_side / short_side > 1.3):
            return [across_short]
        return [across_short, yaw]

    def _tilt_plan(self, target, yaw="target", rect="target", pitch=None):
        """Pick the closing direction and the fingertip height difference
        for a possibly tilted gripper. Returns (closing_dir_or_None,
        height_difference)."""
        if yaw == "target":
            yaw = self.target_yaw
        if rect == "target":
            rect = self.target_rect
        pitch = GRASP_PITCH if pitch is None else float(pitch)
        tilt = abs(np.pi / 2.0 - pitch)
        azimuth = float(np.arctan2(target[1], target[0]))
        best, best_diff = None, None
        for closing in self._closing_candidates(yaw, rect):
            if closing is None:
                # unknown orientation: assume the canonical roll closes in
                # the tilt plane (worst case) so the height is still safe
                diff = GRIPPER_OPENING * np.sin(tilt)
            else:
                # fraction of the closing direction that lies in the tilt
                # (radial-vertical) plane
                in_plane = abs(np.cos(closing - azimuth))
                diff = GRIPPER_OPENING * np.sin(tilt) * in_plane
            if best_diff is None or diff < best_diff:
                best, best_diff = closing, float(diff)
        return best, best_diff

    def _roll_for_closing(self, target, closing, roll_offset=0.0):
        if GRASP_ROLL_MODE != "auto" or closing is None:
            return GRASP_ROLL + roll_offset
        azimuth = float(np.arctan2(target[1], target[0]))
        roll = (GRASP_ROLL_SIGN * (closing - azimuth) + GRASP_ROLL_BASE
                + roll_offset)
        roll = (roll + np.pi / 2.0) % np.pi - np.pi / 2.0   # 180-deg symmetry
        return float(roll)

    def _grasp_roll(self, target, roll_offset=0.0, yaw="target"):
        """Roll of the gripper for a grasp at `target` (tilt-aware)."""
        rect = "target" if yaw == "target" else (None, None)
        closing, diff = self._tilt_plan(target, yaw=yaw, rect=rect)
        roll = self._roll_for_closing(target, closing, roll_offset)
        print("[grasp] closing=%s azimuth=%.2f -> roll=%.2f  fingertip "
              "height difference %.1f mm"
              % ("%.2f" % closing if closing is not None else "canonical",
                 float(np.arctan2(target[1], target[0])), roll,
                 diff * 1000))
        return roll

    def _grasp_heights(self, top, target, yaw="target", rect="target"):
        """Commanded midpoint height for the descent, chosen so the UPPER
        fingertip reaches the wanted depth on the block while the LOWER
        one stays above the table. Returns (grasp_z, diff)."""
        _, diff = self._tilt_plan(target, yaw=yaw, rect=rect)
        wanted_upper = top + GRASP_DZ            # where the upper tip should be
        mid = wanted_upper - diff / 2.0          # midpoint between the tips
        lowest_mid = TABLE_Z + FINGER_TABLE_MARGIN + diff / 2.0
        grasp_z = max(mid, lowest_mid)
        if grasp_z > mid + 1e-6:
            print("[grasp] descent limited by the table: upper fingertip "
                  "reaches %.1f mm below the top instead of %.1f mm"
                  % ((top - (grasp_z + diff / 2.0)) * 1000,
                     -GRASP_DZ * 1000))
        return float(grasp_z), diff

    def hover_over_target(self, dx=0.0, dy=0.0, roll_offset=0.0,
                          height=None):
        """Open the jaw and hold the gripper above the target, in exactly
        the pose the grasp would use (same xy, same roll). For diagnosis:
        diag_grasp.py photographs the finger pads against the block from
        here.

        Returns the commanded fingertip point, the FK pose actually
        reached, the roll, and the fingertip height difference of the
        tilted jaw (the lower fingertip sits half of it below the
        commanded point)."""
        if not self.target_visible:
            return {"ok": False, "reason": "no target"}
        top = float(np.clip(
            self.target_centroid[2] + self.target_extent[2] / 2.0,
            TABLE_Z + 0.03, 0.38))
        dz = PREGRASP_DZ if height is None else float(height)
        target = np.array([
            self.target_centroid[0] + GRASP_XY_OFFSET[0] + dx,
            self.target_centroid[1] + GRASP_XY_OFFSET[1] + dy, top])
        roll = self._grasp_roll(target, roll_offset)
        _, diff = self._tilt_plan(target)
        self._open_gripper()
        commanded = (target + [0, 0, dz]).tolist()
        ok = self._goto_xyz(target + [0, 0, dz], roll=roll)
        return {"ok": bool(ok), "commanded": commanded,
                "reached": self._fingertip_from_wrist(self.last_pitch)
                .tolist(),
                "wrist": self._ee_pos().tolist(),
                "roll": roll, "target_yaw": self.target_yaw,
                "tip_height_difference": float(diff),
                "centroid": self.target_centroid.tolist(),
                "motion_errors": list(self.motion_log)}

    def scripted_grasp_and_place(self, obstacle):
        """Clearing primitive: pick up `obstacle` (a dict with centroid,
        extent, optional yaw) and put it down at the drop point. Returns
        (carried, completed). Same grasp geometry as scripted_grasp, then a
        transfer at lift height, a short descent and a release."""
        centroid = np.asarray(obstacle["centroid"], dtype=float)
        # a pile is picked at its outer end (client-computed), a single
        # block at its centroid
        pick_at = np.asarray(obstacle.get("pick_centroid", centroid),
                             dtype=float)
        extent = np.asarray(obstacle.get("extent", [0.03, 0.03, 0.03]),
                            dtype=float)
        top = float(np.clip(centroid[2] + extent[2] / 2.0,
                            TABLE_Z + 0.03, 0.38))
        pick = np.array([pick_at[0] + GRASP_XY_OFFSET[0],
                         pick_at[1] + GRASP_XY_OFFSET[1], top])
        rect = (obstacle.get("rect_long"), obstacle.get("rect_short"))
        closing, _ = self._tilt_plan(pick, yaw=obstacle.get("yaw"), rect=rect)
        roll = self._roll_for_closing(pick, closing)
        self._open_gripper()
        ok1 = self._goto_xyz(pick + [0, 0, PREGRASP_DZ], roll=roll)
        grasp_z, _ = self._grasp_heights(top, pick, yaw=obstacle.get("yaw"),
                                         rect=rect)
        ok2 = self._goto_xyz([pick[0], pick[1], grasp_z],
                             roll=roll) if ok1 else False
        if ok2:
            self._close_gripper()
        ok3 = self._goto_xyz(pick + [0, 0, LIFT_DZ], roll=roll) if ok2 \
            else False
        carried = bool(ok3 and self._read_gripper_state() == HELD_STATE)
        drop = np.array([DROP_X, DROP_Y, top + LIFT_DZ])
        ok4 = self._goto_xyz(drop) if ok3 else False
        if ok4:
            self._goto_xyz([DROP_X, DROP_Y, TABLE_Z + RELEASE_DZ])
        self._open_gripper()
        self.holding = False
        if ok4:
            self._goto_xyz(drop)
        completed = bool(ok1 and ok2 and ok3 and ok4)
        print("[clear] carried=%s completed=%s obstacle=%s -> drop=%s"
              % (carried, completed, np.round(centroid, 3).tolist(),
                 [DROP_X, DROP_Y]))
        return carried, completed

    def scripted_grasp(self, roll_offset=0.0, xy_offset=(0.0, 0.0)):
        """Descend, close, lift. Returns (held, raw_state, completed).
        xy_offset is the policy's own correction to the grasp point."""
        top = float(np.clip(
            self.target_centroid[2] + self.target_extent[2] / 2.0,
            TABLE_Z + 0.03, 0.38))
        target = np.array([
            self.target_centroid[0] + GRASP_XY_OFFSET[0] + xy_offset[0],
            self.target_centroid[1] + GRASP_XY_OFFSET[1] + xy_offset[1],
            top])
        roll = self._grasp_roll(target, roll_offset)
        self._open_gripper()
        ok1 = self._goto_xyz(target + [0, 0, PREGRASP_DZ], roll=roll)
        grasp_z, _ = self._grasp_heights(top, target)
        ok2 = self._goto_xyz(
            [target[0], target[1], grasp_z], roll=roll) if ok1 else False
        if ok2:
            self._close_gripper()
        ok3 = self._goto_xyz(
            target + [0, 0, LIFT_DZ], roll=roll) if ok2 else False
        raw = self._read_gripper_state()
        held = bool(ok3 and raw == HELD_STATE)
        self.holding = bool(held)
        if held and HOLD_SECONDS > 0:
            # A slipping grasp reads held right after the lift and lets go
            # a moment later. Wait, then read the gripper again.
            time.sleep(HOLD_SECONDS)
            raw_after = self._read_gripper_state()
            if raw_after != HELD_STATE:
                print("[grasp] object slipped out during the %.1f s hold "
                      "(raw %d -> %d)" % (HOLD_SECONDS, raw, raw_after))
                held = False
                raw = raw_after
        completed = bool(ok1 and ok2 and ok3)
        self.holding = bool(held)
        closing, tip_diff = self._tilt_plan(target)
        self.last_grasp = {
            "pregrasp_ok": bool(ok1), "descend_ok": bool(ok2),
            "lift_ok": bool(ok3), "roll": float(roll),
            "closing": None if closing is None else float(closing),
            "tip_height_difference": float(tip_diff),
            "top": float(top), "grasp_z": float(grasp_z),
            "upper_tip_below_top": float(top - (grasp_z + tip_diff / 2.0)),
            "radial": float(np.hypot(target[0], target[1]))}
        if raw == HELD_STATE and not ok3:
            print("[grasp] the gripper closed on the object (raw=%d) but "
                  "the LIFT was refused, so the object is still on the "
                  "table and this does not count as held. Lower LIFT_DZ or "
                  "bring the block inside the reachable radius." % raw)
        print("[grasp] completed=%s raw_gripper_state=%d held=%s"
              % (completed, raw, held))
        return held, raw, completed

    @staticmethod
    def _point_segment_distance(point, start, end):
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom < 1e-9:
            return float(np.linalg.norm(point - start))
        fraction = np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0)
        return float(np.linalg.norm(point - (start + fraction * segment)))

    @staticmethod
    def _inside_table(xy):
        """Distance (m) from xy to the nearest edge of the stored table
        outline, positive inside. None when no outline is stored."""
        poly = CALIB_META.get("table_outline") if CALIB_META else None
        if not poly or len(poly) < 3:
            return None
        poly = np.asarray(poly, dtype=float)
        best = None
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            edge = b - a
            length = float(np.linalg.norm(edge))
            if length < 1e-9:
                continue
            # counter-clockwise outline: inside is to the left of each edge
            d = float(edge[0] * (xy[1] - a[1]) - edge[1] * (xy[0] - a[0])) \
                / length
            best = d if best is None else min(best, d)
        return best

    def _room_on_table(self, c, direction, keep, allowed):
        """How far (<= allowed) the object's centre may move along
        direction and stay `keep` inside the table outline. An object that
        already lies closer to the edge than `keep` may still move, but not
        closer to the edge than it is now."""
        start = self._inside_table(c)
        if start is None:
            return allowed
        floor = min(keep, start) - 1e-4
        t, ok = 0.0, 0.0
        while t <= allowed + 1e-9:
            d = self._inside_table(c + direction * t)
            if d is None or d < floor:
                break
            ok = t
            t += 0.005
        return min(ok, allowed) if t <= allowed + 1e-9 else allowed

    def scripted_push(self, obstacle, direction, distance, is_target=False):
        """Push one selected object in XY. Returns (completed, blocked).
        is_target: the object being pushed IS the target, so the rule that
        refuses a push whose path crosses the target does not apply."""
        # Reset first, so a push that is refused before it starts reports
        # steps=0 and a "refused" reason instead of the previous push's
        # numbers. The client charges a refused action (the arm did not
        # move) more than a plain step; see DEFAULT_W["refused"].
        self.last_push = {"steps": 0, "stopped": "", "advance": 0.0,
                          "refused": ""}
        direction = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm < 0.15:
            direction = (obstacle["centroid"][:2]
                         - self.target_centroid[:2])
            norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            direction = np.array([1.0, 0.0])
        else:
            direction = direction / norm

        center = obstacle["centroid"][:2]
        half_width = 0.5 * (
            abs(direction[0]) * obstacle["extent"][0]
            + abs(direction[1]) * obstacle["extent"][1])
        object_end = center + direction * distance
        if PROTECT_TARGET and not is_target and self.target_visible:
            target_radius = 0.5 * float(np.linalg.norm(
                self.target_extent[:2])) + PUSH_TARGET_MARGIN
            # Refuse only a push that brings the object CLOSER to the
            # target. An obstacle that already touches the target is
            # inside this radius at its start, and the old test (any point
            # of the path within the radius) refused every push of it,
            # including the sideways push that frees the target.
            closest = self._point_segment_distance(
                self.target_centroid[:2], center, object_end)
            start_gap = float(np.linalg.norm(
                np.asarray(center, dtype=float) - self.target_centroid[:2]))
            # 1 cm of slack: a push meant to go sideways is never exactly
            # perpendicular, and a touching neighbour then comes a few mm
            # closer at the start of the stroke.
            if closest < target_radius and closest < start_gap - 0.010:
                print("[push] blocked: predicted path crosses target")
                self.last_push["refused"] = "path crosses the target"
                return False, True

        start_xy = center - direction * (half_width + PUSH_TOOL_OFFSET)
        # The reach scan measured 0.42 m at pitch 1.0. The start point may
        # lie PUSH_START_SLACK beyond MAX_RADIAL; whether the arm really
        # gets there is then the IK's decision (a refusal costs no motion).
        if float(np.linalg.norm(start_xy)) > MAX_RADIAL + PUSH_START_SLACK:
            self._log_motion(
                "push start %s is at radial %.2f, beyond %.2f"
                % (np.round(start_xy, 3).tolist(),
                   float(np.linalg.norm(start_xy)),
                   MAX_RADIAL + PUSH_START_SLACK))
            self.last_push["refused"] = "start beyond reach"
            return False, True
        # Where the fingertips stop. The closed jaw meets the object at its
        # NEAR face (centre - half_width) and should carry that face forward
        # by `distance`, so the object moves by `distance`.
        # Until build 2026-10-09a the stop point was centre + half_width +
        # distance: the fingertips ploughed through the whole object and
        # then went `distance` further. A 3 cm block moved 9 cm for a "6 cm"
        # push, and a 12 cm pile was swept for 18 cm, which is a blow, not
        # a nudge, and made the displacement depend on the object's size.
        # PUSH_FROM_FACE=0 brings the old stroke back.
        reach_past_centre = ((distance - half_width) if PUSH_FROM_FACE
                             else (half_width + distance))
        end_xy = center + direction * reach_past_centre
        end_radial = float(np.linalg.norm(end_xy))
        if end_radial > MAX_RADIAL:
            # Shorten the push so the fingertips stop at the reach limit.
            # Solve |center + direction * t| = MAX_RADIAL for the largest t.
            c = np.asarray(center, dtype=float)
            b = float(np.dot(c, direction))
            disc = b * b - (float(np.dot(c, c)) - MAX_RADIAL ** 2)
            t = (-b + np.sqrt(max(disc, 0.0)))
            t = max(t, reach_past_centre - distance + PUSH_MIN_DISTANCE)
            end_xy = c + direction * t
            print("[push] end shortened to radial %.2f (was %.2f)"
                  % (float(np.linalg.norm(end_xy)), end_radial))
        # Keep the pushed object on the usable table. Two limits:
        #  * the inner reach limit MIN_RADIAL (+2 cm, a pushed block slides
        #    on a little): near the robot the arm hit the table going home;
        #  * the edges of the work area (the table as calibrate_all.py saw
        #    it) minus half the object and 2 cm. The table's front edge is a
        #    straight line, so at the sides of the table it is farther than
        #    MIN_RADIAL and the radial limit alone does not protect it.
        # The object's centre moves by about `distance`. If it would end
        # outside, the stroke is shortened; refused if almost nothing is
        # left.
        c = np.asarray(center, dtype=float)
        half = 0.5 * float(np.linalg.norm(obstacle["extent"][:2]))
        allowed, why = distance, ""
        # Oct 29: the only limit on where a pushed object may END is that
        # it stays ON THE TABLE (the outline calibrate_all.py stored):
        # its centre at least half its size + 1 cm inside the edge. The
        # inner-radius and work-area-box limits are gone; where a block
        # ends up is the reward's business (near_edge, outreach, ...).
        keep = half + 0.01
        if self._inside_table(c) is None:
            # no outline stored (older calibration): fall back to the box
            for axis, (lo, hi) in ((0, WS_X), (1, WS_Y)):
                d = float(direction[axis])
                if d < -1e-6:
                    t = max(0.0, (c[axis] - lo - keep) / -d)
                elif d > 1e-6:
                    t = max(0.0, (hi - keep - c[axis]) / d)
                else:
                    continue
                if t < allowed:
                    allowed, why = t, "off the edge of the work area"
        else:
            on_table = self._room_on_table(c, direction, keep, allowed)
            if on_table < allowed:
                allowed, why = on_table, "off the edge of the table"
        if allowed < distance:
            if allowed < 0.5 * PUSH_MIN_DISTANCE:
                self._log_motion(
                    "push refused: it would move the object at %s %s"
                    % (np.round(c, 3).tolist(), why))
                self.last_push["refused"] = "toward the table edge"
                return False, True
            shorten = distance - allowed
            end_xy = end_xy - direction * shorten
            print("[push] shortened by %.0f mm: the object would end %s"
                  % (shorten * 1000, why))
        if self._inside_table(c) is None:
            # Only without a table outline: keep the fingers in the box.
            # (Clipping the START into the box moves it toward the object
            # and can put the fingers down on the object itself, so with an
            # outline the points are left as computed.)
            margin = 0.015
            start_xy[0] = np.clip(start_xy[0], WS_X[0] + margin,
                                  WS_X[1] - margin)
            start_xy[1] = np.clip(start_xy[1], WS_Y[0] + margin,
                                  WS_Y[1] - margin)
            end_xy[0] = np.clip(end_xy[0], WS_X[0] + margin, WS_X[1] - margin)
            end_xy[1] = np.clip(end_xy[1], WS_Y[0] + margin, WS_Y[1] - margin)

        # The fingers come straight down at start_xy. If another object
        # lies there, the arm would press down on it with all it has; that
        # is the hardest "push" this robot can give and it is never wanted.
        # Refuse instead: the policy has to come at the object from a side
        # where there is room for the fingers.
        # (Build 29e pushed with the jaw open and checked both open
        # fingertips here; 29f is back to a closed jaw, one contact point.)
        start_xy = np.asarray(start_xy, dtype=float)
        finger_drop = 0.0
        push_z = PUSH_EE_Z
        occupied = self._object_at(start_xy, center)
        if occupied is not None:
            self._log_motion(
                "push refused: the fingers would come down at %s, on top "
                "of another object at %s. Push from a side that has room."
                % (np.round(start_xy, 3).tolist(),
                   np.round(occupied, 3).tolist()))
            self.last_push["stopped"] = "no room for the fingers"
            self.last_push["refused"] = "no room for the fingers"
            return False, True

        # Fingers CLOSED before the arm comes down (the arm is still at
        # home here): one contact point, no finger slipping past the block.
        #
        # Oct 29f, why the gripper motor (ID 7) tripped "Overload error":
        # locobot_controller closes the jaw in PWM mode and sets the power
        # to 0 once the fingers are shut (/gripper/state = 3, checked on
        # the robot on Oct 29), so a closed jaw does NOT load the motor.
        # Opening is different: it is a position move held at full
        # strength, and if a finger meets a block or the table on its way
        # open, the motor pushes against it until it trips (the error came
        # 4 s after "Gripper open"). Until 29e the jaw was opened right
        # after the stroke, also when the arm had NOT got back up (the lift
        # at the far end of the reach is often refused by the IK), i.e.
        # with the fingers still down next to the pushed block.
        # So from 29f the jaw is opened only at home (go_episode_home).
        # 29g: that open at home tripped the motor too, so with
        # JAW_STAYS_CLOSED (default) the jaw is closed once and left
        # closed; a jaw that is already closed gets no new command.
        if self.gripper_flag < 0.5:
            self._close_gripper()
        up = push_z + PUSH_APPROACH_DZ
        sweep = np.asarray(end_xy, dtype=float) - start_xy
        length = float(np.linalg.norm(sweep))
        along = sweep / length if length > 1e-6 else direction

        def at(travel, z=push_z):
            xy = start_xy + along * travel
            return [float(xy[0]), float(xy[1]), z]

        def tip_along():
            return float(np.dot(
                self._fingertip_from_wrist(self.last_pitch)[:2], along))

        def tip_z():
            return float(self._fingertip_from_wrist(self.last_pitch)[2])

        # Height hold. A plain move (goto, the hover check) re-commands
        # once with the residual, so it ends where it was sent. The push
        # stroke moves in 4 mm steps WITHOUT that re-command, and under
        # load the arm settles LOWER than sent: on Oct 28-29 a stroke
        # commanded 1-2 cm above the table ran on the table. So the
        # stroke measures how much lower it is (from the arm's encoders)
        # and raises the command by that much. Only ever UP: a correction
        # downward could press into the table.
        lift = [0.0]
        heights = []

        def hold_height():
            low = push_z - tip_z()
            # recorded: the LOWER fingertip above the table
            heights.append(tip_z() - finger_drop - TABLE_Z)
            if low > 0.002:
                lift[0] = min(lift[0] + low, PUSH_MAX_LIFT)

        def touching():
            # fingertips within PUSH_TABLE_GAP of the table top, or held
            # higher than commanded (pressed up by something)
            z = tip_z()
            return (z - finger_drop < TABLE_Z + PUSH_TABLE_GAP
                    or z > push_z + lift[0] + PUSH_CONTACT_TOL)

        def at_h(travel, z=None):
            return at(travel, push_z + lift[0] if z is None else z)

        ok1 = self._goto_xyz(at(0.0, up), any_xy=True)
        # Descend WITHOUT the residual re-command (it could drive the
        # fingers into the table), then raise by however much it sagged.
        ok2 = (self._goto_xyz(at_h(0.0), correct=False, any_xy=True)
               if ok1 else False)
        travel, stopped, steps = 0.0, "", 0
        if ok2:
            hold_height()
            if lift[0] > 0.0:
                ok2 = self._goto_xyz(at_h(0.0), correct=False, quiet=True,
                                     any_xy=True)
                hold_height()
        if ok2 and touching():
            stopped = ("the fingertips are at the table on the way down "
                       "(%.0f mm above it, %.0f mm wanted): no stroke"
                       % ((tip_z() - finger_drop - TABLE_Z) * 1000,
                          (PUSH_EE_Z - TABLE_Z) * 1000))
            self.last_push["table_contact"] = True
        if ok2 and not stopped:
            # free travel: up to 1 cm before the face, in one move
            free = min(max(0.0, PUSH_TOOL_OFFSET - 0.010), length)
            if free > PUSH_STEP:
                if self._goto_xyz(at_h(free), settle=PUSH_STEP_PAUSE,
                                  correct=False, quiet=True, any_xy=True):
                    travel = free
                    hold_height()
                else:
                    stopped = "the arm could not start the stroke"
            # contact: small steps, a stop after each
            last, slow = tip_along(), 0
            step = max(PUSH_STEP, 0.001)
            while not stopped and travel < length - 1e-6:
                target = min(length, travel + step)
                reached = self._goto_xyz(at_h(target),
                                         settle=PUSH_STEP_PAUSE,
                                         correct=False, quiet=True,
                                         any_xy=True)
                now = tip_along()
                moved, last = now - last, now
                steps += 1
                if not reached:
                    stopped = ("the arm did not reach the next step "
                               "(reach limit or something in the way)")
                    break
                travel = target
                if touching():
                    stopped = ("the fingertips touch the table (%.0f mm "
                               "above it, %.0f mm wanted): stroke stopped"
                               % ((tip_z() - finger_drop - TABLE_Z) * 1000,
                                  (PUSH_EE_Z - TABLE_Z) * 1000))
                    self.last_push["table_contact"] = True
                    break
                hold_height()
                slow = slow + 1 if moved < 0.25 * step else 0
                if PUSH_STALL_STEPS > 0 and slow >= PUSH_STALL_STEPS:
                    stopped = ("the fingertips stopped advancing for %d "
                               "steps: something does not give way"
                               % slow)
            if stopped:
                self._log_motion(
                    "push ended early after %.0f of %.0f mm: %s"
                    % (travel * 1000, length * 1000, stopped))
            # always back up and out, also after an early end
            ok4 = self._goto_xyz(at(travel, up), any_xy=True)
        else:
            ok4 = False
        if heights:
            print("[push] fingertip height above the table during the "
                  "stroke: %.0f..%.0f mm (wanted %.0f mm, command raised "
                  "by %.0f mm to hold it)"
                  % (min(heights) * 1000, max(heights) * 1000,
                     (PUSH_EE_Z - TABLE_Z) * 1000, lift[0] * 1000))
        completed = bool(ok1 and ok2 and not stopped and ok4)
        # The jaw stays CLOSED here (no force on the motor, see above). It
        # is opened when the arm is back at home, where nothing is near
        # the fingers (go_episode_home).
        contact = bool(self.last_push.get("table_contact", False))
        self.last_push = {"steps": steps, "stopped": stopped,
                          "table_contact": contact,
                          "advance": float(max(
                              0.0, travel - PUSH_TOOL_OFFSET)),
                          # the arm never got to the start point: the IK or
                          # the workspace check refused the approach
                          "refused": ("" if ok2 else
                                      "the arm refused the approach "
                                      "(IK / workspace)")}
        print("[push] completed=%s direction=%s distance=%.3f (closed jaw, "
              "%d steps of %.0f mm%s)"
              % (completed, np.round(direction, 3).tolist(), distance, steps,
                 PUSH_STEP * 1000,
                 ", ENDED EARLY" if stopped else ""))
        return completed, False

    def _object_at(self, xy, acted_xy):
        """Centre of a known object whose footprint (plus the fingers'
        clearance) covers xy, ignoring the object being acted on."""
        known = []
        if self.target_visible:
            known.append((self.target_centroid, self.target_extent))
        for entry in list(self.obstacles) + list(self.pushables or []):
            known.append((entry["centroid"], entry["extent"]))
        for centroid, extent in known:
            centroid = np.asarray(centroid, dtype=float)
            extent = np.asarray(extent, dtype=float)
            if float(np.linalg.norm(centroid[:2] - np.asarray(
                    acted_xy, dtype=float))) < 0.005:
                continue
            if (abs(xy[0] - centroid[0]) < extent[0] / 2.0
                    + PUSH_FINGER_CLEARANCE
                    and abs(xy[1] - centroid[1]) < extent[1] / 2.0
                    + PUSH_FINGER_CLEARANCE):
                return centroid[:2]
        return None

    # ---------- env API ----------

    def _release_held_object(self):
        """If the gripper is closed on something, put it back on the table
        at the release point before going home. Going home with a block in
        the fingers dragged it across the table edge and the home motion
        failed; releasing first also means a successful episode resets
        itself, no human needed to replace the block."""
        if not self.holding:
            # A closed jaw after a push is not a held object. It is opened
            # at home (go_episode_home), not here: here the fingers may
            # still be down next to a block, and opening into a block is
            # what overloads the gripper motor (Oct 29f).
            return
        release = np.array([RELEASE_X, RELEASE_Y, TABLE_Z + RELEASE_DZ])
        print("[reset] gripper is closed; releasing at %s"
              % np.round(release, 3).tolist())
        moved = self._goto_xyz(release)
        if not moved:
            print("[reset] could not reach the release point; opening the "
                  "gripper where the arm is")
        self._open_gripper()
        self.holding = False
        time.sleep(0.5)
        if moved:
            # back off upward so the fingers do not clip the block on the
            # way home
            self._goto_xyz(release + [0, 0, 0.04])

    def reset(self):
        self.t = 0
        self.motion_log = []
        self._release_held_object()
        if not self.go_episode_home():
            # The most common reason is something in the gripper or a
            # collision on the way; open and try once more before giving up.
            print("[reset] home motion failed; opening the gripper and "
                  "retrying once")
            self._open_gripper()
            time.sleep(1.0)
            if not self.go_episode_home():
                raise RuntimeError(
                    "failed to move to episode home twice; check the arm "
                    "for a collision or a jammed joint")
        time.sleep(1.0)
        if not JAW_STAYS_CLOSED:
            self._open_gripper()
        self.target_centroid = np.array(
            [TARGET_X, TARGET_Y, TARGET_TOP_Z - 0.03])
        self.target_extent = np.zeros(3)
        self.update_perception(reset_target=True)
        return {"obs": self._obs(), "rgbd": self._rgbd_obs(),
                "target": self.target_centroid.tolist(),
                "graspable": self.target_graspable(),
                "home_joints": (None if self.home_joints is None else
                                self.home_joints.tolist())}

    def step(self, action, perception=None, grasp_role="target"):
        """Execute one primitive and report the physical outcome only.

        grasp_role decides what action[0] >= 0 means:
          "target"   - the scripted grasp of the target (the planner; used
                       by the client when the target is judged graspable)
          "obstacle" - clearing: grasp obstacle action[1] and put it at the
                       drop point (a policy action, like push)

        This server is a DUMB EXECUTOR. It does not decide anything, it does
        not score anything, and it never blocks an action:

          * The policy (route 1) on the client chooses the action. The server
            executes whatever it is told.
          * Route 2 (YOLO/DINOv2 on the client) supplies the target/obstacle
            centroids through ``perception``. The server grasps or pushes at
            those coordinates. If route 2 sees no target, a grasp still runs
            as a blind attempt at the last known location; it simply fails,
            which is the correct feedback for the policy to learn from.
          * There is NO graspable gate. There are NO reward penalties. The
            server returns the raw gripper result; the client computes the
            reward ("gripper held AND target gone from the table") and decides
            ``done``.

        The returned ``done`` only reflects the server's step budget
        (``HORIZON``); episode success/termination is the client's call.
        """
        self.t += 1
        self.motion_log = []
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if a.size not in (5, 6, 8):
            raise ValueError("action must have 5, 6 or 8 values")
        # action[5] is the learned roll offset (+-GRASP_ROLL_RANGE rad).
        # action[6:8] is the learned grasp-point offset (+-GRASP_XY_RANGE m).
        # Shorter actions keep the purely geometric pose, so older clients
        # still work.
        roll_offset = (float(a[5]) * GRASP_ROLL_RANGE if a.size >= 6
                       else 0.0)
        xy_offset = ((float(a[6]) * GRASP_XY_RANGE,
                      float(a[7]) * GRASP_XY_RANGE) if a.size == 8
                     else (0.0, 0.0))
        external = perception is not None
        if external:
            self.apply_external_perception(perception)
        done = False
        info = {"t": self.t, "held": False,
                "external_perception": external}

        if a[0] < 0.0:
            info["primitive"] = "push"
            pool = (self.pushables if (external and self.pushables is not None)
                    else self.obstacles)
            if not pool:
                # No obstacle to push. Not an error and not penalised here;
                # the client applies the uniform push step cost regardless.
                info["no_obstacle"] = True
            else:
                selector = 0.5 * (a[1] + 1.0)
                obstacle_index = min(int(selector * len(pool)),
                                     len(pool) - 1)
                chosen = pool[obstacle_index]
                is_target = bool(chosen.get("is_target", False))
                distance = PUSH_MIN_DISTANCE + 0.5 * (a[4] + 1.0) * (
                    PUSH_MAX_DISTANCE - PUSH_MIN_DISTANCE)
                completed, blocked = self.scripted_push(
                    chosen, a[2:4], distance, is_target=is_target)
                info.update({"obstacle_index": obstacle_index,
                             "pushed_target": is_target,
                             "push_completed": completed,
                             "push_blocked": blocked,
                             "push_distance": float(distance),
                             "push_steps": self.last_push.get("steps", 0),
                             "push_stopped": self.last_push.get(
                                 "stopped", ""),
                             "push_advance": self.last_push.get(
                                 "advance", 0.0),
                             # non-empty: nothing was pushed, and why
                             "push_refused": self.last_push.get(
                                 "refused", ""),
                             "table_contact": bool(self.last_push.get(
                                 "table_contact", False))})
            if not external:
                self.update_perception(reset_target=False)
        else:
            if grasp_role == "obstacle":
                info["primitive"] = "clear"
                if not self.obstacles:
                    info["no_obstacle"] = True
                else:
                    selector = 0.5 * (a[1] + 1.0)
                    obstacle_index = min(
                        int(selector * len(self.obstacles)),
                        len(self.obstacles) - 1)
                    carried, completed = self.scripted_grasp_and_place(
                        self.obstacles[obstacle_index])
                    info.update({"obstacle_index": obstacle_index,
                                 "carried": carried,
                                 "clear_completed": completed,
                                 "drop_xy": [DROP_X, DROP_Y]})
            else:
                info["primitive"] = "grasp"
                # No gate: the grasp always runs. With no visible target it
                # is a blind attempt at the last known position and fails.
                info["target_visible"] = bool(self.target_visible)
                held, raw, completed = self.scripted_grasp(
                    roll_offset, xy_offset)
                info["roll_offset"] = roll_offset
                info["grasp_dx"] = xy_offset[0]
                info["grasp_dy"] = xy_offset[1]
                info.update({"held": held, "gripper_raw": raw,
                             "grasp_completed": completed,
                             "grasp_plan": dict(self.last_grasp)})
            if not external:
                self.update_perception(reset_target=False)

        if self.t >= HORIZON and not done:
            done = True
            info["time_limit"] = True
        info.update({"clearance": self.target_clearance,
                     "obstacle_count": len(self.obstacles),
                     "motion_errors": list(self.motion_log)})

        # No reward here. The client computes it from the gripper result plus
        # its own route-2 visual check. A reward field is still returned for
        # protocol compatibility, always 0.0, and the client ignores it.
        return {"obs": self._obs(), "rgbd": self._rgbd_obs(),
                "reward": 0.0,
                "done": done, "info": info}


def serve():
    env = GraspEnv()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.settimeout(None)                       # PyRobot timeout fix
    srv.bind((HOST, PORT))
    srv.listen(1)
    print("[env] listening on %s:%d" % (HOST, PORT))

    while True:
        conn, addr = srv.accept()
        conn.settimeout(None)                  # PyRobot timeout fix
        print("[env] client connected: %s" % (addr,))
        rfile = conn.makefile("r")
        try:
            for line in rfile:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    cmd = msg.get("cmd")
                    if cmd == "ping":
                        resp = {"ok": True, "build": SERVER_BUILD,
                                "file": os.path.abspath(__file__)}
                    elif cmd == "get_config":
                        resp = {"ok": True, "config": current_config(),
                                "meta": dict(CALIB_META,
                                             saved_at=CALIB_SAVED_AT),
                                "calib_file": CALIB_FILE,
                                "calib_loaded": bool(CALIB_LOADED),
                                "ik_numerical": bool(IK_NUMERICAL),
                                "gripper_length": GRIPPER_LENGTH,
                                "held_state": HELD_STATE,
                                "background": {
                                    "pixels": (0 if BACKGROUND["depth"]
                                               is None else int(
                                                   (BACKGROUND["depth"]
                                                    > 0).sum())),
                                    "saved_at": BACKGROUND["saved_at"],
                                    "file": BACKGROUND_FILE},
                                "build": SERVER_BUILD}
                    elif cmd == "set_config":
                        changed = apply_config(msg.get("values") or {})
                        resp = {"ok": True, "changed": changed,
                                "config": current_config()}
                    elif cmd == "save_config":
                        path = save_calibration_file(msg.get("meta"))
                        resp = {"ok": True, "file": path}
                    elif cmd == "reset":
                        resp = env.reset()
                        resp["motion_errors"] = list(env.motion_log)
                    elif cmd == "step":
                        resp = env.step(msg["action"],
                                        msg.get("perception"),
                                        msg.get("grasp_role", "target"))
                    elif cmd == "ee":
                        resp = {"ee": env._ee_pos().tolist()}
                    elif cmd == "hover":
                        env.motion_log = []
                        if msg.get("perception") is not None:
                            env.apply_external_perception(msg["perception"])
                        resp = env.hover_over_target(
                            dx=float(msg.get("dx", 0.0)),
                            dy=float(msg.get("dy", 0.0)),
                            roll_offset=float(msg.get("roll_offset", 0.0)),
                            height=msg.get("height"))
                    elif cmd == "gripper":
                        # Open or close the jaw in place, no arm motion.
                        # Used by diag_grasp.py to measure the jaw stroke.
                        if msg.get("state") == "close":
                            env._close_gripper()
                        else:
                            env._open_gripper()
                        resp = {"ok": True,
                                "raw": int(env._read_gripper_state())}
                    elif cmd == "goto":
                        env.motion_log = []
                        ok = env._goto_xyz(msg["xyz"],
                                           pitch=msg.get("pitch"),
                                           roll=msg.get("roll"))
                        # "ee" is the FINGERTIP estimate, the same frame the
                        # commanded xyz is in; the wrist is reported too.
                        resp = {"ok": ok,
                                "ee": env._fingertip_from_wrist(
                                    env.last_pitch).tolist(),
                                "wrist": env._ee_pos().tolist(),
                                "motion_errors": list(env.motion_log)}
                    elif cmd == "goto_pregrasp":
                        env.update_perception()
                        tgt = env.current_pregrasp()
                        ok = env._goto_xyz(tgt)
                        resp = {"ok": ok, "ee": env._ee_pos().tolist(),
                                "pregrasp": tgt.tolist()}
                    elif cmd == "grasp_test":
                        env.update_perception(reset_target=True)
                        held, raw, completed = env.scripted_grasp()
                        resp = {"held": held, "gripper_raw": raw,
                                "completed": completed}
                    elif cmd == "pc_stats":
                        resp = env.pc_stats()
                    elif cmd == "pc_save":
                        resp = env.pc_save()
                    elif cmd == "rgbd_stats":
                        resp = env.rgbd_stats()
                    elif cmd == "frame":
                        resp = env.full_frame(raw=bool(msg.get("raw")))
                    elif cmd == "set_background":
                        background = decode_packed(msg["depth"]).astype(
                            np.uint16)
                        BACKGROUND["depth"] = background
                        BACKGROUND["saved_at"] = time.strftime(
                            "%Y-%m-%d %H:%M:%S")
                        BACKGROUND["joints"] = env._joints().astype(float)
                        np.savez_compressed(
                            BACKGROUND_FILE, depth=background,
                            saved_at=BACKGROUND["saved_at"],
                            joints=BACKGROUND["joints"])
                        print("[background] stored %d pixels in %s"
                              % (int((background > 0).sum()),
                                 BACKGROUND_FILE))
                        resp = {"ok": True, "file": BACKGROUND_FILE,
                                "pixels": int((background > 0).sum())}
                    elif cmd == "capture_home":
                        joints = env.capture_current_home()
                        resp = {"ok": True, "home_joints": joints.tolist()}
                    elif cmd == "home":
                        ok = env.go_episode_home()
                        resp = {"ok": ok,
                                "home_joints": (
                                    None if env.home_joints is None else
                                    env.home_joints.tolist())}
                    elif cmd == "camera":
                        ok = env.set_camera(float(msg.get("pan", CAM_PAN)),
                                            float(msg.get("tilt", CAM_TILT)))
                        resp = {"ok": ok}
                    elif cmd == "close":
                        conn.sendall(b'{"ok": true}\n')
                        print("[env] shutdown requested")
                        return
                    else:
                        resp = {"error": "unknown cmd: %s" % cmd}
                except Exception as e:
                    traceback.print_exc()
                    resp = {"error": str(e)}
                conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
        finally:
            print("[env] client disconnected")
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    serve()
