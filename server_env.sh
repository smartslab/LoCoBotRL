# server_env.sh  --  environment for grasp_env_server_rgbd.py on the robot
#
# Usage (robot terminal 2, every time the server is started):
#     source ~/server_env.sh && python ~/grasp_env_server_rgbd.py
#
# The server prints two "[env] ..." lines at startup that echo the values
# below. Check them after every restart; a stale value there was the cause
# of more than one wasted afternoon.
#
# Every number in this file was measured on this robot in September 2026,
# with the white table LOWERED (20.5 cm above the floor) and the robot
# placed with the table on its right. Re-run measure_workspace.py after
# moving the robot or the table.

# ---------------- home pose ----------------------------------------------
# Joint angles the arm returns to at reset and before the success-check
# frame. All zeros = arm horizontal, pointing forward, out of the camera's
# right-facing view.
export ARM_HOME_JOINTS="0.0,0.0,0.0,0.0,0.0"

# ---------------- workspace (camera / TF frame) ---------------------------
# Table height as the camera sees it (strongest plane in measure_workspace).
export TABLE_Z=0.166
# Box the primitives may move inside. Also the point-cloud crop. Sized to
# the reachable band around the blocks; a push aimed outside it is refused.
export WS_X_MIN=0.10
export WS_X_MAX=0.38
export WS_Y_MIN=-0.36
export WS_Y_MAX=-0.09
export WS_Z_MAX=0.366
# Layout reference only (colour identity does not use it for the target).
export TARGET_X=0.27
export TARGET_Y=-0.20

# ---------------- arm frame vs camera frame -------------------------------
# Vertical offset between the frame the arm reports/targets and the
# physical fingertips, in the camera frame. 0.040 is the pure frame offset
# (tape measure at home); the remaining 0.110 was measured at pitch 1.0 by
# hovering and reading the fingertip-to-table gap with a ruler. With THIS
# value and no lateral correction the fingers landed on the block
# (numerical IK, pitch 1.0); that is the empirical anchor for everything
# below. Re-measure with grasp_tune.py --height 0.0 after changing pitch.
export ARM_Z_OFFSET=0.150
# Lateral fingertip correction is OFF. A hover-and-grasp comparison showed
# the frame the arm targets already sits at the fingertips horizontally:
# adding the +-L*cos(pitch) radial shift moved the fingers several cm off
# the block. Only the vertical part is needed, and it lives in ARM_Z_OFFSET.
export GRIPPER_LENGTH=0.0
export GRIPPER_TCP_SIGN=1.0
# 1 = numerical solver. It was the solver in use when the fingers landed on
# the block. The analytical solver (0) reaches straight-down poses the
# numerical one refuses, but its target frame has not been checked against
# a ruler yet - switch only after a grasp_tune hover confirms the fingers
# still land on the block with it.
export IK_NUMERICAL=1

# ---------------- grasp primitive (heights relative to the fingertips) ----
# Gripper pitch. 1.0 rad (57 deg) is the ONLY value that has grasped a
# block on this robot: at radial 0.39 it reached, closed and held. It is
# also the pitch ARM_Z_OFFSET was measured at, and the reach scan at this
# table height shows it feasible to radial 0.36 at +0.12 m (0.39 by the
# earlier grasp). 1.4 reaches only to 0.34 and refused every pose at 0.41.
# The finger pads are tilted 33 deg at 1.0; the hold was still solid.
export GRASP_PITCH=1.0
# Hover above the block top before descending: +0.03 above a 3 cm block is
# +0.06 above the table, inside the feasible envelope at this pitch.
export PREGRASP_DZ=0.03
# Lift height above the block top after closing. A 0.10 lift was refused
# by IK at radial 0.39 and turned a real hold into held=False.
export LIFT_DZ=0.05
# Fingertip depth relative to the block top when closing (negative = below
# the top). The previous -0.02 closed only on the tips of a 3 cm block;
# -0.025 reaches the middle. Floor-clamped to TABLE_Z + 0.015 in the server.
export GRASP_DZ=-0.025

# ---------------- push primitive ------------------------------------------
# Fingertip height during the push sweep: table + 2 cm, i.e. the upper part
# of a 3 cm block. (The old default, table + 4.5 cm, sails over a 3 cm
# block.)
export PUSH_EE_Z=0.186
# Height above PUSH_EE_Z for the approach and retreat points.
export PUSH_APPROACH_DZ=0.04
# Gentle push: 6 cm max, executed as 2 cm segments with a short pause, with
# the jaw CLOSED. (10 cm in one stroke scattered a pile of 3 cm blocks.)
export PUSH_MAX_DISTANCE=0.06
export PUSH_SEGMENT=0.02
export PUSH_SEGMENT_PAUSE=0.25

# ---------------- gripper roll (finger alignment) -------------------------
# auto: fingers across the target's short side when it has one (PCA yaw);
# square footprints report no yaw and use the canonical roll.
export GRASP_ROLL_MODE=auto
export GRASP_ROLL_SIGN=1.0
# pi/2: this gripper's roll is measured on the finger-opening axis, so
# without this the fingers came down ALONG the long side (observed).
export GRASP_ROLL_BASE=1.5708
# The policy's action[5] shifts the roll by up to this many radians (45 deg).
export GRASP_ROLL_RANGE=0.785
# The policy's action[6:8] shift the grasp point by up to this many metres.
export GRASP_XY_RANGE=0.03
# Fixed grasp-point correction found with grasp_tune.py. Zero so far.
export GRASP_DX=0.0
export GRASP_DY=0.0

# ---------------- reset: put a held object back ---------------------------
# After a successful grasp the reset moves to this point, opens the gripper
# and only then goes home, so the block is back on the table without a
# human replacing it. Defaults to the anchor if unset.
export RELEASE_X=0.27
export RELEASE_Y=-0.20
export RELEASE_DZ=0.05

# ---------------- clearing-only MDP ---------------------------------------
# Clearing decisions per episode. The rule-triggered planner grasp also
# takes a server step, so leave room for it.
export HORIZON=8
# After the lift the object must stay in the gripper this long to count.
export HOLD_SECONDS=2.0
# Where a cleared obstacle is put down by the pick-and-place primitive.
# Three requirements: on the table; inside the reachable band (radial
# 0.26-0.36); and INSIDE THE CAMERA VIEW and the workspace crop, otherwise
# the cleared object vanishes from perception and the displacement log
# and the object count go wrong. Adjust to your layout.
export DROP_X=0.16
export DROP_Y=-0.30
# Reach limit for pushes at the configured pitch (reach_scan). A push that
# would end beyond it is shortened to stop at the limit; one that starts
# beyond it is refused.
export MAX_RADIAL=0.38

# ---------------- misc ----------------------------------------------------
# Re-issue a goto once if the arm stops more than this far from the target.
export GOTO_RESIDUAL_TOL=0.008

# ---------------- tilt-aware grasp ----------------------------------------
# Jaw opening when fully open; used to compute the fingertip height
# difference of a tilted grasp and compensate for it.
export GRIPPER_OPENING=0.045
# Lowest the lower fingertip may go above the table.
export FINGER_TABLE_MARGIN=0.004
