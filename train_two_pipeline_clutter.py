# -*- coding: utf-8 -*-
"""train_two_pipeline_clutter.py

Real-robot RL for target grasping in clutter. Runs on the TRAINING machine
(Jetson, Python 3.8 + torch). The robot runs grasp_env_server_rgbd.py, which
is a dumb executor: it moves the arm and reports the gripper state, and
nothing else.

Two pipelines from one RGB-D frame
----------------------------------
  Route 1 -> the policy.  Raw RGB + depth pixels, plus 9-D proprioception
      (5 joints, 3 EE position, 1 gripper flag), feed phi_s. The factored
      critic is Q(s,a) = head(phi_s(s) * phi_a(a)). Route 1 sees NO
      centroids, NO labels, NO object identity, NO clearance: the 36
      perception entries of the server's 45-D state vector are stripped
      before the networks. That is deliberate. The claim this project tests
      is that freezing phi_a and retraining phi_s adapts to new objects with
      fewer real-robot samples, and that only means something if phi_s
      carries object-specific structure worth relearning. Feeding it
      pre-clustered centroids would strip exactly that out.

  Route 2 -> execution and scoring.  route2_perception.py turns the same
      frame into per-object centroids in the base frame. They tell the arm
      where to grasp or push, and they answer whether the target has left the
      table. Route 2 never enters the observation.

Who decides what
----------------
  * The policy learns CLEARING only. Its 5-D action is
        a[0] < 0  : push object a[1] in direction a[2:4] by distance a[4];
                    the pushable list is [target (if visible)] + obstacles,
                    so pushing the target out of a pile is allowed
        a[0] >= 0 : pick up obstacle a[1] and put it at the drop point
    It never chooses to grasp the target.
  * After every clearing action route 2 re-perceives and applies a
    geometric rule (grasp_feasibility): target identified, clearance to the
    nearest obstacle >= 3.5 cm, inside the arm's reach. When the rule holds
    the environment runs the scripted grasp planner (the same one
    grasp_once.py exercises) on the target, holds it for HOLD_SECONDS, and
    the episode ends in success if the gripper still reports a hold AND the
    camera sees no red object left on the table.
  * --grasp_trigger policy switches to the older design where a[0] >= 0
    means "grasp the target now" (still via the planner). Kept for
    ablation; the default is auto.

  The point of this split: the paper is about learning to MAKE a target
  graspable in clutter. Grasping itself is a calibrated primitive, not a
  learning problem, so it is executed by rule the moment the scene allows
  it. That keeps every real-robot sample spent on the clearing behaviour.

Reward, computed here and not on the robot
------------------------------------------
  every clearing action (push or pick-and-place) : -0.05
  shaped terms from route 2 (see DEFAULT_W)        : added to that
  rule-triggered grasp succeeded                   : +10.0 ON TOP of the
                                                     step's cost and shaped
                                                     terms; episode ends
  rule-triggered grasp failed                      : -0.05 for that step;
                                                     no new attempt at the
                                                     same place until the
                                                     target has moved

One primitive per step (--one_action 1, the default)
---------------------------------------------------
A step that starts with the target graspable is the planner grasp, and
nothing else. Every other step is the policy's push (or pick-and-place,
unless --allow_pick 0). The old behaviour - push, then grasp, then retry,
all in one step - is --one_action 0.
  red block left the table                         : -5.0, episode ends
  step budget hit                                  : episode ends, but this
                                                     is a time limit, not
                                                     an outcome: the stored
                                                     transition is NOT
                                                     marked terminal

When the script stops and waits for you (--pause 1, the default)
---------------------------------------------------------------
At the start of an episode, with the arm parked, if
  * the red block is already graspable (nothing to learn), or
  * the red block left the table in the last episode, or
  * the camera sees nothing on the table, or
  * the red block has not been seen for --max_blind_episodes episodes.
It prints what it sees and waits for Enter. Rebuild the clutter then; the
arm does not move while it waits.

What the policy sees (--home_each_step 1, the default)
------------------------------------------------------
After every action the arm goes home and ONE frame is taken with the arm
out of the picture. That frame is the policy's next observation, the
"after" picture for the reward, and the "before" picture of the next step.
So every observation shows the scene and never the arm, and the reward
compares two pictures taken from the same arm pose.

Continuing a run that stopped
-----------------------------
  --resume 1 (with the same --save and --save_replay paths) loads the
  networks, the optimizers, the replay buffer and the step/episode
  counters, and appends to the same CSV files. On any error or Ctrl+C the
  checkpoint and the buffer are written before the script exits.

Success needs BOTH signals: the gripper reports a hold after the HOLD
period AND route 2 sees the red target gone from the table. Grasping a
green block therefore never counts.

Freezing hooks for the transfer experiment
------------------------------------------
  --freeze_phia 1    freeze the action-side factor, retrain phi_s. This is
                     the direction this project needs: objects change, the
                     arm does not. It is the opposite of the original FaStR
                     paper, which changes actuators and freezes phi_s.
  --freeze_visual 1  keep a pretrained visual encoder fixed.
  --init_visual PATH load pretrained RGBDEncoder weights.

Examples
--------
Mock test, no robot, no models:
  python3 train_two_pipeline_clutter.py --host 127.0.0.1 --port 9999 \
      --steps 40 --perception geometric --log_dir logs_mock

Real robot, clearing-only policy with the rule-triggered planner grasp:
  python3 train_two_pipeline_clutter.py --host locobot.local \
      --steps 400 --warmup 30 --perception geometric \
      --table_z 0.167 --ws_x_min 0.09 --ws_x_max 0.42 \
      --ws_y_min -0.38 --ws_y_max -0.02 --anchor "0.27,-0.20" \
      --max_radial 0.38 --log_dir logs_clutter --save ckpt_clutter.pt

On the robot, geometric perception (works before the segmentation node is
wired in):
  python3 train_two_pipeline_clutter.py --host locobot.local \
      --steps 400 --perception geometric \
      --log_dir logs_clutter --save ckpt_clutter.pt

On the robot with the segmentation backend, adapting to unseen objects with
the action-side factor frozen:
  python3 train_two_pipeline_clutter.py --host locobot.local \
      --steps 300 --perception segmentation \
      --load ckpt_clutter.pt --freeze_phia 1 \
      --log_dir logs_adapt_frozen --save ckpt_adapt_frozen.pt
"""

import argparse
import base64
import csv
import json
import os
import socket
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import route2_perception


OBS_DIM = 45      # full server observation (protocol; unchanged)
# 6-D action: a[0] chooses push (<0) or grasp (>=0); a[1] selects the
# obstacle; a[2:4] is the push direction; a[4] the push distance; a[5] the
# gripper roll offset for a grasp, scaled by the server to
# +-GRASP_ROLL_RANGE around the geometric default (fingers across the
# target's short side). Roll is a real degree of freedom of the gripper, so
# the policy owns it; the geometric default only gives exploration a sane
# starting point, and in clutter the best approach angle is often not the
# perpendicular one.
# 5-D clearing action: push/pick selector, obstacle selector, push direction
# (2), push distance. The target grasp is not an action; see the docstring.
ACT_DIM = 5

# What the POLICY actually sees. The server's 45-D vector is proprioception
# (5 joints + 3 EE + 1 gripper) followed by 36 perception entries (target
# centroid/extent/visibility, clearance, obstacle slots) computed by point-
# cloud clustering. Those perception entries must NOT reach the networks:
# training is pure pixel RL, and feeding clustered centroids into phi_s would
# hand it the scene pre-digested, leaving nothing object-specific for phi_s
# to learn and hollowing out the freeze-phi_a transfer claim. So the networks
# receive the raw RGB-D pixels plus only the first 9 proprioceptive entries.
# Proprioception is the robot's own internal reading, not perception.
PROPRIO_DIM = 9

# Reward. A sparse -0.05/+10 was tried first and produced exactly what the
# push-grasp literature warns about: the policy learned to shove the pile
# apart as hard as it could, because nothing distinguished a targeted nudge
# from a scattering blow, and the real costs of scattering (blocks off the
# table, target out of reach, a human resetting the scene) were invisible
# to it. The shaped terms below all come from route 2 geometry that is
# already measured every step; none of them touches the observation or the
# action. Weights are CLI flags.
GRASP_SUCCESS_BONUS = 20.0   # fires on the step that first makes the scene
                             # graspable; the scripted grasp afterwards is
                             # physical cleanup and does not change the reward
STEP_COST = 0.05
DEFAULT_W = {
    "clearance": 10.0,  # per metre of clearance gained (target visible)
    "expose": 2.0,      # target hidden -> visible: +; visible -> hidden: -
    "split": 0.0,       # disabled: only the red block is a scored subject;
                        # the number of other clusters is not a reward signal
    "disturb": 5.0,     # per metre of displacement of objects NOT acted on,
                        # and of pieces thrown further than the push went
    "lost": 0.0,        # disabled: only the red block leaving the workspace
                        # is a terminal failure (target_lost / outreach)
    "outreach": 5.0,    # red pushed beyond the arm's reach; TERMINAL
    "target_lost": 15.0,  # red left the table; TERMINAL
    "pose_edge": 20.0,  # per metre the RED block moved into the outer 3 cm
                        # of the reach ring (toward a reach limit): a red
                        # block that can only be grasped at the limit is a
                        # worse state. Potential-based, so pushing it back
                        # earns the same amount back.
    "pose_angle": 1.0,  # the red bar turned so the jaw must close along the
                        # line to the robot (tilted fingers at different
                        # heights, the awkward grasp): 0 = across, 1 = along
    "crowd": 50.0,      # per metre a pushed obstacle ended CLOSER to the red
                        # block (beyond 5 mm of noise): squeezing the red
                        # block is the opposite of the task. 2 cm closer =
                        # -1.0, a full 6 cm push into it = -3.0. Added on
                        # top of the lost clearance and "hit_target".
    "hit_target": 10.0, # a push aimed at ANOTHER object moved the red block
                        # by more than 1 cm (the obstacle was driven into
                        # it). Once per step, on top of "disturb" and the
                        # lost clearance. The server no longer refuses such
                        # pushes; this is how the policy learns not to.
    "near_edge": 2.0,   # an obstacle ended up inside the inner reach limit
                        # (the near table edge), once per object
    "refused": 5.0,     # the chosen push could not be executed at all
                        # (start point beyond the arm's reach, IK refused
                        # the approach, fingers would land on another
                        # object, object would be pushed off the table edge
                        # or into the inner limit): the arm did not move.
                        # 100x the step cost, as large as pushing the red
                        # block out of reach. Smaller than driving an
                        # obstacle into the red block (hit_target 10 +
                        # crowd), so "try nothing" is never the cheap way
                        # out. --w_refused sets it.
}
# "expose" and "split" are DIFFERENCES between the picture after and the
# picture before, with the same size in both directions. Burying the target
# and digging it out again therefore sums to zero and cannot be repeated
# for profit. (They used to pay only in the + direction: pushing a green
# block against the red one and away again earned about +0.9 per round.)
#
# Oct 2026 overhaul: the big "success" reward used to fire only when the
# scripted grasp lifted the red block, which mixed policy learning with
# gripper calibration noise. The policy's only job is to make the scene
# graspable; the scripted grasp is a calibrated primitive. The bonus now
# fires on the step that FIRST satisfies the graspable rule and the
# episode terminates. Outreach was 0.5 and non-terminal; it is now 5 and
# terminal, matching "red pushed out of reach is a run-ending failure".


class StopTraining(Exception):
    """The human asked to stop at a scene check."""


def ask_human(tag, text):
    """Every pause goes through here (tests replace it). Returns what was
    typed, lower case. The arm is parked while this waits."""
    print("\n>>> [%s] %s" % (tag, text))
    try:
        return input("    Enter = look again   g = go on anyway   q = save "
                     "and quit : ").strip().lower()
    except EOFError:
        print("\n[train] no keyboard to answer the pause (run from a "
              "terminal, or use --pause 0); stopping")
        return "q"


def center_crop_resize(array, size):
    """Centre-crop to a square and resize, nearest neighbour. Identical to
    the server's function of the same name, so an observation built here
    from a full frame equals the one the server would have sent."""
    h, w = array.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = array[y0:y0 + side, x0:x0 + side]
    iy = np.linspace(0, side - 1, size).round().astype(np.int64)
    ix = np.linspace(0, side - 1, size).round().astype(np.int64)
    return np.ascontiguousarray(crop[iy[:, None], ix[None, :]])


# ---------------- remote environment ----------------

def decode_array(packet):
    if packet.get("encoding") != "zlib+base64":
        raise ValueError("unsupported array encoding: %r"
                         % packet.get("encoding"))
    raw = zlib.decompress(base64.b64decode(packet["data"]))
    arr = np.frombuffer(raw, dtype=np.dtype(packet["dtype"]))
    expected = int(np.prod(packet["shape"]))
    if arr.size != expected:
        raise ValueError("RGB-D packet has %d values, expected %d"
                         % (arr.size, expected))
    return arr.reshape(packet["shape"]).copy()


class RemoteEnv(object):
    """Thin client for the dumb executor on the robot."""

    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=300)
        self.sock.settimeout(None)
        self.rfile = self.sock.makefile("r")
        assert self._send({"cmd": "ping"}).get("ok"), "server ping failed"

    def _send(self, msg):
        self.sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))
        line = self.rfile.readline()
        if not line:
            raise ConnectionError("environment server closed connection")
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError("environment server error: %s" % resp["error"])
        return resp

    @staticmethod
    def _observation(resp):
        if "rgbd" not in resp:
            raise RuntimeError(
                "server response has no rgbd field; run the RGB-D server")
        rgbd = resp["rgbd"]
        rgb = decode_array(rgbd["rgb"])
        depth = decode_array(rgbd["depth"])
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("expected HxWx3 RGB, got %r" % (rgb.shape,))
        if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
            raise ValueError("RGB/depth size mismatch: %r vs %r"
                             % (rgb.shape, depth.shape))
        return {
            "state": np.asarray(resp["obs"], dtype=np.float32),
            "rgb": np.asarray(rgb, dtype=np.uint8),
            "depth": np.asarray(depth, dtype=np.uint16),
            "rgb_stamp": float(rgbd.get("rgb_stamp", 0.0)),
            "depth_stamp": float(rgbd.get("depth_stamp", 0.0)),
        }

    def reset(self):
        resp = self._send({"cmd": "reset"})
        for reason in resp.get("motion_errors", []):
            print("[motion] during reset: %s" % reason)
        return self._observation(resp)

    def step(self, action, perception=None, grasp_role="target"):
        """Execute one primitive. `perception` carries route 2's centroids;
        the server acts on them. grasp_role: "obstacle" makes a[0] >= 0 a
        pick-and-place of obstacle a[1]; "target" makes it the planner
        grasp of the target. The server's reward field is always 0 and is
        ignored: the reward is computed here."""
        msg = {"cmd": "step",
               "action": np.asarray(action, dtype=float).tolist(),
               "grasp_role": grasp_role}
        if perception is not None:
            msg["perception"] = perception
        resp = self._send(msg)
        return (self._observation(resp), bool(resp["done"]),
                resp.get("info", {}))

    def frame(self):
        """One full-resolution frame for route 2. Moves nothing."""
        return route2_perception.decode_frame(self._send({"cmd": "frame"}))

    def home(self):
        """Move the arm to the episode home. Used once per episode, right
        before the success-confirmation frame: the lifted arm sits exactly
        on the camera's line of sight to the target's old spot, so the
        confirmation must be taken with the arm out of the way. The reset
        that follows was going to make this same motion anyway."""
        return self._send({"cmd": "home"})

    def close(self):
        try:
            self.rfile.close()
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


# ---------------- networks ----------------
# Layout kept identical to train_pixel_factored_sac.py so pretrained
# RGBDEncoder weights are interchangeable between the two scripts.

def mlp_body(in_dim, hidden, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim))


class ConvBranch(nn.Module):
    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)))
        self.proj = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 4 * 4, out_dim),
            nn.LayerNorm(out_dim), nn.ReLU())

    def forward(self, image):
        return self.proj(self.conv(image))


class RGBDEncoder(nn.Module):
    """Separate RGB and depth encoders followed by learned fusion."""

    def __init__(self, visual_dim=128, depth_max_m=1.5):
        super().__init__()
        branch_dim = visual_dim // 2
        self.depth_max_m = float(depth_max_m)
        self.rgb_encoder = ConvBranch(3, branch_dim)
        self.depth_encoder = ConvBranch(1, branch_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * branch_dim, visual_dim),
            nn.LayerNorm(visual_dim), nn.ReLU())

    def forward(self, rgb, depth):
        rgb = rgb / 255.0
        depth = torch.clamp(depth / 1000.0, 0.0, self.depth_max_m)
        depth = depth / self.depth_max_m
        return self.fusion(torch.cat(
            [self.rgb_encoder(rgb), self.depth_encoder(depth)], dim=-1))


class StateVisualEncoder(nn.Module):
    def __init__(self, state_dim, out_dim, visual_dim=128, hidden=256,
                 depth_max_m=1.5):
        super().__init__()
        self.visual = RGBDEncoder(visual_dim, depth_max_m)
        self.fuse = mlp_body(state_dim + visual_dim, hidden, out_dim)

    def forward(self, state, rgb, depth):
        return self.fuse(torch.cat([state, self.visual(rgb, depth)], dim=-1))


class FactoredQ(nn.Module):
    """Q(s, image, a) = head(phi_s(s, image) * phi_a(a))."""

    def __init__(self, state_dim, act_dim, d=64, hidden=256,
                 visual_dim=128, depth_max_m=1.5):
        super().__init__()
        self.phi_s = StateVisualEncoder(
            state_dim, d, visual_dim, hidden, depth_max_m)
        self.phi_a = mlp_body(act_dim, hidden, d)
        self.head = nn.Sequential(
            nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, state, rgb, depth, action):
        return self.head(self.phi_s(state, rgb, depth) * self.phi_a(action))


class Actor(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, state_dim, act_dim, hidden=256, visual_dim=128,
                 depth_max_m=1.5):
        super().__init__()
        self.encoder = StateVisualEncoder(
            state_dim, hidden, visual_dim, hidden, depth_max_m)
        self.body = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, state, rgb, depth):
        h = self.body(self.encoder(state, rgb, depth))
        return self.mu(h), torch.clamp(
            self.log_std(h), self.LOG_STD_MIN, self.LOG_STD_MAX)

    def sample(self, state, rgb, depth):
        mu, log_std = self(state, rgb, depth)
        dist = torch.distributions.Normal(mu, log_std.exp())
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)
        logp = dist.log_prob(pre_tanh) - torch.log(
            1.0 - action.pow(2) + 1e-6)
        return action, logp.sum(-1, keepdim=True)

    def act(self, state, rgb, depth, deterministic=False):
        with torch.no_grad():
            mu, log_std = self(state, rgb, depth)
            if deterministic:
                return torch.tanh(mu)
            return torch.tanh(mu + log_std.exp() * torch.randn_like(mu))


# ---------------- replay and augmentation ----------------

class Replay(object):
    def __init__(self, capacity, state_dim, act_dim, image_shape):
        h, w = image_shape
        self.capacity = capacity
        self.state = np.zeros((capacity, state_dim), np.float32)
        self.rgb = np.zeros((capacity, h, w, 3), np.uint8)
        self.depth = np.zeros((capacity, h, w), np.uint16)
        self.action = np.zeros((capacity, act_dim), np.float32)
        self.reward = np.zeros((capacity, 1), np.float32)
        self.next_state = np.zeros((capacity, state_dim), np.float32)
        self.next_rgb = np.zeros((capacity, h, w, 3), np.uint8)
        self.next_depth = np.zeros((capacity, h, w), np.uint16)
        self.done = np.zeros((capacity, 1), np.float32)
        self.n = 0
        self.i = 0

    def add(self, obs, action, reward, next_obs, done):
        """`done` must be 1 only when the task really ended (success, or
        the target left the table). An episode cut off by the step budget
        is stored with 0: the scene after it is an ordinary state and the
        value of going on from there must still be counted. Storing 1
        there teaches the critic that running out of steps is worth
        nothing, which drags every Q value down."""
        i = self.i
        self.state[i] = obs["state"][:self.state.shape[1]]
        self.rgb[i] = obs["rgb"]
        self.depth[i] = obs["depth"]
        self.action[i] = action
        self.reward[i] = reward
        self.next_state[i] = next_obs["state"][:self.state.shape[1]]
        self.next_rgb[i] = next_obs["rgb"]
        self.next_depth[i] = next_obs["depth"]
        self.done[i] = done
        self.i = (i + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def save(self, path):
        """Write the filled part of the buffer to an .npz. This is the raw
        (s, a, r, s') data the spectral analysis needs (effective rank of
        the transitions); the checkpoint alone does not keep it."""
        n = self.n
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        np.savez_compressed(
            path, state=self.state[:n], rgb=self.rgb[:n],
            depth=self.depth[:n], action=self.action[:n],
            reward=self.reward[:n], next_state=self.next_state[:n],
            next_rgb=self.next_rgb[:n], next_depth=self.next_depth[:n],
            done=self.done[:n])
        print("[train] saved %d transitions to %s" % (n, path))

    def load(self, path):
        """Refill the buffer from a file written by save()."""
        data = np.load(path)
        n = min(int(len(data["reward"])), self.capacity)
        if data["rgb"].shape[1:] != self.rgb.shape[1:]:
            raise ValueError("replay file image size %r does not match %r"
                             % (data["rgb"].shape[1:], self.rgb.shape[1:]))
        for name in ("state", "rgb", "depth", "action", "reward",
                     "next_state", "next_rgb", "next_depth", "done"):
            getattr(self, name)[:n] = data[name][-n:]
        self.n = n
        self.i = n % self.capacity
        print("[train] loaded %d transitions from %s" % (n, path))

    @staticmethod
    def _tensor(array, idx, device):
        # float32 in numpy first: torch before 2.3 (the Jetson's) cannot
        # take uint16 arrays, which is what depth is stored as
        return torch.as_tensor(np.asarray(array[idx], dtype=np.float32),
                               device=device)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.n, size=batch_size)
        return (self._tensor(self.state, idx, device),
                self._tensor(self.rgb, idx, device).permute(0, 3, 1, 2),
                self._tensor(self.depth, idx, device).unsqueeze(1),
                self._tensor(self.action, idx, device),
                self._tensor(self.reward, idx, device),
                self._tensor(self.next_state, idx, device),
                self._tensor(self.next_rgb, idx, device).permute(0, 3, 1, 2),
                self._tensor(self.next_depth, idx, device).unsqueeze(1),
                self._tensor(self.done, idx, device))


def random_shift_pair(rgb, depth, pad):
    """DrQ-style translation; RGB and depth receive the identical shift."""
    if pad <= 0:
        return rgb, depth
    x = torch.cat([rgb, depth], dim=1)
    n, _, h, w = x.shape
    if h != w:
        raise ValueError("random_shift_pair expects square images")
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    size = h + 2 * pad
    eps = 1.0 / size
    coordinates = torch.linspace(
        -1.0 + eps, 1.0 - eps, size, device=x.device, dtype=x.dtype)[:h]
    grid_x = coordinates.unsqueeze(0).repeat(h, 1)
    grid_y = coordinates.unsqueeze(1).repeat(1, h)
    base_grid = torch.stack([grid_x, grid_y], dim=-1)
    base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
    shift = torch.randint(
        0, 2 * pad + 1, size=(n, 1, 1, 2), device=x.device,
        dtype=torch.int64).to(x.dtype)
    shift *= 2.0 / size
    out = F.grid_sample(x, base_grid + shift, mode="bilinear",
                        padding_mode="zeros", align_corners=False)
    return out[:, :3], out[:, 3:]


def observation_tensors(obs, device):
    state = torch.as_tensor(
        obs["state"][:PROPRIO_DIM], device=device,
        dtype=torch.float32).unsqueeze(0)
    rgb = torch.as_tensor(np.asarray(obs["rgb"], dtype=np.float32),
                          device=device)
    rgb = rgb.permute(2, 0, 1).unsqueeze(0)
    depth = torch.as_tensor(np.asarray(obs["depth"], dtype=np.float32),
                            device=device)
    return state, rgb, depth.unsqueeze(0).unsqueeze(0)


# ---------------- SAC agent ----------------

class SAC(object):
    def __init__(self, device, lr=3e-4, gamma=0.99, tau=0.005,
                 d=64, visual_dim=128, depth_max_m=1.5,
                 freeze_phia=False, freeze_visual=False, aug_pad=4):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.freeze_phia = freeze_phia
        self.freeze_visual = freeze_visual
        self.aug_pad = aug_pad
        net_args = (PROPRIO_DIM, ACT_DIM, d, 256, visual_dim, depth_max_m)
        self.actor = Actor(
            PROPRIO_DIM, ACT_DIM, 256, visual_dim, depth_max_m).to(device)
        self.q1 = FactoredQ(*net_args).to(device)
        self.q2 = FactoredQ(*net_args).to(device)
        self.q1_target = FactoredQ(*net_args).to(device)
        self.q2_target = FactoredQ(*net_args).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q1_target.requires_grad_(False)
        self.q2_target.requires_grad_(False)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.target_entropy = -float(ACT_DIM)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)
        self._build_optimizers()

    # -- freezing --

    def _critic_trainable(self):
        params = []
        for critic in (self.q1, self.q2):
            for name, param in critic.named_parameters():
                frozen = (
                    (self.freeze_phia and name.startswith("phi_a"))
                    or (self.freeze_visual
                        and name.startswith("phi_s.visual")))
                param.requires_grad_(not frozen)
                if not frozen:
                    params.append(param)
        return params

    def _actor_trainable(self):
        params = []
        for name, param in self.actor.named_parameters():
            frozen = self.freeze_visual and name.startswith("encoder.visual")
            param.requires_grad_(not frozen)
            if not frozen:
                params.append(param)
        return params

    def _build_optimizers(self):
        self.opt_critic = torch.optim.Adam(
            self._critic_trainable(), lr=self.lr)
        self.opt_actor = torch.optim.Adam(
            self._actor_trainable(), lr=self.lr)

    def set_freeze(self, freeze_phia=None, freeze_visual=None):
        if freeze_phia is not None:
            self.freeze_phia = bool(freeze_phia)
        if freeze_visual is not None:
            self.freeze_visual = bool(freeze_visual)
        self._build_optimizers()

    def frozen_counts(self):
        critic = sum(1 for c in (self.q1, self.q2)
                     for p in c.parameters() if not p.requires_grad)
        actor = sum(1 for p in self.actor.parameters()
                    if not p.requires_grad)
        return critic, actor

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update(self, batch):
        (state, rgb, depth, action, reward, next_state, next_rgb,
         next_depth, done) = batch
        rgb, depth = random_shift_pair(rgb, depth, self.aug_pad)
        next_rgb, next_depth = random_shift_pair(
            next_rgb, next_depth, self.aug_pad)

        with torch.no_grad():
            next_action, next_logp = self.actor.sample(
                next_state, next_rgb, next_depth)
            target_q = torch.min(
                self.q1_target(next_state, next_rgb, next_depth, next_action),
                self.q2_target(next_state, next_rgb, next_depth, next_action))
            target = reward + self.gamma * (1.0 - done) * (
                target_q - self.alpha * next_logp)

        q1 = self.q1(state, rgb, depth, action)
        q2 = self.q2(state, rgb, depth, action)
        loss_q = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.opt_critic.zero_grad(set_to_none=True)
        loss_q.backward()
        self.opt_critic.step()

        critic_grad_flags = [
            [p.requires_grad for p in critic.parameters()]
            for critic in (self.q1, self.q2)]
        for critic in (self.q1, self.q2):
            critic.requires_grad_(False)
        policy_action, logp = self.actor.sample(state, rgb, depth)
        q_policy = torch.min(
            self.q1(state, rgb, depth, policy_action),
            self.q2(state, rgb, depth, policy_action))
        loss_actor = (self.alpha.detach() * logp - q_policy).mean()
        self.opt_actor.zero_grad(set_to_none=True)
        loss_actor.backward()
        self.opt_actor.step()
        for critic, flags in zip((self.q1, self.q2), critic_grad_flags):
            for param, requires_grad in zip(critic.parameters(), flags):
                param.requires_grad_(requires_grad)

        loss_alpha = -(
            self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        loss_alpha.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            for critic, target_critic in (
                    (self.q1, self.q1_target), (self.q2, self.q2_target)):
                for param, target_param in zip(
                        critic.parameters(), target_critic.parameters()):
                    target_param.mul_(1.0 - self.tau).add_(
                        param, alpha=self.tau)

        return {"loss_q": float(loss_q.item()),
                "loss_actor": float(loss_actor.item()),
                "alpha": float(self.alpha.item())}

    # -- pretrained visual encoder --

    def load_visual(self, path):
        """Load pretrained RGBDEncoder weights into the actor and both
        critics. Accepts a raw RGBDEncoder state dict, {"visual": sd}, or a
        full agent checkpoint (the actor's encoder.visual.* is extracted)."""
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "visual" in checkpoint:
            visual_sd = checkpoint["visual"]
        elif isinstance(checkpoint, dict) and "actor" in checkpoint:
            prefix = "encoder.visual."
            visual_sd = {k[len(prefix):]: v
                         for k, v in checkpoint["actor"].items()
                         if k.startswith(prefix)}
        else:
            visual_sd = checkpoint
        targets = [self.actor.encoder.visual,
                   self.q1.phi_s.visual, self.q2.phi_s.visual,
                   self.q1_target.phi_s.visual, self.q2_target.phi_s.visual]
        for module in targets:
            module.load_state_dict(visual_sd)
        print("[train] loaded pretrained visual encoder from %s "
              "(%d tensors into %d modules)"
              % (path, len(visual_sd), len(targets)))

    def save(self, path, counters=None):
        """counters: {"global_step": n, "episode": m}, kept for --resume."""
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        temp = path + ".tmp"
        torch.save({
            "format": "two_pipeline_clutter_v5",
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "opt_alpha": self.opt_alpha.state_dict(),
            "counters": dict(counters or {}),
        }, temp)
        os.replace(temp, path)   # never leave a half-written checkpoint

    def load_optimizers(self, checkpoint):
        """For --resume. Only valid when the same parameters are trainable
        as when the checkpoint was written; otherwise the optimizers start
        fresh, which is what a new training phase wants anyway."""
        try:
            self.opt_actor.load_state_dict(checkpoint["opt_actor"])
            self.opt_critic.load_state_dict(checkpoint["opt_critic"])
            self.opt_alpha.load_state_dict(checkpoint["opt_alpha"])
            return True
        except Exception as error:
            self._build_optimizers()
            print("[train] optimizer state not restored (%s); starting the "
                  "optimizers fresh" % error)
            return False

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if checkpoint.get("format") != "two_pipeline_clutter_v5":
            raise ValueError(
                "checkpoint format is %r, expected two_pipeline_clutter_v5. "
                "v5 is the clearing-only MDP (5-D action, rule-triggered "
                "planner grasp); earlier checkpoints have a different "
                "action width and cannot be loaded"
                % checkpoint.get("format"))
        self.actor.load_state_dict(checkpoint["actor"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.q1_target.load_state_dict(checkpoint["q1_target"])
        self.q2_target.load_state_dict(checkpoint["q2_target"])
        with torch.no_grad():
            self.log_alpha.copy_(checkpoint["log_alpha"].to(self.device))
        return checkpoint


# ---------------- route 2 wiring ----------------

STEP_COLUMNS = [
    "wall_time", "global_step", "episode", "ep_step", "primitive",
    "obstacle_index", "pushed_target", "push_dir_x", "push_dir_y",
    "push_distance",
    "obst_before_x", "obst_before_y", "obst_after_x", "obst_after_y",
    "obst_disp",
    "target_found_before", "target_source", "arm_blobs",
    "n_objects_before", "target_x", "target_y",
    "graspable_after", "clearance_after", "rule_reason", "auto_grasp",
    "held", "gripper_raw", "target_on_table_after",
    "success", "reward", "r_shaping", "r_clearance", "r_expose", "r_split",
    "r_disturb", "r_lost", "r_outreach", "r_refused", "r_near_edge",
    "r_pose", "r_hit_target", "r_crowd",
    "done",
    "motion_errors",
    # added 2026-10-09 (at the end, so older column positions still hold)
    "r_target_lost", "target_lost", "terminal", "grasp_attempts"]
EPISODE_COLUMNS = [
    "wall_time", "global_step", "episode", "return", "length", "success",
    "grasps", "pushes", "end", "started_graspable"]


def build_route2(args):
    config = route2_perception.PerceptionConfig(
        ws_x=(args.ws_x_min, args.ws_x_max),
        ws_y=(args.ws_y_min, args.ws_y_max),
        table_z=args.table_z, z_max=args.table_z + 0.20)
    backend = route2_perception.make_backend(
        args.perception, config, model_dir=args.model_dir)
    anchor = tuple(float(v) for v in args.anchor.split(","))
    route2 = route2_perception.Route2Perception(
        backend, identity=args.identity, target_label=args.target_label,
        anchor_xy=anchor, track_max=args.track_max, hue_tol=args.hue_tol,
        target_max_extent=args.target_max_extent)
    if args.identity == "color":
        print("[route2] backend=%s identity=color target_label=%r"
              % (backend.name, args.target_label))
    else:
        print("[route2] backend=%s identity=track track_max=%.3f"
              % (backend.name, args.track_max))
    return route2


class EpisodeRunner(object):
    """Runs one clearing action end to end and computes its reward.

    Per step:
      1. the policy chooses a clearing action from pixels (route 1)
      2. route 2 locates the target and the obstacles
      3. the server executes the push or the pick-and-place
      4. route 2 re-perceives; grasp_feasibility() applies the rule
      5. if graspable (and the trigger is "auto"), the planner grasp runs;
         the arm homes; route 2 checks the table; success is decided
      6. the reward is computed here
    """

    def __init__(self, env, route2, verbose=True, dump_dir=None,
                 dump_every=1, grasp_trigger="auto", clearance_min=0.06,
                 max_radial=0.38, credit_clear_grasp=True, weights=None,
                 min_radial=0.0, max_ep_steps=0, push_frame="world",
                 reset_prompt=False, scene_text="", show_anchor=False,
                 pause=False, home_each_step=False, grasp_retries=0,
                 max_blind_episodes=3, one_action=False, allow_pick=True):
        # one_action: every step is exactly ONE physical primitive. A step
        # that starts with the target graspable is the planner grasp (the
        # environment's move; the policy's action is not executed), any
        # other step is the policy's push or pick. Nothing is chained.
        self.one_action = bool(one_action)
        # allow_pick=False: a[0] is ignored and every policy action is a
        # push. The pick-and-place of an obstacle is a grasp too, and with
        # an uncalibrated gripper it mostly closes on nothing.
        self.allow_pick = bool(allow_pick)
        # where the last planner grasp missed; no new attempt there until
        # the target has moved
        self.failed_grasp_xy = None
        # The three defaults above are the OLD behaviour, so code that
        # builds a runner directly keeps working. main() passes the command
        # line values, whose defaults are the new behaviour (1, 1, 1).
        self.pause = bool(pause)
        self.home_each_step = bool(home_each_step)
        self.grasp_retries = max(0, int(grasp_retries))
        self.max_blind_episodes = int(max_blind_episodes)
        self.view = None            # clean route-2 view the last step ended on
        self.home_state = None      # proprioception with the arm parked
        self.image_size = None
        self.last_end = None        # how the previous episode ended
        self.blind_episodes = 0     # episodes in a row that never saw the target
        self.saw_target = False
        self.started_graspable = False
        self.env = env
        self.route2 = route2
        self.verbose = verbose
        self.dump_dir = dump_dir
        self.dump_every = max(1, int(dump_every))
        self._dump_count = 0
        self.grasp_trigger = grasp_trigger
        self.clearance_min = float(clearance_min)
        self.max_radial = float(max_radial)
        self.min_radial = float(min_radial)
        # episode length cut by the client (0 = only the server's HORIZON)
        self.max_ep_steps = int(max_ep_steps)
        self.ep_t = 0
        # "relative": a[2:4] is a push direction in the frame of the chosen
        # object, x axis = straight away from the red block. (1, 0) is then
        # "push it away from the red one" for EVERY obstacle wherever it
        # lies, so what is learned once holds everywhere. "world": a[2:4]
        # is the robot-frame direction (the old behaviour).
        self.push_frame = push_frame
        # 0 never, 1 only when the scene is used up, 2 every episode
        self.reset_prompt = int(reset_prompt)
        # every push of the last episode was refused: the scene cannot be
        # worked on as it lies, ask for a new one
        self.stuck = False
        self.ep_refused = 0
        self.show_anchor = bool(show_anchor)
        self.scene_text = scene_text
        self.credit_clear_grasp = bool(credit_clear_grasp)
        self.w = dict(DEFAULT_W)
        if weights:
            self.w.update(weights)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)

    def _dump_frame(self, frame, tag):
        """Save the full-resolution RGB-D frame to disk for inspection:
        <tag>_rgb.png (viewable), <tag>_depth.png (16-bit millimetres) and
        <tag>_depth_vis.png (normalised, viewable)."""
        if not self.dump_dir:
            return
        self._dump_count += 1
        if (self._dump_count - 1) % self.dump_every:
            return
        try:
            from PIL import Image
            base = os.path.join(self.dump_dir, tag)
            Image.fromarray(frame["rgb"]).save(base + "_rgb.png")
            Image.fromarray(frame["depth"]).save(base + "_depth.png")
            depth = frame["depth"].astype(np.float32)
            top = np.percentile(depth[depth > 0], 99) if (depth > 0).any() \
                else 1.0
            vis = np.clip(depth / max(top, 1.0), 0, 1) * 255
            Image.fromarray(vis.astype(np.uint8)).save(
                base + "_depth_vis.png")
        except Exception as error:
            print("[dump] WARNING: could not save %s (%s)" % (tag, error))

    def _perceive(self, ee_xyz, tag):
        frame = self.env.frame()
        self._dump_frame(frame, tag)
        return self.route2.perceive(frame, ee_xyz=ee_xyz)

    def _look(self, obs, tag, force_home=False):
        """The route-2 view after an action, and the observation that goes
        with it. Returns (view, obs, clean).

        clean=True: the arm was sent home first, so the frame shows the
        scene without the arm. With --home_each_step 1 that is always the
        case, and the policy's observation is rebuilt from this same frame
        (pixels as the server would have cropped them, proprioception of
        the parked arm, the gripper flag as it is now)."""
        if self.home_each_step or force_home:
            self.env.home()
            frame = self.env.frame()
            self._dump_frame(frame, tag)
            view = self.route2.perceive(
                frame, ee_xyz=(None if self.home_state is None
                               else self.home_state[5:8]))
            if view.get("target_source") == "cache":
                # route 2 fills in the remembered target position when the
                # arm blocks the view. With the arm parked nothing blocks
                # it, so "not detected" means not visible: report that.
                view = dict(view, target=None, target_found=False,
                            target_source="none")
            if self.home_each_step and obs is not None \
                    and self.home_state is not None:
                state = np.array(obs["state"], dtype=np.float32)
                state[:8] = self.home_state[:8]
                obs = {"state": state,
                       "rgb": center_crop_resize(frame["rgb"],
                                                 self.image_size),
                       # the picture as the camera saw it: the server's
                       # own observations are not masked either
                       "depth": center_crop_resize(
                           frame.get("depth_raw", frame["depth"]),
                           self.image_size),
                       "rgb_stamp": obs.get("rgb_stamp", 0.0),
                       "depth_stamp": obs.get("depth_stamp", 0.0)}
            return view, obs, True
        frame = self.env.frame()
        self._dump_frame(frame, tag)
        ee = None if obs is None else obs["state"][5:8]
        return self.route2.perceive(frame, ee_xyz=ee), obs, False

    @staticmethod
    def _something_near(view, xy, gap=0.10):
        """Is any detected object (single block or pile) within `gap` of
        the point xy, measured to its footprint?"""
        for d in view.get("detections") or []:
            distance = float(np.linalg.norm(
                np.asarray(d["centroid"][:2], dtype=float) - xy))
            if distance - route2_perception.xy_radius(d["extent"]) < gap:
                return True
        return False

    def _target_gone(self, before, after, obs, clean, tag):
        """The target was seen before the action and is not seen after it.
        Two very different things look like that:
          * it merged into a pile or sits behind another block: still on
            the table, the policy can dig it out again;
          * it left the table (pushed over the edge, or far out of the
            camera's crop): nothing the policy does can bring it back.
        It counts as GONE only when no object at all is left near where it
        was, and two frames taken with the arm parked both say so.
        Returns (gone, view, obs, clean)."""
        last = np.asarray(before["target"]["centroid"][:2], dtype=float)
        if self._something_near(after, last):
            return False, after, obs, clean
        if not clean:
            after, obs, clean = self._look(obs, tag, force_home=True)
            if after.get("target") is not None \
                    or self._something_near(after, last):
                return False, after, obs, clean
        second = self._perceive(
            None if self.home_state is None else self.home_state[5:8],
            tag + "_again")
        if second.get("target_source") == "cache":
            second = dict(second, target=None)
        if second.get("target") is not None \
                or self._something_near(second, last):
            return False, second, obs, clean
        return True, after, obs, clean

    @staticmethod
    def _chosen(action, pushables):
        """The pushable the server will pick for selector a[1] (the same
        rule as the server: index = floor((a1 + 1) / 2 * n))."""
        n = len(pushables)
        index = min(int(0.5 * (float(action[1]) + 1.0) * n), n - 1)
        return max(index, 0)

    def _world_direction(self, action, perception):
        """a[2:4] given in the chosen object's frame -> robot frame.
        x axis: from the red block to the object (straight away from red);
        for the red block itself: away from its nearest obstacle."""
        pushables = perception["pushables"]
        chosen = pushables[self._chosen(action, pushables)]
        target = perception.get("target")
        here = np.asarray(chosen["centroid"][:2], dtype=float)
        if chosen.get("is_target"):
            others = [np.asarray(o["centroid"][:2], dtype=float)
                      for o in perception.get("obstacles") or []]
            if not others:
                return np.asarray(action[2:4], dtype=float)
            nearest = min(others, key=lambda o: float(np.linalg.norm(
                o - here)))
            away = here - nearest
        elif target is not None:
            away = here - np.asarray(target["centroid"][:2], dtype=float)
        else:
            return np.asarray(action[2:4], dtype=float)
        norm = float(np.linalg.norm(away))
        if norm < 1e-6:
            return np.asarray(action[2:4], dtype=float)
        u = away / norm
        v = np.array([-u[1], u[0]])
        return float(action[2]) * u + float(action[3]) * v

    def scripted_action(self, noise=0.15, rng=np.random):
        """The obvious move, for demonstration episodes. Returned in the
        policy's own action format (relative frame: x = away from red).

        Take the obstacle nearest to the red block. If there is a gap of
        3.5 cm or more between them, push it straight away from red. If
        they touch, "straight away" is impossible: the closed jaw has to
        come down on the far side of the obstacle, between it and red,
        and there is no room there (the server refuses). Then push it
        SIDEWAYS, out of the row, to the side that keeps it nearer the
        middle of the reach ring. A red block pressed between two others
        is freed by two such sideways pushes."""
        view = self.view or {}
        pushables = self._pushables(view) if view else []
        action = np.array([-1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        target = view.get("target")
        if target is None or not view.get("obstacles"):
            return np.clip(action + rng.normal(0, 0.5, ACT_DIM), -1, 1
                           ).astype(np.float32)
        n = len(pushables)
        index = 1 if PUSH_RED else 0  # nearest obstacle
        action[1] = 2.0 * (index + 0.5) / n - 1.0
        obstacle = pushables[index]
        red = np.asarray(target["centroid"][:2], dtype=float)
        here = np.asarray(obstacle["centroid"][:2], dtype=float)
        pts = np.asarray(obstacle.get("points_xy") or [here], dtype=float)
        red_half = 0.5 * float(min(target["extent"][0],
                                   target["extent"][1]))
        gap = float(np.min(np.linalg.norm(pts - red, axis=1))) - red_half
        if gap < 0.035:
            u = here - red
            u = u / max(float(np.linalg.norm(u)), 1e-6)
            v = np.array([-u[1], u[0]])
            half = 0.5 * float(np.linalg.norm(obstacle["extent"][:2]))
            best = max((1.0, -1.0), key=lambda sgn: self._room(
                here, sgn * v, half))
            action[2], action[3] = 0.0, best
        relative = action.copy()
        if self.push_frame != "relative":
            perception = {"target": target,
                          "obstacles": view.get("obstacles"),
                          "pushables": pushables}
            action[2:4] = self._world_direction(relative, perception)
        jitter = rng.normal(0, noise, ACT_DIM)
        jitter[1] = 0.0               # keep the chosen object
        if gap < 0.035:
            # sideways push: noise may tilt it AWAY from red, never toward
            # it (toward red is what the server refuses)
            jitter[2] = abs(jitter[2])
        return np.clip(action + jitter, -1, 1).astype(np.float32)

    def _room(self, xy, direction, half):
        """How far an object at xy can travel along direction before it
        leaves the work area (minus half its size and 2 cm) or the reach
        ring. The same limits the server enforces on a push."""
        room = 0.20
        cfg = getattr(getattr(self.route2, "backend", None), "config", None)
        if cfg is not None:
            keep = half + 0.02
            for axis, (lo, hi) in ((0, cfg.box_x), (1, cfg.box_y)):
                d = float(direction[axis])
                if d < -1e-6:
                    room = min(room, max(0.0, (xy[axis] - lo - keep) / -d))
                elif d > 1e-6:
                    room = min(room, max(0.0, (hi - keep - xy[axis]) / d))
        for t in np.arange(0.0, room + 1e-9, 0.005):
            r = float(np.linalg.norm(xy + direction * t))
            if r < self.min_radial + 0.02 or r > self.max_radial - 0.02:
                return float(t)
        return float(room)

    @staticmethod
    def _pushables(view):
        """Objects the push selector may choose. Default (Oct 29): the
        obstacles ONLY. The red block stays where it is and the task is to
        clear the obstacles away from it so it can be grasped in place.
        With --push_red 1 the red block comes first in the list (the old
        behaviour) and may be pushed out of a pile."""
        pushables = []
        if PUSH_RED and view.get("target") is not None:
            entry = dict(view["target"])
            entry["is_target"] = True
            pushables.append(entry)
        for obstacle in view.get("obstacles") or []:
            entry = dict(obstacle)
            entry["is_target"] = False
            pushables.append(entry)
        return pushables

    def _obstacle_displacement(self, before, info, after):
        """Where the acted-on object was before and after the action."""
        empty = {"before": (None, None), "after": (None, None), "norm": None}
        index = info.get("obstacle_index")
        pool = (self._pushables(before) if info.get("primitive") == "push"
                else (before.get("obstacles") or []))
        if index is None or index >= len(pool):
            return empty
        start = np.asarray(pool[index]["centroid"][:2], dtype=float)
        result = {"before": (round(float(start[0]), 4),
                             round(float(start[1]), 4)),
                  "after": (None, None), "norm": None}
        if not after:
            return result
        candidates = [np.asarray(o["centroid"][:2], dtype=float)
                      for o in self._pushables(after)]
        if not candidates:
            return result
        distances = [float(np.linalg.norm(c - start)) for c in candidates]
        # the pick-and-place moves an object to the drop point, far away;
        # a push moves it at most ~0.15 m. Accept the nearest match within
        # 0.40 m so both cases associate.
        k = int(np.argmin(distances))
        if distances[k] > 0.40:
            return result
        end = candidates[k]
        result["after"] = (round(float(end[0]), 4), round(float(end[1]), 4))
        result["norm"] = round(distances[k], 4)
        return result

    def _shaping(self, before, after, info, primitive):
        """Progress and collateral terms from route 2, all in reward units.
        Returns (total, breakdown dict). The "target_lost" entry is filled
        in by step(), not here."""
        w = self.w
        terms = dict((name, 0.0) for name in DEFAULT_W)
        if not before or not after:
            return 0.0, terms
        tb, ta = before.get("target"), after.get("target")

        # ---- collateral: what moved that the action was not aimed at ----
        # Matched by class (target with target, obstacles with obstacles),
        # each obstacle with its nearest counterpart, the acted-on one
        # removed from both sides first.
        acted = info.get("obstacle_index")
        acted_is_target = bool(info.get("pushed_target", False))
        obs_b = list(before.get("obstacles") or [])
        obs_a = list(after.get("obstacles") or [])
        acted_obs_index = None
        if acted is not None and not acted_is_target:
            acted_obs_index = acted
            if primitive == "push" and tb is not None:
                acted_obs_index = acted - (1 if PUSH_RED else 0)
        disturbance = 0.0
        lost_objects = []
        if tb is not None and ta is not None and not acted_is_target:
            red_moved = float(np.linalg.norm(
                np.asarray(ta["centroid"][:2]) - np.asarray(tb["centroid"][:2])))
            disturbance += red_moved
            if red_moved > 0.01:
                terms["hit_target"] = -w["hit_target"]
        remaining_a = [np.asarray(o["centroid"][:2], dtype=float) for o in obs_a]
        acted_object, allowed, acted_after = None, 0.0, None
        if acted_is_target and tb is not None:
            acted_object = tb
        elif acted_obs_index is not None and 0 <= acted_obs_index < len(obs_b):
            acted_object = obs_b[acted_obs_index]
        if acted_object is not None:
            # how far the action itself can have carried the object (or a
            # piece of it): its own radius + the commanded push + 3 cm
            carry = float(info.get("push_distance") or 0.0) \
                if primitive == "push" else 0.0
            allowed = (route2_perception.xy_radius(acted_object["extent"])
                       + carry + 0.03)
        if acted_object is not None and not acted_is_target:
            # Where the acted-on object should be now: at the drop point
            # after a pick-and-place, within reach of its start otherwise
            # (a pick that closed on nothing leaves it where it was).
            origin = np.asarray(acted_object["centroid"][:2], dtype=float)
            places = [(origin, allowed)]
            if primitive == "clear" and info.get("drop_xy") is not None:
                places.insert(0, (np.asarray(info["drop_xy"], dtype=float),
                                  0.10))
            found, acted_after = False, None
            for place, limit in places:
                if not remaining_a:
                    break
                dist = [float(np.linalg.norm(a - place)) for a in remaining_a]
                k = int(np.argmin(dist))
                if dist[k] <= limit:
                    acted_after = remaining_a.pop(k)  # not collateral
                    found = True
                    break
            if not found:
                # the object the action was aimed at is nowhere it could
                # be: it left the workspace. (This used to claim whatever
                # other object was nearest, so a block pushed off the
                # table went uncharged and earned clearance.)
                lost_objects.append(acted_object)
        all_a = [np.asarray(o["centroid"][:2], dtype=float) for o in obs_a]
        for i, o in enumerate(obs_b):
            if i == acted_obs_index:
                continue
            start = np.asarray(o["centroid"][:2], dtype=float)
            dists = [float(np.linalg.norm(a - start)) for a in remaining_a]
            if dists and min(dists) <= 0.20:
                k = int(np.argmin(dists))
                disturbance += dists[k]
                remaining_a.pop(k)
                continue
            # No unclaimed counterpart. If an already-claimed after-object
            # sits within two block widths, the two merged into one cluster:
            # collateral displacement, charged by distance. Otherwise the
            # object left the workspace: lost.
            near_any = ([float(np.linalg.norm(a - start)) for a in all_a]
                        if all_a else [])
            if near_any and min(near_any) <= 0.08:
                disturbance += min(near_any)
            else:
                lost_objects.append(o)
        # Pieces. What is still unclaimed in the after-picture has no
        # counterpart before: it broke off something, in practice off the
        # pile that was acted on. Breaking a pile apart is fine and often
        # the whole point. THROWING pieces is not. A piece is charged for
        # the distance it lies beyond where the action could have carried
        # it: the pile's own radius plus the commanded push distance plus
        # 3 cm. (Before this, a pushed pile was exempt as "the acted-on
        # object" and its pieces were counted nowhere, so scattering a pile
        # cost nothing and earned the split bonus.)
        if acted_object is not None and remaining_a:
            origin = np.asarray(acted_object["centroid"][:2], dtype=float)
            for piece in remaining_a:
                disturbance += max(
                    0.0, float(np.linalg.norm(piece - origin)) - allowed)
        terms["disturb"] = -w["disturb"] * disturbance
        # an obstacle pushed toward the red block: no refusal on the
        # server any more, the reward says it is wrong
        if (acted_object is not None and not acted_is_target
                and tb is not None and ta is not None
                and acted_after is not None):
            red_b = np.asarray(tb["centroid"][:2], dtype=float)
            red_a = np.asarray(ta["centroid"][:2], dtype=float)
            gap_b = float(np.linalg.norm(
                np.asarray(acted_object["centroid"][:2], dtype=float) - red_b))
            gap_a = float(np.linalg.norm(acted_after - red_a))
            closer = gap_b - gap_a - 0.005
            if closer > 0.0:
                terms["crowd"] = -w["crowd"] * closer
        terms["lost"] = -w["lost"] * len(lost_objects)

        # ---- the action could not be executed ----
        # The server says why in push_refused (build 2026-10-22+); older
        # servers only have push_blocked, which also covers refusals.
        if primitive == "push" and not info.get("no_obstacle"):
            if "push_refused" in info:
                refused = bool(info.get("push_refused"))
            else:
                refused = bool(info.get("push_blocked"))
            if info.get("table_contact"):
                refused = True      # charged like a refused action
            if refused:
                terms["refused"] = -w["refused"]
                if self.verbose:
                    print("[rule] the push was refused (%s): %.1f"
                          % (info.get("push_refused") or "blocked",
                             terms["refused"]))

        # ---- progress toward a graspable target ----
        if tb is not None and ta is not None:
            cb = route2_perception.grasp_clearance(
                tb, before.get("obstacles", []))
            # An obstacle that left the table must not count as "cleared":
            # for this term it stays where it was. Otherwise knocking the
            # nearest block off the table earned up to +1.6 of clearance
            # against a loss charge of 1.0.
            ca = route2_perception.grasp_clearance(
                ta, obs_a + lost_objects)
            # Pay only for the space the grasp needs (the grasp circle plus
            # 2 cm). Pushing an obstacle far beyond that earns nothing
            # more; it only sends blocks toward the table edges.
            cap = self.clearance_min + 0.02
            terms["clearance"] = w["clearance"] * (min(ca, cap)
                                                   - min(cb, cap))
            terms["pose"] = self._pose_value(ta) - self._pose_value(tb)
        seen_b, seen_a = tb is not None, ta is not None
        terms["expose"] = w["expose"] * (int(seen_a) - int(seen_b))
        if not (seen_b and seen_a):
            terms["split"] = w["split"] * (
                int(after.get("n_objects", 0))
                - int(before.get("n_objects", 0)))
        if ta is not None:
            radial = float(np.hypot(ta["centroid"][0], ta["centroid"][1]))
            radial_b = (float(np.hypot(tb["centroid"][0], tb["centroid"][1]))
                        if tb is not None else 0.0)
            if radial > self.max_radial and radial_b <= self.max_radial:
                terms["outreach"] = -w["outreach"]
            # too CLOSE is out of reach too: the near table edge
            if (radial < self.min_radial and tb is not None
                    and radial_b >= self.min_radial):
                terms["outreach"] = -w["outreach"]
        if self.min_radial > 0.0:
            def inside(o):
                c = o["centroid"]
                return float(np.hypot(c[0], c[1])) < self.min_radial
            n_b = sum(1 for o in obs_b if inside(o))
            n_a = sum(1 for o in obs_a if inside(o))
            if n_a > n_b:
                terms["near_edge"] = -w["near_edge"] * (n_a - n_b)
        return float(sum(terms.values())), terms

    def _pose_value(self, target):
        """How easy the red block is to grasp where and how it lies, as a
        number <= 0 (0 = easy). Two parts:
          * how far it lies in the outer 3 cm of the reach ring (near
            MIN_RADIAL or MAX_RADIAL), per metre;
          * for a bar: how much the jaw would have to close ALONG the line
            to the robot (the tilted fingers then end at different heights,
            the awkward grasp), 0..1.
        Used as a potential: the reward is value(after) - value(before), so
        it cannot be farmed by pushing back and forth."""
        if target is None:
            return 0.0
        c = target["centroid"]
        radial = float(np.hypot(c[0], c[1]))
        lo, hi = self.min_radial + 0.03, self.max_radial - 0.03
        edge = max(0.0, lo - radial) + max(0.0, radial - hi)
        angle = 0.0
        long_side, short_side = target.get("rect_long"), \
            target.get("rect_short")
        if (target.get("yaw") is not None and long_side and short_side
                and long_side / max(short_side, 1e-3) > 1.3):
            closing = float(target["yaw"]) + np.pi / 2.0
            azimuth = float(np.arctan2(c[1], c[0]))
            angle = abs(np.cos(closing - azimuth))
        return -(self.w["pose_edge"] * edge + self.w["pose_angle"] * angle)

    def _feasibility(self, view):
        return route2_perception.grasp_feasibility(
            view, clearance_min=self.clearance_min,
            max_radial=self.max_radial, min_radial=self.min_radial)

    def _point_at_anchor(self):
        """Hold the closed gripper 8 cm above the anchor so the person
        setting the scene sees exactly where the red block goes. One move;
        the caller sends the arm home before the photo. False if the arm
        cannot (or should not) do it."""
        if self.route2 is None or not self.show_anchor:
            return False
        try:
            anchor = np.asarray(self.route2.anchor, dtype=float)
            table_z = float(self.route2.backend.config.table_z)
            self.env._send({"cmd": "gripper", "state": "close"})
            resp = self.env._send({"cmd": "goto", "xyz": [
                float(anchor[0]), float(anchor[1]), table_z + 0.08]})
            return bool(resp.get("ok"))
        except Exception as error:
            print("[scene] could not point at the anchor (%s)" % error)
            return False

    def _start_problem(self, summary):
        """Why this scene is not worth starting an episode on, or None."""
        if not summary:
            return None
        if summary.get("n_objects", 0) == 0:
            return "empty"
        if summary.get("target_found"):
            c = summary["target"]["centroid"]
            radial = float(np.hypot(c[0], c[1]))
            # (no position check: the red block stays where it was put;
            # pushes that cannot be done are refused and charged)
            pass
            return ("graspable" if self._feasibility(summary)["graspable"]
                    else None)
        if self.last_end == "target_lost":
            return "lost"
        if self.blind_episodes >= self.max_blind_episodes:
            return "unseen"
        return None

    reach_note = ""

    @property
    def PROBLEM_TEXT(self):
        return dict(self._PROBLEM_TEXT, reach=(
            self.reach_note + " From there almost every push starts out of "
            "the arm's reach and is refused, so nothing can be learned. "
            "Move the scene under the gripper."))

    _PROBLEM_TEXT = {
        "empty": "The camera sees nothing on the table.",
        "graspable": "The red block can already be grasped as it lies. "
                     "There is nothing to clear, so nothing to learn.",
        "lost": "The red block left the table in the last episode and is "
                "still not in view.",
        "unseen": "The red block has not been seen for several episodes "
                  "in a row. It may have left the table inside a pile.",
    }

    def start_episode(self):
        """Reset the robot, settle the target identity for this episode,
        and make sure the scene is worth an episode. With --pause 1 the
        script waits here, arm parked, until a human has rebuilt a scene
        that is not trivial."""
        self.ep_t = 0
        first = True
        while True:
            obs = self.env.reset()
            used_up = self.last_end in (None, "success", "target_lost",
                                        "outreach", "cleanup")
            if first and (self.reset_prompt >= 2
                          or (self.reset_prompt == 1 and used_up)):
                pointing = self._point_at_anchor()
                answer = ask_human(
                    "SETUP", "%s %s Take your hands away, then press "
                    "Enter." % (
                        "The gripper is pointing at the spot: put the red "
                        "block right under it." if pointing else
                        "The arm is parked.", self.scene_text))
                self.stuck = False
                if pointing:
                    obs = self.env.reset()      # arm home before the photo
                if answer == "q":
                    raise StopTraining("stopped at the scene setup")
            first = False
            self.home_state = np.array(obs["state"][:PROPRIO_DIM],
                                       dtype=np.float32)
            self.image_size = int(obs["rgb"].shape[0])
            summary, problem = {}, None
            if self.route2 is not None:
                try:
                    frame = self.env.frame()
                    self._dump_frame(frame, "reset_%05d" % self._dump_count)
                    summary = self.route2.start_episode(
                        frame, ee_xyz=obs["state"][5:8])
                    problem = self._start_problem(summary)
                    self._print_start(summary)
                except Exception as error:
                    print("[route2] WARNING: start_episode failed (%s)"
                          % error)
                    summary = {}
            if problem is None:
                break
            if not self.pause:
                print("[scene] %s (--pause 0: going on anyway)"
                      % self.PROBLEM_TEXT[problem])
                break
            for d in summary.get("detections") or []:
                print("[scene]   %s at (%.2f, %.2f)"
                      % (d.get("label", "object"), d["centroid"][0],
                         d["centroid"][1]))
            pointing = self._point_at_anchor()
            answer = ask_human(
                "SCENE", "%s The arm is parked and will not move. Rebuild "
                "the clutter (red block hidden or boxed in by the others, "
                "all inside the reachable band), take your hands away, "
                "then press Enter." % self.PROBLEM_TEXT[problem])
            if pointing:
                self.env.reset()
            if answer == "q":
                raise StopTraining("stopped at the scene check")
            if answer == "g":
                break
            # a human touched the scene: what was true of it no longer is
            self.last_end, self.blind_episodes = None, 0
        self.view = summary if (summary and self.home_each_step) else None
        self.saw_target = bool(summary.get("target_found"))
        self.started_graspable = problem == "graspable"
        self.failed_grasp_xy = None
        return obs, summary

    def _print_start(self, summary):
        if not self.verbose:
            return
        if not summary.get("target_found"):
            print("[route2] no identifiable target at reset (%d objects on "
                  "the table). If the target starts hidden inside the pile "
                  "this is expected and the policy must dig it out."
                  % summary.get("n_objects", -1))
            return
        feas = self._feasibility(summary)
        print("[route2] episode target at %s, %d objects on the table, "
              "clearance %s -> %s"
              % (np.round(summary["target_xy"], 3).tolist(),
                 summary["n_objects"],
                 ("%.3f" % feas["clearance"]
                  if feas["clearance"] is not None else "n/a"),
                 "already graspable" if feas["graspable"]
                 else "not graspable (%s)" % feas["reason"]))

    def _end_episode(self, end):
        """Book-keeping the next start_episode needs."""
        self.last_end = end
        self.ep_refused = 0
        self.blind_episodes = (0 if self.saw_target or end == "success"
                               else self.blind_episodes + 1)
        self.view = None

    def _planner_grasp(self, view, obs):
        """Run the scripted grasp of the target and decide success.
        Returns (next_obs, done_flag, success, held, target_on_table)."""
        perception = {"target": view["target"],
                      "obstacles": view["obstacles"]}
        next_obs, done, info = self.env.step(
            [1.0, 0.0, 0.0, 0.0, 0.0], perception, grasp_role="target")
        for reason in info.get("motion_errors", []):
            print("[motion] %s" % reason)
        held = bool(info.get("held", False))
        success, target_on_table = False, None
        if held:
            # Home first: the lifted arm stands on the camera's line of
            # sight to the target's old spot. One motion per episode end,
            # which the reset was going to make anyway.
            try:
                self.env.home()
            except Exception as error:
                print("[route2] WARNING: homing before the confirmation "
                      "frame failed (%s)" % error)
            try:
                after = self.route2.target_on_table(
                    self.env.frame(), exclude_xyz=next_obs["state"][5:8])
                target_on_table = bool(after["target_on_table"])
                success = not target_on_table
            except Exception as error:
                print("[route2] WARNING: table check failed (%s); scored "
                      "as a failure" % error)
        return next_obs, done, success, held, target_on_table, info

    def step(self, action, ee_xyz=None):
        """One clearing action, then the rule-triggered grasp if it
        applies. Returns (next_obs, reward, done, record).

        record["terminal"] says whether the task really ended (success or
        target lost). `done` is also True when the step budget ran out;
        that case is NOT terminal and must not be stored as such."""
        is_pick = self.allow_pick and float(action[0]) >= 0.0
        primitive = "clear" if is_pick else "push"
        # What the SERVER gets. The server reads a[0] itself: a[0] >= 0 is
        # a grasp there. With allow_pick off the client labels every step
        # a push, so a[0] must be forced negative before sending, or the
        # server runs a blind grasp at the stale target position while the
        # log says "push". (That is what happened in the 1014b run: 20 of
        # 44 steps were such grasps.) The replay keeps the policy's own
        # action; the environment simply ignores a[0], which the critic
        # learns.
        to_server = np.asarray(action, dtype=float).copy()
        if not self.allow_pick:
            to_server[0] = -1.0
        perception, before = None, {}
        if self.route2 is not None:
            try:
                if self.home_each_step and self.view is not None:
                    # the clean picture the last step ended on IS this
                    # step's "before": same pose, no second frame
                    before = self.view
                else:
                    before = self._perceive(
                        ee_xyz, "step_%05d" % self._dump_count)
                perception = {"target": before["target"],
                              "obstacles": before["obstacles"],
                              "pushables": self._pushables(before)}
            except Exception as error:
                print("[route2] WARNING: perceive failed (%s); the server "
                      "will fall back to its own point cloud" % error)

        if (not is_pick and self.push_frame == "relative" and perception
                and perception.get("pushables")):
            to_server[2:4] = self._world_direction(action, perception)

        after, clean = None, False
        shaping, shaping_terms = 0.0, {}
        success, held, on_table, auto = False, False, None, False
        target_lost, grasp_attempts = False, 0
        cleanup_end = False   # scene was already graspable: cleanup only
        grasp_now, skip_note = False, ""
        if (self.one_action and self.grasp_trigger == "auto"
                and self.route2 is not None and before
                and before.get("target") is not None):
            feas = self._feasibility(before)
            if feas["graspable"] and not (self.home_each_step
                                          and self.view is before):
                # the picture was taken with the arm over the table; look
                # again from home before deciding to grasp
                try:
                    clean_before, _, _ = self._look(
                        None, "clean_%05d" % self._dump_count,
                        force_home=True)
                    if clean_before.get("target") is not None:
                        before = clean_before
                        perception = {"target": before["target"],
                                      "obstacles": before["obstacles"],
                                      "pushables": self._pushables(before)}
                    feas = self._feasibility(clean_before)
                except Exception as error:
                    print("[route2] WARNING: clean look failed (%s)" % error)
            if feas["graspable"]:
                txy = np.asarray(before["target"]["centroid"][:2])
                if (self.failed_grasp_xy is not None and float(
                        np.linalg.norm(txy - self.failed_grasp_xy)) < 0.01):
                    skip_note = ("graspable, but the last grasp missed here "
                                 "and the target has not moved")
                else:
                    grasp_now = True
        if grasp_now:
            # ONE primitive: the planner grasp. The policy's action is not
            # executed in this step. (In a graspable scene the environment
            # always grasps, whatever the policy picks, so the stored
            # transition is still a correct sample of the dynamics.)
            #
            # The policy's success event is "scene BECAME graspable", so a
            # scene that was ALREADY graspable at the start of the step
            # does not count as success and gets no bonus. The scripted
            # grasp still runs so the table is physically cleared.
            primitive, auto, grasp_attempts, is_pick = "grasp", True, 1, True
            after_feas = self._feasibility(before)
            if self.verbose:
                print("[rule] target already graspable (clearance %.3f, "
                      "radial %.2f): running the scripted grasp as physical "
                      "cleanup, NO success bonus (the policy did not earn "
                      "this)."
                      % (after_feas["clearance"], after_feas["radial"]))
            next_obs, done, grasp_held_ok, held, on_table, info = \
                self._planner_grasp(before, None)
            success = False       # policy did not transition into graspable
            reward = -STEP_COST
            # The policy's action was not executed in this step, so nothing
            # that happened here is charged to it: no shaping, no
            # target_lost. The episode ends whatever the cleanup grasp did
            # (held: the red block is in the gripper and the next reset
            # puts it down; slipped: the next episode starts from a fresh
            # frame). Without this the run went on with the block in the
            # jaw and the next frame was scored as "red block lost".
            cleanup_end = True
            if not grasp_held_ok:
                self.failed_grasp_xy = np.asarray(
                    before["target"]["centroid"][:2], dtype=float)
            try:
                after, next_obs, clean = self._look(
                    next_obs, "after_%05d" % self._dump_count,
                    force_home=True)
                after_feas = self._feasibility(after)
            except Exception as error:
                print("[route2] WARNING: perception after the cleanup "
                      "grasp failed (%s)" % error)
                after = None
        elif self.grasp_trigger == "policy" and is_pick:
            # Ablation mode: the policy grasps the target itself.
            next_obs, done, success, held, on_table, info = \
                self._planner_grasp(before, None) if before else \
                (None, True, False, False, None, {})
            primitive = "grasp"
            grasp_attempts = 1
            reward = -STEP_COST
            after_feas = {"graspable": None, "clearance": None,
                          "reason": ""}
            if (self.route2 is not None and self.home_each_step
                    and next_obs is not None and not success):
                try:
                    after, next_obs, clean = self._look(
                        next_obs, "after_%05d" % self._dump_count)
                except Exception as error:
                    print("[route2] WARNING: post-grasp perceive failed "
                          "(%s)" % error)
        else:
            next_obs, done, info = self.env.step(
                to_server, perception,
                grasp_role="obstacle" if is_pick else "target")
            for reason in info.get("motion_errors", []):
                print("[motion] %s" % reason)
            reward = -STEP_COST
            after_feas = {"graspable": False, "clearance": None,
                          "reason": "route 2 off"}
            if self.route2 is not None:
                try:
                    after, next_obs, clean = self._look(
                        next_obs, "after_%05d" % self._dump_count)
                    # Seen before, not seen now: merged into a pile, or
                    # off the table? Settle that before scoring the step.
                    if (before.get("target") is not None
                            and after.get("target") is None):
                        target_lost, after, next_obs, clean = \
                            self._target_gone(
                                before, after, next_obs, clean,
                                "gone_%05d" % self._dump_count)
                    after_feas = self._feasibility(after)
                    shaping, shaping_terms = self._shaping(
                        before, after, info, primitive)
                    if target_lost:
                        shaping_terms["target_lost"] = -self.w["target_lost"]
                        shaping += shaping_terms["target_lost"]
                        if self.verbose:
                            print("[rule] the red block left the table: "
                                  "%.1f, episode over"
                                  % shaping_terms["target_lost"])
                    reward += shaping
                except Exception as error:
                    print("[route2] WARNING: post-step perceive failed "
                          "(%s)" % error)
                    after = None
                # A clearing pick applied to a pile may come up with the
                # RED block. That is a grasp of the target: it was lifted,
                # carried and set down. If the target is now identified at
                # the drop point, credit it as success (--credit_clear_grasp).
                if (self.credit_clear_grasp and is_pick and after is not None
                        and info.get("carried") and after.get("target")
                        and info.get("drop_xy") is not None):
                    drop = np.asarray(info["drop_xy"], dtype=float)
                    at = np.asarray(after["target"]["centroid"][:2])
                    if float(np.linalg.norm(at - drop)) < 0.07:
                        success, held, on_table = True, True, False
                        if self.verbose:
                            print("[rule] the clearing pick carried the red "
                                  "block to the drop point: success")
                # NEW DESIGN (Oct 2026): the policy's job is to make the
                # scene graspable. The moment the clean frame says the red
                # block is graspable, that is the policy's success and the
                # episode ends. The scripted grasp then runs ONCE as
                # physical cleanup so the table is clear for the next
                # episode; its outcome does NOT change the reward. (The
                # old code retried the scripted grasp up to grasp_retries
                # times and only credited success when it lifted the
                # block, which punished the policy for calibration noise.)
                if skip_note and self.verbose:
                    print("[rule] %s; no cleanup grasp this step" % skip_note)
                # Only a "becomes graspable" transition counts as success.
                # If before was already graspable (e.g. one_action is off,
                # or a race), the policy did not earn the bonus.
                before_graspable = (self._feasibility(before)["graspable"]
                                    if before is not None else False)
                if (self.grasp_trigger == "auto" and not success
                        and not target_lost and after is not None
                        and after_feas["graspable"]
                        and not before_graspable):
                    if not clean:
                        # The firing frame was taken with the arm over the
                        # table; home and look again before crediting it.
                        try:
                            view, next_obs, clean = self._look(
                                next_obs, "clean_%05d" % self._dump_count,
                                force_home=True)
                            feas = self._feasibility(view)
                            after, after_feas = view, feas
                        except Exception as error:
                            print("[route2] WARNING: clean re-perception "
                                  "failed (%s); using the hovering frame"
                                  % error)
                    if after_feas["graspable"]:
                        success, auto, grasp_attempts = True, True, 1
                        if self.verbose:
                            print("[rule] target became graspable "
                                  "(clearance %.3f, radial %.2f): +%.1f, "
                                  "episode ends. Scripted grasp runs as "
                                  "physical cleanup; its outcome does not "
                                  "change the reward."
                                  % (after_feas["clearance"],
                                     after_feas["radial"],
                                     GRASP_SUCCESS_BONUS))
                        try:
                            grasp_obs, done2, _, held, on_table, ginfo = \
                                self._planner_grasp(after, next_obs)
                            info["grasp_info"] = ginfo
                            next_obs = grasp_obs
                            done = done or done2
                        except Exception as error:
                            print("[rule] WARNING: cleanup grasp failed "
                                  "(%s); the +%.1f reward still stands"
                                  % (error, GRASP_SUCCESS_BONUS))
                    elif self.verbose:
                        print("[rule] graspable from the hovering frame but "
                              "not from the clean one (%s); no success"
                              % after_feas["reason"])
        # Outreach is now TERMINAL: a red block pushed beyond the arm's
        # reach ends the episode just like it falling off the table.
        outreach_hit = bool(shaping_terms.get("outreach"))
        if success:
            # ON TOP of the step cost and the shaped terms, not instead of
            # them: a last push that also knocks a block off the table must
            # score lower than a clean one.
            reward += GRASP_SUCCESS_BONUS
            done = True
        if target_lost or outreach_hit or cleanup_end:
            done = True
        self.ep_t += 1
        if shaping_terms.get("refused"):
            self.ep_refused += 1
        if self.max_ep_steps and self.ep_t >= self.max_ep_steps:
            done = True             # time limit: NOT terminal
        # "cleanup" ends the episode but is not a terminal state of the
        # task (nothing the policy did decided it), so it bootstraps like
        # a time limit.
        terminal = bool(success or target_lost or outreach_hit)
        if after is not None and after.get("target") is not None:
            self.saw_target = True
        self.view = after if (self.home_each_step and clean
                              and after is not None) else None
        if done:
            self._end_episode("success" if success else
                              "target_lost" if target_lost else
                              "outreach" if outreach_hit else
                              "cleanup" if cleanup_end else "time_limit")

        # Displacement of the obstacle the action was applied to: the
        # "what did this push/pick do" data for the action-side invariance
        # claim (same action parameters, different objects -> same
        # displacement?). Matched by nearest centroid after the action.
        disp = self._obstacle_displacement(before, info, after
                                           if self.route2 is not None and
                                           after is not None else None)
        if is_pick:
            push_dir, push_dist = (None, None), None
        else:
            d = np.asarray(to_server[2:4], dtype=float)
            n = float(np.linalg.norm(d))
            push_dir = ((d / n).tolist() if n > 1e-6 else (None, None))
            push_dist = info.get("push_distance")

        target_xy = before.get("target_xy")
        record = {
            "primitive": primitive,
            "obstacle_index": info.get("obstacle_index"),
            "pushed_target": (bool(info.get("pushed_target", False))
                              if primitive == "push" else None),
            "push_dir_x": push_dir[0], "push_dir_y": push_dir[1],
            "push_distance": (None if push_dist is None
                              else round(float(push_dist), 4)),
            "obst_before_x": disp["before"][0], "obst_before_y": disp["before"][1],
            "obst_after_x": disp["after"][0], "obst_after_y": disp["after"][1],
            "obst_disp": disp["norm"],
            "target_found_before": bool(before.get("target_found", False)),
            "target_source": before.get("target_source", ""),
            "arm_blobs": int(before.get("arm_blobs", 0)),
            "n_objects_before": int(before.get("n_objects", -1)),
            "target_x": (None if target_xy is None else round(target_xy[0], 4)),
            "target_y": (None if target_xy is None else round(target_xy[1], 4)),
            "graspable_after": after_feas.get("graspable"),
            "clearance_after": (None if after_feas.get("clearance") is None
                                else round(after_feas["clearance"], 4)),
            "rule_reason": after_feas.get("reason", ""),
            "auto_grasp": auto,
            "held": held,
            "gripper_raw": int((info.get("grasp_info") or info)
                               .get("gripper_raw", -1)),
            "target_on_table_after": on_table,
            "success": success,
            "r_shaping": round(shaping, 4),
            "r_clearance": round(shaping_terms.get("clearance", 0.0), 4),
            "r_expose": round(shaping_terms.get("expose", 0.0), 4),
            "r_split": round(shaping_terms.get("split", 0.0), 4),
            "r_disturb": round(shaping_terms.get("disturb", 0.0), 4),
            "r_lost": round(shaping_terms.get("lost", 0.0), 4),
            "r_outreach": round(shaping_terms.get("outreach", 0.0), 4),
            "r_refused": round(shaping_terms.get("refused", 0.0), 4),
            "r_near_edge": round(shaping_terms.get("near_edge", 0.0), 4),
            "r_pose": round(shaping_terms.get("pose", 0.0), 4),
            "r_hit_target": round(shaping_terms.get("hit_target", 0.0), 4),
            "r_crowd": round(shaping_terms.get("crowd", 0.0), 4),
            "motion_errors": " | ".join(info.get("motion_errors", [])),
            "r_target_lost": round(shaping_terms.get("target_lost", 0.0), 4),
            "target_lost": target_lost,
            "terminal": terminal,
            "grasp_attempts": grasp_attempts,
            "end": (self.last_end if done else ""),
            "started_graspable": self.started_graspable,
        }
        return next_obs, reward, done, record


# ---------------- training and evaluation ----------------

def run_eval(args, agent, runner, device, obs):
    path = os.path.join(args.log_dir, "eval_episodes.csv")
    with open(path, "a", newline="") as eval_file:
        writer = csv.writer(eval_file)
        writer.writerow(["wall_time", "episode", "return", "length",
                         "success", "grasps", "pushes"])
        n_success = 0
        for episode in range(1, args.episodes + 1):
            if episode > 1:     # main() already started the first one
                obs, _ = runner.start_episode()
            episode_return, length = 0.0, 0
            grasps, pushes, success = 0, 0, False
            done = False
            while not done:
                state, rgb, depth = observation_tensors(obs, device)
                action = agent.actor.act(
                    state, rgb, depth, deterministic=True)[0]
                obs, reward, done, record = runner.step(
                    action.cpu().numpy(), ee_xyz=obs["state"][5:8])
                episode_return += reward
                length += 1
                grasps += int(record["auto_grasp"]
                              or record["primitive"] == "grasp")
                pushes += int(record["primitive"] == "push")
                success = success or record["success"]
            n_success += int(success)
            print("[eval %2d/%d] return=%7.2f len=%2d success=%s "
                  "grasp=%d push=%d"
                  % (episode, args.episodes, episode_return, length,
                     success, grasps, pushes))
            writer.writerow([time.time(), episode, episode_return, length,
                             int(success), grasps, pushes])
            eval_file.flush()
        print("[eval] success %d/%d" % (n_success, args.episodes))


def open_csv(path, columns):
    """Open a log for appending. The header is written only into a new
    file. An existing file with different columns (an older version of
    this script) is set aside, never mixed into."""
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        with open(path, newline="") as handle:
            first = next(csv.reader(handle), [])
        if first != columns:
            aside = "%s.%s.old" % (path, time.strftime("%m%d_%H%M%S"))
            os.rename(path, aside)
            print("[train] %s has different columns; moved to %s"
                  % (path, aside))
    new = not os.path.isfile(path) or os.path.getsize(path) == 0
    handle = open(path, "a", newline="")
    writer = csv.writer(handle)
    if new:
        writer.writerow(columns)
    return handle, writer


# command-line name -> server config name, and the value used when the
# server is too old to report it
SERVER_PARAMS = [
    ("table_z", "TABLE_Z", 0.225),
    ("ws_x_min", "WS_X_MIN", 0.28), ("ws_x_max", "WS_X_MAX", 0.60),
    ("ws_y_min", "WS_Y_MIN", -0.25), ("ws_y_max", "WS_Y_MAX", 0.25),
    ("max_radial", "MAX_RADIAL", 0.38), ("min_radial", "MIN_RADIAL", 0.0),
    ("grasp_clearance", "GRASP_CLEARANCE", 0.06),
    ("target_max_extent", "TARGET_MAX_EXTENT", 0.085),
]


MIN_SERVER_BUILD = "2026-10-29g"   # jaw closed once, never opened
PUSH_RED = False      # set from --push_red in main()


def params_from_server(env, args):
    """Fill the geometry arguments from the server's stored calibration.
    Stops when a value given on the command line disagrees with it."""
    try:
        build = env._send({"cmd": "ping"}).get("build") or ""
    except Exception:
        build = ""
    if build and build < MIN_SERVER_BUILD:
        raise SystemExit(
            "[params] the robot runs server build %s; this script needs %s "
            "or newer. Copy the new grasp_env_server_rgbd.py to the robot "
            "and restart it:\n    scp grasp_env_server_rgbd.py "
            "locobot@locobot.local:~/\n    (robot) source ~/server_env.sh "
            "&& python ~/grasp_env_server_rgbd.py" % (build, MIN_SERVER_BUILD))
    print("[params] robot server build %s" % (build or "unknown"))
    try:
        state = env._send({"cmd": "get_config"})
        config = state.get("config") or {}
        meta = state.get("meta") or {}
    except Exception as error:
        print("[params] WARNING: the server does not report its settings "
              "(%s); using command-line values and old defaults" % error)
        config, meta = {}, {}
    clashes, missing = [], []
    for flag, key, fallback in SERVER_PARAMS:
        given = getattr(args, flag)
        if key in config:
            value = float(config[key])
            if given is not None and abs(float(given) - value) > 0.005:
                clashes.append("--%s %.3f but the server has %s=%.3f"
                               % (flag, given, key, value))
            setattr(args, flag, value)
        else:
            if given is None:
                setattr(args, flag, fallback)
                missing.append(key)
    if "TARGET_X" in config and "TARGET_Y" in config:
        anchor = (float(config["TARGET_X"]), float(config["TARGET_Y"]))
        if args.anchor is not None:
            ax, ay = (float(v) for v in args.anchor.split(","))
            if abs(ax - anchor[0]) > 0.005 or abs(ay - anchor[1]) > 0.005:
                clashes.append("--anchor %s but the server has %.3f,%.3f"
                               % (args.anchor, anchor[0], anchor[1]))
        args.anchor = "%.3f,%.3f" % anchor
    elif args.anchor is None:
        args.anchor, missing = "0.40,0.03", missing + ["TARGET_X/Y"]
    if clashes:
        raise SystemExit(
            "[params] the command line disagrees with the robot server:\n  "
            + "\n  ".join(clashes) +
            "\nThe server holds the calibrated numbers. Remove these flags "
            "from the command (the script reads them from the server), or "
            "run calibrate_all.py again if the table or camera was moved.")
    if missing:
        print("[params] WARNING: the server does not report %s (old "
              "server build); defaults used" % ", ".join(missing))
    print("[params] from the robot server%s:"
          % (" (calibrated %s)" % meta["saved_at"]
             if meta.get("saved_at") else ""))
    print("[params]   table_z=%.3f  work area x %.2f..%.2f  y %.2f..%.2f  "
          "anchor=%s" % (args.table_z, args.ws_x_min, args.ws_x_max,
                         args.ws_y_min, args.ws_y_max, args.anchor))
    print("[params]   reach %.2f..%.2f m  grasp circle %.3f m  target max "
          "size %.3f m" % (args.min_radial, args.max_radial,
                           args.grasp_clearance, args.target_max_extent))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="locobot.local")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--steps", type=int, default=None,
                        help="training primitives to execute")
    parser.add_argument("--load", default=None)
    parser.add_argument("--save", default=None)
    parser.add_argument("--save_replay", default=None,
                        help=".npz path for the raw (s,a,r,s') buffer; "
                             "written with every checkpoint and at the "
                             "end. Needed for the transition-rank analysis.")
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--utd", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=30,
                        help="pure random primitives before the policy acts")
    parser.add_argument("--min_buffer", type=int, default=16)
    parser.add_argument("--replay_size", type=int, default=10000)
    parser.add_argument("--factored_dim", type=int, default=64)
    parser.add_argument("--visual_dim", type=int, default=128)
    parser.add_argument("--depth_max_m", type=float, default=1.5)
    parser.add_argument("--aug_pad", type=int, default=4,
                        help="DrQ random-shift padding; 0 disables it")
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    # transfer experiment hooks
    parser.add_argument("--freeze_phia", type=int, default=0,
                        help="1 = freeze the action-side factor and retrain "
                             "phi_s (objects change, the arm does not)")
    parser.add_argument("--init_visual", default=None,
                        help="checkpoint with pretrained RGBDEncoder weights")
    parser.add_argument("--freeze_visual", type=int, default=0)
    # route 2
    parser.add_argument("--perception",
                        choices=["segmentation", "geometric", "off"],
                        default="geometric",
                        help="off disables route 2 entirely: the server uses "
                             "its own point cloud and success falls back to "
                             "the gripper signal alone")
    parser.add_argument("--identity", choices=["color", "track"],
                        default="color",
                        help="how the target keeps its identity across "
                             "steps. color: the target IS the object with "
                             "--target_label's hue (the task's definition: "
                             "one red block among green ones); no position "
                             "involved, no fallback when it is hidden. "
                             "track: distance tracking, for label-less "
                             "detectors such as a single-class YOLO.")
    parser.add_argument("--target_label", default="red_object",
                        help="colour of the target for --identity color. "
                             "Matching is by hue distance, so lighting "
                             "drift across a colour-name boundary does not "
                             "break identity.")
    parser.add_argument("--hue_tol", type=float, default=35.0,
                        help="max hue distance (degrees) for the target "
                             "match. Keep well below the target-to-"
                             "obstacle hue gap (red to green is 120).")
    parser.add_argument("--track_max", type=float, default=0.05,
                        help="largest accepted jump of the target between "
                             "steps. Beyond it route 2 reports no target "
                             "instead of latching onto a neighbour. Must stay "
                             "well below the spacing between objects: at 0.10 "
                             "it equals that spacing and will grab the wrong "
                             "object when the target is occluded.")
    # The geometry below is NOT given on the command line any more. It is
    # read from the robot server at start-up, which holds the numbers that
    # calibrate_all.py measured and saved. One set of numbers, one place.
    # A flag given here must agree with the server (within 5 mm), otherwise
    # the run stops; that catches a stale copy-pasted command.
    parser.add_argument("--anchor", default=None,
                        help="(from the server: TARGET_X,TARGET_Y)")
    parser.add_argument("--table_z", type=float, default=None,
                        help="(from the server: TABLE_Z)")
    parser.add_argument("--ws_x_min", type=float, default=None)
    parser.add_argument("--ws_x_max", type=float, default=None)
    parser.add_argument("--ws_y_min", type=float, default=None)
    parser.add_argument("--ws_y_max", type=float, default=None)
    parser.add_argument("--min_radial", type=float, default=None,
                        help="(from the server: MIN_RADIAL) inner reach "
                             "limit; the near table edge")
    parser.add_argument("--grasp_trigger", choices=["auto", "policy"],
                        default="auto",
                        help="auto: the scripted grasp runs by rule as soon "
                             "as the target is graspable (the policy only "
                             "clears). policy: a[0]>=0 grasps the target "
                             "(ablation).")
    parser.add_argument("--grasp_clearance", type=float, default=None,
                        help="radius (m) of the circle drawn at the target's "
                             "xy centroid; the grasp rule fires only when no "
                             "obstacle point lies inside this circle")
    parser.add_argument("--graspable_rule", choices=["fingers", "circle"],
                        default="fingers",
                        help="fingers (default, Oct 30): only the two places "
                             "where the fingers come down, along the closing "
                             "direction the grasp will use, must be free out "
                             "to --grasp_clearance from the target centre. "
                             "circle: the old rule, nothing within "
                             "--grasp_clearance in any direction. The "
                             "clearance reward uses the same measure")
    parser.add_argument("--finger_half_width", type=float, default=0.015,
                        help="half width (m) of the strip each finger needs, "
                             "measured across the closing direction: half "
                             "the finger's width plus a margin")
    parser.add_argument("--max_radial", type=float, default=None,
                        help="largest radial distance the planner grasp "
                             "can reach at the configured pitch; from "
                             "reach_scan.py")
    for name, default in DEFAULT_W.items():
        parser.add_argument("--w_%s" % name, type=float, default=default,
                            help="shaped reward weight '%s' (default %.2f)"
                                 % (name, default))
    parser.add_argument("--credit_clear_grasp", type=int, default=1,
                        help="1: a clearing pick that carries the red "
                             "block to the drop point counts as success")
    parser.add_argument("--dump_frames", default=None,
                        help="directory to save every fetched full-res "
                             "RGB-D frame as PNGs (rgb, 16-bit depth, "
                             "depth visualisation). Meant for the short "
                             "safety runs; a 400-step run writes several "
                             "hundred MB.")
    parser.add_argument("--dump_every", type=int, default=1,
                        help="save every N-th frame when --dump_frames is "
                             "set")
    parser.add_argument("--target_max_extent", type=float, default=None,
                        help="max XY size (m) a detection may have and "
                             "still be the target. A single block is under "
                             "0.07; merged piles measured 0.095+ on the "
                             "robot, and one was mistaken for the target "
                             "before this gate existed.")
    parser.add_argument("--model_dir", default=None,
                        help="directory with the YOLO/NanoSAM models")
    parser.add_argument("--pause", type=int, default=1,
                        help="1: at the start of an episode, wait for a "
                             "human when the scene is not worth an episode "
                             "(target already graspable, target gone, "
                             "table empty). The arm is parked while it "
                             "waits. 0: never wait (tests, mock runs).")
    parser.add_argument("--max_ep_steps", type=int, default=3,
                        help="episode length in steps (pushes). The easy "
                             "3-block scene needs 1 push; 3 leaves room for "
                             "one mistake and one correction. 0 = the "
                             "server's HORIZON")
    parser.add_argument("--push_frame", choices=["relative", "world"],
                        default="relative",
                        help="relative: a[2:4] is the push direction in the "
                             "chosen object's frame, x = straight away from "
                             "the red block")
    parser.add_argument("--reset_prompt", type=int, default=1,
                        help="when to ask you to set the scene by hand. "
                             "1 (default): only when the last episode used "
                             "it up (success: the red block was taken away; "
                             "red off the table or out of reach) and at the "
                             "very start. After a plain time limit the "
                             "blocks stay where they are and the next "
                             "episode starts from there. 2: every episode. "
                             "0: never")
    parser.add_argument("--scene", default="Put the red block {where}, "
                        "with one other-coloured block touching it on each "
                        "side, in a row ACROSS the line from the robot to "
                        "the red block (green, red, green).",
                        help="what the setup prompt asks you to build")
    parser.add_argument("--push_red", type=int, default=0,
                        help="0 (default): the red block is never pushed; "
                             "the policy only pushes obstacles away from it. "
                             "1: the red block may be pushed as well")
    parser.add_argument("--show_anchor", type=int, default=0,
                        help="1: while you set the scene, the gripper "
                             "hovers 8 cm over the anchor to show where the "
                             "red block goes. Off by default")
    parser.add_argument("--demo_episodes", type=int, default=0,
                        help="episodes at the start that use the scripted "
                             "move (push the obstacle nearest to red "
                             "straight away from it, a little noise) "
                             "instead of random actions. They go into the "
                             "replay buffer like any other data, so the "
                             "critic sees successes from the first minute. "
                             "Skipped when resuming or loading weights")
    parser.add_argument("--max_blind_episodes", type=int, default=3,
                        help="pause after this many episodes in a row in "
                             "which the target was never seen")
    parser.add_argument("--home_each_step", type=int, default=1,
                        help="1: after every action the arm goes home and "
                             "one frame without the arm is used for the "
                             "next observation and for the reward. 0: the "
                             "old behaviour (frames with the arm hovering "
                             "over the table; a few seconds faster per "
                             "step, noisier reward).")
    parser.add_argument("--one_action", type=int, default=1,
                        help="1: every step is ONE primitive. A step that "
                             "starts with the target graspable is the "
                             "planner grasp; any other step is the "
                             "policy's push/pick; a grasp never follows a "
                             "push in the same step and a missed grasp is "
                             "not repeated until the target moves. 0: the "
                             "old chained behaviour (push, then grasp, "
                             "then --grasp_retries more grasps).")
    parser.add_argument("--allow_pick", type=int, default=1,
                        help="0: a[0] is ignored and every policy action "
                             "is a push (no pick-and-place of obstacles)")
    parser.add_argument("--grasp_retries", type=int, default=1,
                        help="extra planner-grasp attempts when the rule "
                             "fires and the grasp fails, before the policy "
                             "has to act again")
    parser.add_argument("--resume", type=int, default=0,
                        help="1: continue the run stored at --save and "
                             "--save_replay (networks, optimizers, buffer, "
                             "counters); --steps then counts the steps "
                             "still to run. Different from --load, which "
                             "starts a NEW phase from stored weights.")
    parser.add_argument("--load_replay", default=None,
                        help=".npz buffer to start from (written by "
                             "--save_replay). Implied by --resume 1.")
    args = parser.parse_args(argv)

    if not args.eval and args.steps is None:
        parser.error("--steps is required for training")
    if args.eval and not args.load:
        parser.error("--eval 1 requires --load <checkpoint>")
    if args.resume:
        if not args.save or not os.path.isfile(args.save):
            parser.error("--resume 1 needs --save pointing at the "
                         "checkpoint of the run to continue")
        args.load = args.save
        if args.load_replay is None and args.save_replay \
                and os.path.isfile(args.save_replay):
            args.load_replay = args.save_replay

    global PUSH_RED
    PUSH_RED = bool(args.push_red)
    env = RemoteEnv(args.host, args.port)
    params_from_server(env, args)
    ax, ay = (float(v) for v in args.anchor.split(","))
    args.scene = args.scene.replace(
        "{where}", "%.0f cm in front of the robot centre and %.0f cm to its "
        "%s (the anchor)" % (ax * 100, abs(ay) * 100,
                             "right" if ay < 0 else "left"))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)
    print("[train] device=%s freeze_phia=%d freeze_visual=%d perception=%s "
          "grasp_trigger=%s clearance>=%.3f max_radial=%.2f"
          % (device, args.freeze_phia, args.freeze_visual, args.perception,
             args.grasp_trigger, args.grasp_clearance, args.max_radial))
    print("[train] pause=%d home_each_step=%d one_action=%d allow_pick=%d "
          "grasp_retries=%s resume=%d"
          % (args.pause, args.home_each_step, args.one_action,
             args.allow_pick,
             "n/a" if args.one_action else args.grasp_retries, args.resume))

    route2_perception.set_grasp_rule(args.graspable_rule,
                                     args.finger_half_width)
    print("[train] graspable rule: %s (%s)"
          % (args.graspable_rule,
             "the two finger places along the closing direction free out to "
             "%.0f mm, strip %.0f mm wide"
             % ((args.grasp_clearance or 0.06) * 1000,
                2 * args.finger_half_width * 1000)
             if args.graspable_rule == "fingers" else
             "no obstacle point within %.0f mm of the target centre"
             % ((args.grasp_clearance or 0.06) * 1000)))
    route2 = None if args.perception == "off" else build_route2(args)
    try:
        runner = EpisodeRunner(env, route2, dump_dir=args.dump_frames,
                               dump_every=args.dump_every,
                               grasp_trigger=args.grasp_trigger,
                               clearance_min=args.grasp_clearance,
                               max_radial=args.max_radial,
                               min_radial=args.min_radial,
                               max_ep_steps=args.max_ep_steps,
                               push_frame=args.push_frame,
                               reset_prompt=args.reset_prompt,
                               scene_text=args.scene,
                               show_anchor=bool(args.show_anchor),
                               credit_clear_grasp=bool(
                                   args.credit_clear_grasp),
                               weights={k: getattr(args, "w_%s" % k)
                                        for k in DEFAULT_W},
                               pause=bool(args.pause),
                               home_each_step=bool(args.home_each_step),
                               grasp_retries=args.grasp_retries,
                               max_blind_episodes=args.max_blind_episodes,
                               one_action=bool(args.one_action),
                               allow_pick=bool(args.allow_pick))
        try:
            obs, _ = runner.start_episode()
        except StopTraining as stop:
            print("[train] %s; nothing was run" % stop)
            return
        if obs["state"].shape != (OBS_DIM,):
            raise ValueError("expected %d-D state, got %r"
                             % (OBS_DIM, obs["state"].shape))
        image_shape = obs["rgb"].shape[:2]
        print("[train] RGB-D image size=%s" % (image_shape,))
        print("[train] policy input: %sx%s RGB-D pixels + %d-D "
              "proprioception (perception entries of the 45-D state are "
              "NOT fed to the networks)"
              % (image_shape[0], image_shape[1], PROPRIO_DIM))

        agent = SAC(device, d=args.factored_dim, visual_dim=args.visual_dim,
                    depth_max_m=args.depth_max_m,
                    freeze_phia=bool(args.freeze_phia),
                    freeze_visual=bool(args.freeze_visual),
                    aug_pad=args.aug_pad)
        checkpoint = None
        if args.load:
            checkpoint = agent.load(args.load)
            print("[train] loaded %s" % args.load)
        if args.init_visual:
            agent.load_visual(args.init_visual)
        agent.set_freeze(bool(args.freeze_phia), bool(args.freeze_visual))
        step0, episode0 = 0, 0
        if args.resume:
            agent.load_optimizers(checkpoint)
            counters = checkpoint.get("counters") or {}
            step0 = int(counters.get("global_step", 0))
            episode0 = int(counters.get("episode", 0))
            print("[train] resuming after step %d, episode %d"
                  % (step0, episode0))
        n_critic, n_actor = agent.frozen_counts()
        print("[train] frozen critic tensors: %d   frozen actor tensors: %d"
              % (n_critic, n_actor))

        if args.eval:
            try:
                run_eval(args, agent, runner, device, obs)
            except StopTraining as stop:
                print("[eval] %s" % stop)
            return

        replay = Replay(args.replay_size, PROPRIO_DIM, ACT_DIM, image_shape)
        if args.load_replay:
            replay.load(args.load_replay)
        # random steps still owed: none for a new phase from stored weights
        # (as before), the remainder for a resumed run
        if args.resume:
            warmup = step0 + max(0, args.warmup - step0)
        else:
            warmup = args.warmup if args.load is None else 0
        demo_left = (args.demo_episodes
                     if not (args.resume or args.load) else 0)
        if demo_left:
            warmup = 0        # the demonstrations replace the random steps
            print("[train] the first %d episodes use the scripted move "
                  "(demonstrations); learning starts after them"
                  % demo_left)
        recent = []           # (success, length) of the last episodes
        episode_return, episode_length = 0.0, 0
        episode_i = episode0 + 1
        grasps, pushes = 0, 0
        start_time = time.time()
        step = step0
        completed = [step0]

        def checkpoint_now(why):
            # only steps that finished (they are in the buffer and the CSV);
            # an episode cut off midway counts as used, so a resumed run
            # starts the next number
            counters = {"global_step": completed[0],
                        "episode": (episode_i if episode_length > 0
                                    else episode_i - 1)}
            if args.save:
                agent.save(args.save, counters)
            if args.save_replay:
                replay.save(args.save_replay)
            if why:
                print("[train] %s: saved %s%s"
                      % (why, args.save or "(no --save given)",
                         " and %s" % args.save_replay
                         if args.save_replay else ""))

        step_file, step_writer = open_csv(
            os.path.join(args.log_dir, "steps.csv"), STEP_COLUMNS)
        episode_file, episode_writer = open_csv(
            os.path.join(args.log_dir, "episodes.csv"), EPISODE_COLUMNS)
        finished = False
        try:
            for step in range(step0 + 1, step0 + args.steps + 1):
                if demo_left > 0:
                    action = runner.scripted_action()
                    warmup = max(warmup, step)   # no updates yet
                elif step <= warmup:
                    action = np.random.uniform(
                        -1, 1, ACT_DIM).astype(np.float32)
                else:
                    state, rgb, depth = observation_tensors(obs, device)
                    action = agent.actor.act(
                        state, rgb, depth)[0].cpu().numpy()

                next_obs, reward, done, record = runner.step(
                    action, ee_xyz=obs["state"][5:8])
                # terminal, not done: see Replay.add
                replay.add(obs, action, reward, next_obs,
                           float(record["terminal"]))
                episode_return += reward
                episode_length += 1
                grasps += int(record["auto_grasp"]
                              or record["primitive"] == "grasp")
                pushes += int(record["primitive"] == "push")
                obs = next_obs

                blank = lambda v: "" if v is None else v
                step_writer.writerow([
                    time.time(), step, episode_i, episode_length,
                    record["primitive"],
                    blank(record["obstacle_index"]),
                    ("" if record["pushed_target"] is None
                     else int(record["pushed_target"])),
                    blank(record["push_dir_x"]), blank(record["push_dir_y"]),
                    blank(record["push_distance"]),
                    blank(record["obst_before_x"]),
                    blank(record["obst_before_y"]),
                    blank(record["obst_after_x"]),
                    blank(record["obst_after_y"]),
                    blank(record["obst_disp"]),
                    int(record["target_found_before"]),
                    record["target_source"], record["arm_blobs"],
                    record["n_objects_before"],
                    blank(record["target_x"]), blank(record["target_y"]),
                    ("" if record["graspable_after"] is None
                     else int(record["graspable_after"])),
                    blank(record["clearance_after"]), record["rule_reason"],
                    int(record["auto_grasp"]),
                    int(record["held"]), record["gripper_raw"],
                    ("" if record["target_on_table_after"] is None
                     else int(record["target_on_table_after"])),
                    int(record["success"]), reward,
                    record["r_shaping"], record["r_clearance"],
                    record["r_expose"], record["r_split"],
                    record["r_disturb"], record["r_lost"],
                    record["r_outreach"], record["r_refused"],
                    record["r_near_edge"], record["r_pose"],
                    record["r_hit_target"], record["r_crowd"], int(done),
                    record["motion_errors"],
                    record["r_target_lost"], int(record["target_lost"]),
                    int(record["terminal"]), record["grasp_attempts"]])
                completed[0] = step     # stored in the buffer and the log

                if replay.n >= args.min_buffer and step > warmup:
                    for _ in range(args.utd):
                        agent.update(replay.sample(args.batch, device))

                if done:
                    kind = "demo" if demo_left > 0 else "policy"
                    if demo_left > 0:
                        demo_left -= 1
                    else:
                        recent.append((int(record["success"]),
                                       episode_length))
                    last = recent[-10:]
                    rate = ("%d/%d" % (sum(r[0] for r in last), len(last))
                            if last else "-")
                    print("[ep %3d] %-6s step=%4d return=%7.2f len=%2d "
                          "end=%-11s push=%d | last 10 policy episodes: "
                          "success %s"
                          % (episode_i, kind, step, episode_return,
                             episode_length, record["end"], pushes, rate))
                    episode_writer.writerow([
                        time.time(), step, episode_i, episode_return,
                        episode_length, int(record["success"]),
                        grasps, pushes, record["end"],
                        int(record["started_graspable"])])
                    episode_file.flush()
                    step_file.flush()
                    episode_return, episode_length = 0.0, 0
                    grasps, pushes = 0, 0
                    episode_i += 1
                    if step < step0 + args.steps:
                        obs, _ = runner.start_episode()

                if step % 50 == 0:
                    checkpoint_now(None)
            finished = True
        except StopTraining as stop:
            print("[train] %s" % stop)
        except KeyboardInterrupt:
            print("\n[train] interrupted")
        finally:
            # Whatever happened (finished, stopped by hand, server error),
            # what was learned so far is written before leaving.
            step_file.close()
            episode_file.close()
            checkpoint_now("finished" if finished else
                           "stopped after step %d" % completed[0])
            if not finished and args.save:
                print("[train] to go on from here, run the same command "
                      "with --resume 1 (and --steps set to the steps still "
                      "wanted)")
        print("[train] %s, %.1f min" % ("done" if finished else "ended early",
                                        (time.time() - start_time) / 60.0))
    finally:
        env.close()


if __name__ == "__main__":
    main()
