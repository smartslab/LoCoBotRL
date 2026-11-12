# -*- coding: utf-8 -*-
"""route2_perception.py

Runs on the TRAINING machine (Jetson, Python 3.8). Not on the robot.

Route 2 of the two-pipeline design
----------------------------------
One RGB-D frame feeds two separate paths:

  Route 1 (into the policy):  raw RGB + depth pixels -> phi_s -> actor/critic.
                              No centroids, no labels, no object identity.
  Route 2 (this file):        YOLO CPU boxes + NanoSAM GPU masks + depth -> centroids
                              in the base frame.

Route 2 does exactly two jobs, and nothing else:

  1. perceive(frame)        -> where the target and the obstacles are, so the
                               robot can execute a grasp or a push there.
  2. target_on_table(frame) -> is the target still on the table, used together
                               with the gripper state to decide success.

Route 2 NEVER chooses an action and NEVER enters the observation. The policy
picks the action from pixels alone, whether or not route 2 found anything.

Why route 2 must not paper over a failure
-----------------------------------------
When the target is occluded or buried, route 2 reports "no target". That is
the honest answer and it is the answer we want. A grasp then runs blind and
fails, the reward is the small step cost, and the policy learns that grasping
a buried target is useless. Over time it learns to push the occluding objects
away first, because only then does a grasp pay off.

If route 2 instead guessed a fallback position whenever recognition failed,
the target would never appear to vanish, and the policy would never get a
reason to clear the occlusion. The occlusion is not noise to be smoothed
away; it is the problem the policy is supposed to solve.

The cost of this choice: a false negative from the recognizer (the target is
actually visible but was missed in this frame) looks exactly like a real
occlusion. Control that in perception, by keeping the detector's confidence
threshold low and reading a settled frame, never by adding a fallback in the
action path.

Target identity across steps
----------------------------
Two modes, chosen by the training script:

  identity="color" (the training default). The task itself defines the
  target by colour: one red block hidden among green ones. So identity IS
  the label: the target is the detection whose hue label equals
  target_label ("red_object"). Position plays no part; the red block is the
  target wherever it sits or gets pushed to. When no detection carries the
  label, there is NO target this step, with no positional fallback, ever.
  Two blocks pressed together merge into one cluster whose mean hue is
  neither red nor green, so the red block genuinely disappears from the
  detections. That is not a perception failure to patch; it is the task.
  The policy is the thing that must fix it, by pushing the pile apart until
  a clean red cluster is visible, and the reward gap teaches exactly that:
  grasping the pile pays -0.05, separating then grasping pays +10.

  identity="track". Distance tracking for detectors that cannot tell
  objects apart (a single-class YOLO, for instance): target = detection
  nearest the last known position within track_max, else none. Kept for the
  segmentation backend later; measurements behind the 0.05 m track_max are
  in the git history of this comment: target self-motion measured 0 m over
  30 seeds (pushes through it are refused), centroid noise ~1.2 cm, minimum
  object spacing ~0.098 m, so 0.10 would adopt a neighbour when the target
  is occluded (caught by a test) and 0.05 cannot.

Either way, identity is only ever assigned among objects actually detected
in this frame. Nothing is invented when the target is hidden: an occluded
or merged-away target reports as missing, the grasp runs blind and fails,
and the policy gets the honest signal that it should clear the scene first.

Success detection strength per mode
-----------------------------------
In colour mode "the target left the table" is exact: is any red cluster
still on the table? The two mis-scoring cases of pure distance tracking
(grabbing a green block while the red one happens to be occluded reading as
success; a green block pushed into the red one's old spot reading as
failure) both disappear, because neither position nor occupancy is
consulted, only colour.

In track mode the check can only ask whether anything is detected near the
target's last known position, so those two cases exist and show up in the
logs. No extra rule is layered on to patch them; measure first.

Backends
--------
  "segmentation": custom YOLOv8 CPU boxes + NanoSAM GPU instance masks.
  "geometric" : no learned parts. Back-project depth, crop to the table,
                cluster, name each cluster by its dominant hue. Exists so the
                whole path runs today and can be tested against the mock
                server before the real models are wired in.

Both return the same detection dicts:

    {"label": str, "score": float, "centroid": [x, y, z] in base_link,
     "extent": [ex, ey, ez], "n_points": int}

Frame format (what {"cmd":"frame"} returns, after decode_frame):

    {"rgb":         HxWx3 uint8,
     "depth":       HxW uint16 millimetres, pixel-aligned with rgb,
     "intrinsics":  3x3 colour camera K,
     "cam_to_base": 4x4 transform base_link <- camera optical frame}

The centroids come out in the base frame because the server ships the colour
intrinsics and the TF with every frame. That is the same intrinsics/TF pair
the server's own point-cloud path uses, so there is a single source of
extrinsics and the two cannot drift apart.
"""

import colorsys

import numpy as np


# ---------------- frame decoding and geometry ----------------

def decode_array(packet):
    """Decode one zlib+base64 array packet from the env server."""
    import base64
    import zlib
    if packet.get("encoding") != "zlib+base64":
        raise ValueError("unsupported array encoding: %r"
                         % packet.get("encoding"))
    raw = zlib.decompress(base64.b64decode(packet["data"]))
    arr = np.frombuffer(raw, dtype=np.dtype(packet["dtype"]))
    return arr.reshape(packet["shape"]).copy()


def decode_frame(resp):
    """Turn a raw {"cmd":"frame"} response into numpy arrays.

    When the server sends a static-background mask (the arm at home,
    cables; recorded by calibrate_all.py), those pixels get depth 0 in
    frame["depth"], so every user of the frame ignores them. The untouched
    depth is kept in frame["depth_raw"] for anything that wants the picture
    as the camera saw it (the policy's observation)."""
    frame = {
        "rgb": decode_array(resp["rgb"]).astype(np.uint8),
        "depth": decode_array(resp["depth"]).astype(np.uint16),
        "intrinsics": np.asarray(resp["intrinsics"], dtype=np.float64),
        "cam_to_base": np.asarray(resp["cam_to_base"], dtype=np.float64),
        "background_match": resp.get("background_match"),
        "static_pixels": 0,
    }
    if "static_mask" in resp:
        mask = decode_array(resp["static_mask"]).astype(bool)
        if mask.shape == frame["depth"].shape:
            frame["depth_raw"] = frame["depth"]
            frame["depth"] = np.where(mask, 0, frame["depth"]).astype(
                np.uint16)
            frame["static_pixels"] = int(mask.sum())
    return frame


def backproject(depth_mm, K, cam_to_base, stride=4, rgb=None):
    """Depth image -> point cloud in the base frame.

    Returns (points Nx3, colours Nx3 or None). Uses every stride-th pixel.
    The depth is the aligned stream, so it is pixel-aligned with RGB and the
    colours come from the same pixel positions.
    """
    h, w = depth_mm.shape
    vs = np.arange(0, h, stride)
    us = np.arange(0, w, stride)
    uu, vv = np.meshgrid(us, vs)
    z = depth_mm[vv, uu].astype(np.float64) / 1000.0
    valid = z > 0.05
    uu, vv, z = uu[valid], vv[valid], z[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)
    pts_base = pts_cam.dot(cam_to_base.T)[:, :3]
    colors = None
    if rgb is not None:
        colors = rgb[vv, uu].astype(np.float64)
    return pts_base, colors


# Nominal hue (degrees) for each colour name, used to resolve a target
# label such as "red_object" into a hue for distance matching. Identity
# matching works on hue DISTANCE, not on these bucket names: a red block
# that drifts to 20 degrees under different lighting still matches "red"
# (distance 20), while green at 120 degrees never can. The names remain
# for logging only.
LABEL_HUES = {
    "red_object": 0.0, "orange_object": 30.0, "yellow_object": 55.0,
    "green_object": 120.0, "cyan_object": 185.0, "blue_object": 230.0,
    "purple_object": 300.0,
}
GRAY_SAT_MIN = 0.25    # below this saturation, hue is meaningless noise
                       # (the arm, cables, the table edge); such a
                       # detection can never be the colour target


def hue_distance(a, b):
    """Circular distance between two hues in degrees."""
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def detection_hue_match(detection, target_hue, tol):
    """Does this detection's colour match the target hue?

    Returns the distance when it matches, else None. Falls back to None
    when the detection carries no hue (a label-only backend); the caller
    then compares labels instead."""
    if target_hue is None or "hue" not in detection:
        return None
    if detection.get("sat", 1.0) < GRAY_SAT_MIN:
        return None
    d = hue_distance(detection["hue"], target_hue)
    return d if d <= tol else None


def colored(detections):
    """Detections whose hue means something (saturated, hue field present)."""
    return [d for d in detections
            if "hue" in d and d.get("sat", 1.0) >= GRAY_SAT_MIN]


def hue_outlier(detections, min_separation=60.0):
    """Index (into `detections`) of the colour outlier, or None.

    The task defines the target RELATIVELY: one block whose colour differs
    from all the others. Relative identity survives lighting drift, because
    drift shifts every hue together and "which one is different" does not
    change - unlike any absolute rule ("hue near 0 degrees"), which is what
    kept breaking on the white table.

    Rule: among colour-carrying detections, take the one whose nearest hue
    neighbour is farthest away; it counts only if that separation exceeds
    min_separation (red-vs-green is ~120, so 60 leaves a wide margin).

    Needs at least THREE coloured detections. With exactly two, "the odd
    one out" is undefined - the two are simply far from each other, and
    the rule would pick one arbitrarily (a merged pile next to a lone
    obstacle got picked this way in a test). Two same-coloured obstacles
    are what anchors "the others"; below three, the caller falls back to
    the episode's remembered target hue, which resolves the pair case
    correctly."""
    cands = [(i, d) for i, d in enumerate(detections)
             if "hue" in d and d.get("sat", 1.0) >= GRAY_SAT_MIN]
    if len(cands) < 3:
        return None
    best_index, best_sep = None, -1.0
    for i, di in cands:
        sep = min(hue_distance(di["hue"], dj["hue"])
                  for j, dj in cands if j != i)
        if sep > best_sep:
            best_index, best_sep = i, sep
    if best_sep < min_separation:
        return None
    return best_index


def hue_label(rgb_mean):
    """Name a colour by its hue. Used only by the geometric backend."""
    r, g, b = (float(c) / 255.0 for c in rgb_mean)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if s < 0.25 or v < 0.15:
        return "gray_object"
    deg = h * 360.0
    names = [(15, "red"), (45, "orange"), (70, "yellow"), (170, "green"),
             (200, "cyan"), (260, "blue"), (345, "purple"), (360, "red")]
    for upper, name in names:
        if deg < upper:
            return name + "_object"
    return "red_object"


def cluster_points(pts, voxel, min_points):
    """Voxel connected components (26-neighbourhood). Returns a list of
    integer index arrays, one per cluster, largest first."""
    if len(pts) == 0:
        return []
    keys = np.floor(pts / voxel).astype(np.int64)
    voxel_ids = {}
    for i, key in enumerate(map(tuple, keys)):
        voxel_ids.setdefault(key, []).append(i)
    parent = {key: key for key in voxel_ids}

    def find(key):
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != root:
            parent[key], key = root, parent[key]
        return root

    offsets = [(dx, dy, dz)
               for dx in (-1, 0, 1) for dy in (-1, 0, 1)
               for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
    for key in voxel_ids:
        for off in offsets:
            neighbor = (key[0] + off[0], key[1] + off[1], key[2] + off[2])
            if neighbor in voxel_ids:
                ra, rb = find(key), find(neighbor)
                if ra != rb:
                    parent[rb] = ra
    groups = {}
    for key, indices in voxel_ids.items():
        groups.setdefault(find(key), []).extend(indices)
    clusters = [np.asarray(v) for v in groups.values()
                if len(v) >= min_points]
    clusters.sort(key=len, reverse=True)
    return clusters


def principal_axis(xy):
    """Orientation of a footprint: (yaw, elongation).

    yaw is the angle (radians, base frame, in [-pi/2, pi/2)) of the
    footprint's long axis from a 2-D PCA of its points. elongation is the
    ratio of the two standard deviations (1 = round/square, larger = more
    elongated). A parallel-jaw grasp closes across the short side, i.e.
    perpendicular to yaw. For a square footprint yaw is arbitrary and any
    roll works, which elongation near 1 tells the caller."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 3:
        return 0.0, 1.0
    centered = xy - xy.mean(axis=0)
    cov = centered.T.dot(centered) / float(len(xy))
    values, vectors = np.linalg.eigh(cov)
    major = vectors[:, int(np.argmax(values))]
    yaw = float(np.arctan2(major[1], major[0]))
    yaw = (yaw + np.pi / 2.0) % np.pi - np.pi / 2.0
    small = max(float(values.min()), 1e-9)
    elongation = float(np.sqrt(values.max() / small))
    return yaw, elongation


def convex_hull(xy):
    """Andrew's monotone chain. Returns hull points in order."""
    pts = sorted(set(map(tuple, np.asarray(xy, dtype=float).tolist())))
    if len(pts) <= 2:
        return np.asarray(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def min_area_rect(xy):
    """Orientation of a footprint from its minimum-area bounding rectangle
    (rotating calipers over the convex hull): (yaw, long_side, short_side).

    yaw is the angle of the rectangle's LONG side, in [-pi/2, pi/2). Unlike
    a PCA axis this is meaningful for a square: a square has no long axis
    but it does have edges, and a parallel-jaw gripper must close across a
    FACE. A 3 cm cube presents 3.0 cm across a face and 4.2 cm across the
    diagonal; with a 4.5 cm jaw that is the difference between a grasp and
    a finger on the edge. For a square the two sides are equal and either
    edge angle works, which also makes the result immune to the 90-degree
    ambiguity of the gripper's roll convention."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 3:
        return 0.0, 0.0, 0.0
    hull = convex_hull(xy)
    if len(hull) < 3:
        return 0.0, 0.0, 0.0
    best = None
    for i in range(len(hull)):
        edge = hull[(i + 1) % len(hull)] - hull[i]
        angle = float(np.arctan2(edge[1], edge[0]))
        c, s_ = np.cos(-angle), np.sin(-angle)
        rot = np.array([[c, -s_], [s_, c]])
        proj = xy.dot(rot.T)
        w = float(proj[:, 0].max() - proj[:, 0].min())
        h = float(proj[:, 1].max() - proj[:, 1].min())
        area = w * h
        if best is None or area < best[0]:
            best = (area, angle, w, h)
    _, angle, w, h = best
    if w >= h:
        long_side, short_side, yaw = w, h, angle
    else:
        long_side, short_side, yaw = h, w, angle + np.pi / 2.0
    yaw = (yaw + np.pi / 2.0) % np.pi - np.pi / 2.0
    return float(yaw), long_side, short_side


# ---------------- configuration ----------------

class PerceptionConfig(object):
    """Geometry route 2 reasons over.

    The workspace box is grown by ws_margin on each side relative to the
    server's RL crop box. An object pushed just outside the crop box is still
    physically on the table, and route 2 must not report it as gone: "gone"
    is supposed to mean grasped or knocked off, not merely out of the crop.
    """

    def __init__(self, ws_x=(0.28, 0.60), ws_y=(-0.25, 0.25),
                 table_z=0.225, z_max=0.42, ws_margin=0.08,
                 table_cut=0.015, voxel=0.02, min_points=25):
        self.ws_x = (ws_x[0] - ws_margin, ws_x[1] + ws_margin)
        self.ws_y = (ws_y[0] - ws_margin, ws_y[1] + ws_margin)
        # the server's own box (perception looks a little wider)
        self.box_x, self.box_y = tuple(ws_x), tuple(ws_y)
        self.table_z = float(table_z)
        self.z_max = float(z_max)
        self.table_cut = float(table_cut)
        self.voxel = float(voxel)
        self.min_points = int(min_points)

    def crop_mask(self, pts):
        return ((pts[:, 0] > self.ws_x[0]) & (pts[:, 0] < self.ws_x[1]) &
                (pts[:, 1] > self.ws_y[0]) & (pts[:, 1] < self.ws_y[1]) &
                (pts[:, 2] > self.table_z + self.table_cut) &
                (pts[:, 2] < self.z_max))


# ---------------- backends ----------------

def point_hsv(colors):
    """Per-point hue (degrees), saturation and value of an (N, 3) uint8/
    float RGB array, vectorised."""
    rgb = np.asarray(colors, dtype=float) / 255.0
    mx, mn = rgb.max(axis=1), rgb.min(axis=1)
    delta = mx - mn
    sat = np.where(mx > 1e-6, delta / np.maximum(mx, 1e-6), 0.0)
    hue = np.zeros(len(rgb))
    nz = delta > 1e-6
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    rmax = nz & (mx == r)
    gmax = nz & (mx == g) & ~rmax
    bmax = nz & ~rmax & ~gmax
    hue[rmax] = (60.0 * ((g[rmax] - b[rmax]) / delta[rmax])) % 360.0
    hue[gmax] = 60.0 * ((b[gmax] - r[gmax]) / delta[gmax]) + 120.0
    hue[bmax] = 60.0 * ((r[bmax] - g[bmax]) / delta[bmax]) + 240.0
    return hue, sat, mx


class GeometricBackend(object):
    """No learned parts: crop, drop the table by height, cluster by voxel
    connectivity, name each cluster by its dominant hue.

    Blocks that touch become ONE cluster by connectivity. When a cluster
    contains a clear patch of the target's colour (split_hue, red by
    default) next to points of other colours, it is split by colour: the
    target-coloured points become their own detection and the rest is
    clustered again by connectivity (the blocks on either side of the red
    one then come apart). Without this, a red block pressed between two
    green ones was "not seen", and every reward term that needs the red
    block's position was blind in exactly the scene the task starts from.
    split_hue=None turns the split off."""

    name = "geometric"

    def __init__(self, config, split_hue=0.0, split_tol=25.0,
                 split_min_sat=0.35):
        self.config = config
        self.split_hue = split_hue
        self.split_tol = float(split_tol)
        self.split_min_sat = float(split_min_sat)

    def _target_coloured(self, colors):
        hue, sat, val = point_hsv(colors)
        diff = np.abs((hue - float(self.split_hue) + 180.0) % 360.0 - 180.0)
        return ((diff <= self.split_tol) & (sat >= self.split_min_sat)
                & (val >= 0.15))

    def _split(self, cluster, colors):
        """Index arrays (into cluster) of the parts of one cluster."""
        cfg = self.config
        if self.split_hue is None or len(cluster) < 2 * cfg.min_points:
            return [np.arange(len(cluster))]
        mask = self._target_coloured(colors)
        n_red, n_rest = int(mask.sum()), int((~mask).sum())
        if (n_red < max(cfg.min_points, 0.10 * len(cluster))
                or n_rest < max(cfg.min_points, 0.10 * len(cluster))):
            return [np.arange(len(cluster))]
        red_idx = np.nonzero(mask)[0]
        rest_idx = np.nonzero(~mask)[0]
        parts = []
        for sub in cluster_points(cluster[red_idx], cfg.voxel,
                                  cfg.min_points):
            parts.append(red_idx[sub])
        for sub in cluster_points(cluster[rest_idx], cfg.voxel,
                                  cfg.min_points):
            parts.append(rest_idx[sub])
        return parts or [np.arange(len(cluster))]

    def detect(self, frame):
        cfg = self.config
        pts, colors = backproject(
            frame["depth"], frame["intrinsics"], frame["cam_to_base"],
            stride=4, rgb=frame["rgb"])
        keep = cfg.crop_mask(pts)
        pts, colors = pts[keep], colors[keep]
        detections = []
        for idx0 in cluster_points(pts, cfg.voxel, cfg.min_points):
            for part in self._split(pts[idx0], colors[idx0]):
                idx = idx0[part]
                detection = self._detection(pts[idx], colors[idx])
                if detection is not None:
                    if len(part) < len(idx0):
                        detection["split_from_pile"] = True
                    detections.append(detection)
        return detections

    @staticmethod
    def _detection(cluster, colors):
        centroid = cluster.mean(axis=0)
        extent = cluster.max(axis=0) - cluster.min(axis=0)
        # Drop blobs that cannot be table objects (walls, the bin, merged
        # junk). No lower bound on the z extent: an overhead view of a
        # flat top can legitimately have almost no height spread.
        if not (0.005 < extent[0] < 0.30
                and 0.005 < extent[1] < 0.30
                and extent[2] < 0.30):
            return None
        rgb_mean = colors.mean(axis=0)
        r, g, b = (float(c) / 255.0 for c in rgb_mean)
        h, sat, _ = colorsys.rgb_to_hsv(r, g, b)
        yaw, long_side, short_side = min_area_rect(cluster[:, :2])
        elongation = (long_side / short_side if short_side > 1e-3
                      else 1.0)
        # The two ends of the footprint along its long axis. For a pile
        # (several blocks merged into one cluster) the centroid lies
        # between blocks and a pick there closes on nothing; picking at
        # an END takes the outermost block off the pile.
        axis = np.array([np.cos(yaw), np.sin(yaw)])
        along = (cluster[:, :2] - centroid[:2]).dot(axis)
        ends = [(centroid[:2] + axis * float(along.min())).tolist(),
                (centroid[:2] + axis * float(along.max())).tolist()]
        # 2D point cloud, downsampled to keep the JSON payload small.
        # The downstream clearance check only reads these points to
        # find the one closest to the target centre; 60 points cover
        # the boundary of a block-sized footprint well enough. A
        # deterministic subsample keeps the same points frame-to-frame
        # when the cluster is stable, so clearance does not jitter.
        points_xy = cluster[:, :2]
        if len(points_xy) > 60:
            step = int(np.ceil(len(points_xy) / 60.0))
            points_xy = points_xy[::step][:60]
        return {
            "label": hue_label(rgb_mean),
            "hue": h * 360.0,
            "sat": sat,
            "mean_rgb": [float(v) for v in rgb_mean],
            "score": 1.0,
            "centroid": centroid.tolist(),
            "extent": extent.tolist(),
            # orientation of the footprint's long side (min-area
            # rectangle), for finger alignment; the fingers close
            # across the short side
            "yaw": yaw,
            "elongation": elongation,
            "rect_long": long_side,
            "rect_short": short_side,
            "ends": ends,
            "n_points": int(len(cluster)),
            "points_xy": points_xy.tolist(),
        }


class SegmentationBackend(object):
    """Custom YOLOv8 detector on CPU -> NanoSAM masks on GPU -> 3D geometry.

    Environment variables:
      YOLO_WEIGHTS: trained block detector .pt
      YOLO_CONF: detection threshold, default 0.25
      YOLO_IMGSZ: inference size, default 1280
      NANOSAM_ENCODER_ENGINE / NANOSAM_DECODER_ENGINE: TensorRT paths
      SAM_MASK_THRESHOLD: NanoSAM mask threshold, default 0.95

    The heavy NanoSAM predictor is loaded once. Each frame's masks are
    independent: no first-frame IDs, no CuPy, no DINOv2. Both the training
    caller and CLI use this same backend, with no robot motion commands here.
    """

    name = "segmentation"

    def __init__(self, config, model_dir=None):
        import os
        from ultralytics import YOLO
        from nanosam.utils.predictor import Predictor

        self.config = config
        self.model_dir = model_dir
        self.last_instances = []
        self.last_accepted = []
        self.conf = float(os.environ.get("YOLO_CONF", "0.5"))
        self.imgsz = int(os.environ.get("YOLO_IMGSZ", "1280"))
        self.mask_threshold = float(os.environ.get("SAM_MASK_THRESHOLD", "0.95"))
        self.min_mask_pixels = int(os.environ.get("PERCEPTION_MIN_MASK_PIXELS", "25"))
        self.weights = os.environ.get("YOLO_WEIGHTS", "/home/smartslab_jetson_xavier/runs/block_training/blocks_final/weights/best.pt")
        encoder = os.environ.get("NANOSAM_ENCODER_ENGINE",
            "/home/smartslab_jetson_xavier/nanosam/data/resnet18_image_encoder.engine")
        decoder = os.environ.get("NANOSAM_DECODER_ENGINE",
            "/home/smartslab_jetson_xavier/nanosam/data/mobile_sam_mask_decoder.engine")
        if not os.path.isfile(self.weights):
            raise FileNotFoundError("Custom YOLO weights not found: " + self.weights)
        for path in (encoder, decoder):
            if not os.path.isfile(path):
                raise FileNotFoundError("NanoSAM TensorRT engine not found: " + path)
        if not (0 < self.conf <= 1 and 0 < self.mask_threshold < 1):
            raise ValueError("YOLO_CONF and SAM_MASK_THRESHOLD must be in (0,1)")
        self.detector = YOLO(self.weights)
        # Explicit device='cpu' below: do NOT move this model to CUDA.
        self.segmenter = Predictor(
            image_encoder_engine=encoder,
            mask_decoder_engine=decoder)
        print("[route2] YOLO CPU weights=%s imgsz=%d; NanoSAM GPU" %
              (self.weights, self.imgsz))

    def _sam_mask(self, raw, h, w):
        """Convert the lab NanoSAM prediction to one HxW boolean mask."""
        import cv2
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        if hasattr(raw, "detach"):
            raw = raw.detach().float().cpu().numpy()
        raw = np.asarray(raw)
        while raw.ndim > 2:
            raw = raw[0]
        if raw.shape != (h, w):
            raw = cv2.resize(raw.astype(np.float32), (w, h),
                             interpolation=cv2.INTER_LINEAR)
        return raw > self.mask_threshold

    def detect(self, frame):
        """Return the existing Route2Perception detection-dict format."""
        import cv2
        from PIL import Image as PILImage
        rgb = np.ascontiguousarray(frame["rgb"], dtype=np.uint8)
        depth = np.asarray(frame["depth"], dtype=np.uint16)
        K = np.asarray(frame["intrinsics"], dtype=np.float64)
        T = np.asarray(frame["cam_to_base"], dtype=np.float64)
        h, w = depth.shape
        if rgb.shape != (h, w, 3):
            raise ValueError("RGB and aligned depth sizes disagree")
        self.last_instances = []
        self.last_accepted = []
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        results = self.detector.predict(
            source=bgr, imgsz=self.imgsz, conf=self.conf,
            device="cpu", verbose=False)
        detections = []
        if not results or len(results[0].boxes) == 0:
            return detections

        # Run image encoder only once for the current image, not once per box.
        self.segmenter.set_image(PILImage.fromarray(rgb))
        names = results[0].names
        for box in results[0].boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().tolist()]
            x1, x2 = sorted((float(np.clip(x1, 0, w - 1)),
                             float(np.clip(x2, 0, w - 1))))
            y1, y2 = sorted((float(np.clip(y1, 0, h - 1)),
                             float(np.clip(y2, 0, h - 1))))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            cls_id = int(box.cls.item())
            detector_label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            score = float(box.conf.item())
            # Matches the NanoSAM Predictor API in the supplied lab script:
            # top-left corner type 2 and bottom-right corner type 3.
            points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
            point_labels = np.array([2, 3], dtype=np.int32)
            masks, _, _ = self.segmenter.predict(points, point_labels)
            mask = self._sam_mask(masks, h, w)
            # Aligned depth must be valid for 3D reconstruction.
            mask &= (depth > 0)
            instance = {"label": str(detector_label), "score": score,
                        "box": [x1, y1, x2, y2], "mask": mask,
                        "reject_reason": "unprocessed"}
            idx = len(self.last_instances)
            self.last_instances.append(instance)
            if int(mask.sum()) < self.min_mask_pixels:
                instance["reject_reason"] = "stuff (%d)" % int(mask.sum())
                continue
            # Project only pixels inside this SAM mask, not the whole box.
            masked_depth = np.where(mask, depth, 0)
            pts, colors = backproject(masked_depth, K, T, stride=1, rgb=rgb)
            keep = self.config.crop_mask(pts)
            pts, colors = pts[keep], colors[keep]
            if len(pts) < self.config.min_points:
                instance["reject_reason"] = "stuff (%d)" % len(pts)
                continue
            # Existing geometry function computes hue, footprint yaw, extents,
            # point samples, and pick candidates for compatibility with RL.
            entry = GeometricBackend._detection(pts, colors)
            if entry is None:
                extent = np.ptp(pts, axis=0)
                instance["reject_reason"] = "3D extent rejected (m): %s" % np.round(extent, 3).tolist()
                continue
            instance["reject_reason"] = None
            instance["mean_rgb"] = entry["mean_rgb"]
            entry["score"] = score
            entry["detector_label"] = str(detector_label)
            entry["instance_index"] = idx
            detections.append(entry)
            self.last_accepted.append(idx)
        return detections


def make_backend(name, config, model_dir=None):
    if name == "geometric":
        return GeometricBackend(config)
    if name == "segmentation":
        return SegmentationBackend(config, model_dir)
    raise ValueError("unknown perception backend: %r" % name)


# ---------------- route 2 front end ----------------

class Route2Perception(object):
    """Turns detections into the two things route 2 owes the training loop.

    identity="color": the target is the detection labelled target_label
    ("red_object"). The task defines the target by colour, so colour IS the
    identity; position never enters it. No detection with that label means
    no target, with no fallback of any kind - see the module docstring for
    why that is the point and not a defect.

    identity="track": distance tracking for label-less detectors (kept for
    the single-class segmentation backend). track_max must stay well below
    the object spacing; 0.05 m, not 0.10.

    anchor_xy is only a scene-layout check at reset in colour mode: if the
    red block is found far from it, a warning is printed and nothing else
    changes. In track mode it designates the initial target.
    """

    def __init__(self, backend, identity="color", target_label="red_object",
                 anchor_xy=(0.40, 0.03), track_max=0.05, max_obstacles=4,
                 arm_radius=0.10, occlusion_radius=0.15, hue_tol=35.0,
                 target_max_extent=0.085, hover_max=0.12,
                 min_elongation=1.6):
        if identity not in ("color", "track"):
            raise ValueError("identity must be 'color' or 'track'")
        if identity == "color" and not target_label:
            raise ValueError("colour identity needs a target_label")
        self.backend = backend
        self.identity = identity
        self.target_label = target_label
        # Colour identity is RELATIVE, the way the task defines it: the
        # target is the one block whose colour differs from all the
        # others. Relative identity survives lighting drift (drift moves
        # every hue together, "which one is different" stays put), which
        # is exactly where absolute rules ("hue near 0") kept breaking.
        #
        # Per frame: the hue outlier is the target. When there are not
        # two coloured detections to compare (obstacles pushed away, or
        # everything else merged), fall back to distance from the
        # EPISODE'S OWN remembered target hue - recorded the first time
        # the target is identified this episode, so the reference is
        # self-calibrating, never a hard-coded colour. target_label seeds
        # that reference before the first sighting (e.g. "red_object" ->
        # 0 deg) and is otherwise just a log name; hue_tol bounds the
        # fallback match.
        self.target_hue = LABEL_HUES.get(target_label)
        self.episode_hue = None
        self.hue_tol = float(hue_tol)
        # Size gate on identity: a candidate whose XY footprint exceeds a
        # single block cannot be the target, whatever its colour. Field
        # data made this necessary: a red+green pile measuring
        # 0.095 x 0.120 m read as a reddish hue and was adopted as the
        # target by the hue fallback.
        #
        # IMPORTANT: this must sit between the longest single block and the
        # smallest merged pile. With 3 cm cubes the default 0.085 is very
        # safe. If the target block is long (a 9 cm bar lying down measures
        # 0.09), the default would reject the target itself - raise
        # --target_max_extent above the block's longest side, and keep the
        # blocks far enough apart that a pile is still bigger than that.
        # Consequence to know: after a failed grasp shoves the target into
        # a pile, "target on table" reads False (the pile is not the
        # target), so the two success conditions can in principle both
        # pass while the target sits inside a pile - the gripper must ALSO
        # report a hold for that, which a missed grasp rarely does.
        self.target_max_extent = float(target_max_extent)
        # Anything whose centroid sits more than hover_max above the table
        # is not resting on it: a cable, a mount, a piece of the robot, the
        # wrist. Seen on the robot as a grey blob 0.145 m up and a blue one
        # 0.186 m up, both within the workspace box; without this rule they
        # would have been offered to the policy as obstacles to push. A
        # 3 cm block's centroid is ~0.03 above the table; raise this for
        # tall objects (a 19 cm YCB bottle has its centroid near +0.10).
        self.hover_max = float(hover_max)
        # A parallel-jaw grasp only has a preferred angle when the footprint
        # actually has a long side. Below this length-to-width ratio the
        # PCA direction is noise: a 3 cm cube measured 2.8 x 2.9 cm on the
        # robot and reported a confident-looking 55 degrees, which then
        # rotated the gripper by a random amount on every attempt. Below the
        # threshold the yaw is reported as None and the gripper keeps its
        # canonical roll, with the policy's roll offset on top.
        self.min_elongation = float(min_elongation)
        self.table_z = float(getattr(backend, "config", None).table_z) \
            if getattr(backend, "config", None) is not None else None
        self.anchor = np.asarray(anchor_xy, dtype=float)
        self.track_max = float(track_max)
        self.max_obstacles = int(max_obstacles)
        # Detections within arm_radius (XY) of the end effector are the
        # robot's own fingers and are dropped. The wrist and forearm blob
        # is caught by the hover rule below (it sits 0.11-0.15 m up), so
        # this radius only has to cover the fingertips, a few centimetres.
        # It was 0.18 before the hover rule existed, and at that size it
        # deleted a real block 0.13 m from the parked gripper.
        self.arm_radius = float(arm_radius)
        # Self-occlusion test: when the target is not detected AND the end
        # effector sits within this distance of the camera-to-cached-target
        # line of sight, the arm is judged to be blocking the view and the
        # cached position is used for execution. When the arm is NOT on
        # that line, a missing target is missing for real (buried, merged,
        # pushed away) and nothing is invented - that case belongs to the
        # policy.
        self.occlusion_radius = float(occlusion_radius)
        # Last position the target was seen at (base frame). In colour mode
        # this is logging and obstacle-ordering only, never identity.
        self.target_xy = None
        # Cached full target entry from the last frame it was actually
        # seen in. Cleared at episode start. Only ever consulted under the
        # self-occlusion test above.
        self.cached_target = None
        self.n_objects_start = 0

    def _extent_ok(self, detection):
        extent = detection.get("extent")
        if extent is None:
            return True
        return (float(extent[0]) <= self.target_max_extent
                and float(extent[1]) <= self.target_max_extent)

    def _drop_arm_blobs(self, detections, ee_xyz):
        """Remove detections that are the arm itself, and anything hovering
        too high above the table to be a resting object."""
        kept = detections
        if self.table_z is not None:
            kept = [d for d in kept
                    if d["centroid"][2] - self.table_z <= self.hover_max]
        if ee_xyz is not None:
            ee = np.asarray(ee_xyz, dtype=float).reshape(3)[:2]
            kept = [d for d in kept
                    if np.linalg.norm(np.asarray(d["centroid"][:2]) - ee)
                    > self.arm_radius]
        return kept, len(detections) - len(kept)

    @staticmethod
    def _point_segment_distance(point, start, end):
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom < 1e-9:
            return float(np.linalg.norm(point - start))
        t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
        return float(np.linalg.norm(point - (start + t * segment)))

    def _self_occluded(self, frame, ee_xyz):
        """Is the arm plausibly between the camera and the cached target?"""
        if self.cached_target is None or ee_xyz is None:
            return False
        cam = np.asarray(frame["cam_to_base"], dtype=float)[:3, 3]
        tgt = np.asarray(self.cached_target["centroid"], dtype=float)
        ee = np.asarray(ee_xyz, dtype=float).reshape(3)
        return self._point_segment_distance(ee, cam, tgt) \
            < self.occlusion_radius

    # -- helpers --

    def _pick_nearest(self, detections, reference, max_distance):
        """Index of the detection nearest `reference`, or None when the
        nearest one is further away than max_distance."""
        if not detections:
            return None
        distances = [
            float(np.linalg.norm(np.asarray(d["centroid"][:2]) - reference))
            for d in detections]
        index = int(np.argmin(distances))
        if max_distance is not None and distances[index] > max_distance:
            return None
        return index

    def _pick_by_label(self, detections):
        """For red-target scenes, choose the most red *accepted* 3D object.

        RGB is averaged over the SAM-masked, workspace-filtered depth points,
        rather than a bounding box. A relative redness score orders candidates,
        but a minimum red-dominance gate prevents choosing a green object when
        the red target is occluded. No position fallback is used.
        """
        if self.target_label == "red_object":
            import os
            min_ratio = float(os.environ.get("RED_DOMINANCE_RATIO", "1.25"))
            min_red = float(os.environ.get("RED_MIN_CHANNEL", "45"))
            candidates = []
            for i, d in enumerate(detections):
                if not self._extent_ok(d):
                    d["target_reject_reason"] = "target footprint exceeds maximum"
                    continue
                rgb = d.get("mean_rgb")
                if rgb is None:
                    d["target_reject_reason"] = "missing mean RGB"
                    continue
                r, g, b = [float(x) for x in rgb]
                if r < min_red or r < min_ratio * max(g, b):
                    d["target_reject_reason"] = "not sufficiently red"
                    continue
                # Higher normalized redness means closer to pure red [1,0,0]
                # after removing overall brightness differences.
                chroma = np.array([r, g, b], dtype=float) / max(r + g + b, 1.0)
                red_distance = float(np.linalg.norm(chroma - [1.0, 0.0, 0.0]))
                d["red_distance"] = red_distance
                candidates.append((red_distance, -r, i))
            if not candidates:
                return None
            index = min(candidates)[2]
            self.episode_hue = float(detections[index].get("hue", 0.0))
            return index

        # Preserve the original behavior for other target colors.
        index = hue_outlier(detections)
        if index is not None and not self._extent_ok(detections[index]):
            index = None
        if index is not None:
            self.episode_hue = float(detections[index]["hue"])
            return index
        reference = (self.episode_hue if self.episode_hue is not None
                     else self.target_hue)
        best, best_key = None, None
        for i, detection in enumerate(detections):
            if not self._extent_ok(detection):
                continue
            distance = detection_hue_match(
                detection, reference, self.hue_tol)
            if distance is not None:
                key = (0, distance, -detection["n_points"])
            elif ("hue" not in detection
                  and detection["label"] == self.target_label):
                key = (1, 0.0, -detection["n_points"])
            else:
                continue
            if best_key is None or key < best_key:
                best, best_key = i, key
        return best

    def _pick_target(self, detections, at_reset=False):
        if self.identity == "color":
            return self._pick_by_label(detections)
        if at_reset or self.target_xy is None:
            # Track mode designation: unbounded nearest-to-anchor at reset.
            return self._pick_nearest(detections, self.anchor, None)
        return self._pick_nearest(detections, self.target_xy, self.track_max)

    def _pick_point(self, detection, entry, single_block=0.05):
        """Where a pick-and-place should close on this object. A single
        block: its centroid. A pile (long side clearly bigger than one
        block): the outermost block, i.e. the end of the footprint farther
        from the target's last known position (or the anchor), pulled in by
        half a block so the fingers centre on that block."""
        long_side = detection.get("rect_long")
        ends = detection.get("ends")
        if long_side is None or ends is None or long_side <= single_block:
            return list(entry["centroid"])
        reference = (self.target_xy if self.target_xy is not None
                     else self.anchor)
        far = max(ends, key=lambda e: float(np.linalg.norm(
            np.asarray(e) - reference)))
        far = np.asarray(far, dtype=float)
        centre = np.asarray(entry["centroid"][:2], dtype=float)
        toward_centre = centre - far
        norm = float(np.linalg.norm(toward_centre))
        if norm > 1e-6:
            far = far + toward_centre / norm * 0.015
        return [float(far[0]), float(far[1]), float(entry["centroid"][2])]

    def _split(self, detections, target_index):
        """Split detections into the target entry and the obstacle list, in
        the format the server expects."""
        target, obstacles = None, []
        for index, detection in enumerate(detections):
            entry = {"centroid": detection["centroid"],
                     "extent": detection["extent"],
                     "n": detection["n_points"],
                     "points_xy": detection.get("points_xy", [])}
            if "yaw" in detection:
                # The min-area-rectangle angle is an EDGE direction and is
                # valid for squares as well, so it is always passed on.
                # (A PCA axis was used before and was suppressed for
                # squares; that left the gripper at a fixed angle, and a
                # cube dropped at 45 degrees then presented its 4.2 cm
                # diagonal to a 4.5 cm jaw.)
                entry["yaw"] = detection["yaw"]
                entry["elongation"] = detection.get("elongation", 1.0)
                entry["rect_long"] = detection.get("rect_long")
                entry["rect_short"] = detection.get("rect_short")
            if index == target_index:
                target = entry
            else:
                entry["pick_centroid"] = self._pick_point(detection, entry)
                obstacles.append(entry)
        # Nearest obstacles first so the policy's obstacle selector indexes a
        # stable ordering. With no target, order by the last place the target
        # was seen (the pile that swallowed it is usually right there), else
        # the anchor.
        if target is not None:
            reference = np.asarray(target["centroid"][:2])
        elif self.target_xy is not None:
            reference = self.target_xy
        else:
            reference = self.anchor
        obstacles.sort(key=lambda entry: float(np.linalg.norm(
            np.asarray(entry["centroid"][:2]) - reference)))
        return target, obstacles[:self.max_obstacles]

    # -- episode bookkeeping --

    def start_episode(self, frame, ee_xyz=None):
        """Find the target and record what is on the table."""
        detections = self.backend.detect(frame)
        detections, arm_blobs = self._drop_arm_blobs(detections, ee_xyz)
        self.n_objects_start = len(detections)
        self.cached_target = None
        self.episode_hue = None
        index = self._pick_target(detections, at_reset=True)
        self.target_xy = (
            np.asarray(detections[index]["centroid"][:2], dtype=float)
            if index is not None else None)
        target, obstacles = self._split(detections, index)
        if target is not None:
            self.cached_target = dict(target)
        anchor_distance = (
            float(np.linalg.norm(self.target_xy - self.anchor))
            if self.target_xy is not None else None)
        return {
            "n_objects": len(detections),
            "target_found": target is not None,
            "target_xy": (None if self.target_xy is None
                          else self.target_xy.tolist()),
            # Layout check only: a red block far from the anchor usually
            # means the scene was arranged differently than planned. In
            # colour mode the target may legitimately start hidden inside a
            # pile, in which case target_found is False here and that is
            # fine - the episode starts with the policy having to dig it out.
            "anchor_distance": anchor_distance,
            "target": target,
            "obstacles": obstacles,
            "arm_blobs": arm_blobs,
            "detections": detections,
        }

    # -- job 1: where to act --

    def perceive(self, frame, ee_xyz=None):
        """Locate the target and the obstacles for primitive execution.

        Three outcomes, reported in target_source:
          "seen"  - the target is detected in this frame; the cache is
                    refreshed and the fresh position is used.
          "cache" - the target is not detected, but the end effector sits on
                    the camera's line of sight to the cached position, so
                    the arm is judged to be blocking the view. The cached
                    position is used, the way a person keeps reaching for a
                    cup their own hand is hiding.
          "none"  - the target is not detected and the arm is not in the
                    way. It is buried, merged into a pile, or gone. Nothing
                    is invented; a grasp now runs blind and fails, which is
                    the signal that teaches the policy to clear the scene.
        """
        detections = self.backend.detect(frame)
        detections, arm_blobs = self._drop_arm_blobs(detections, ee_xyz)
        index = self._pick_target(detections)
        if index is not None:
            self.target_xy = np.asarray(
                detections[index]["centroid"][:2], dtype=float)
        target, obstacles = self._split(detections, index)
        source = "seen" if target is not None else "none"
        if target is not None:
            self.cached_target = dict(target)
        elif self._self_occluded(frame, ee_xyz):
            target = dict(self.cached_target)
            source = "cache"
        return {
            "target": target,
            "obstacles": obstacles,
            "target_found": target is not None,
            "target_source": source,
            "arm_blobs": arm_blobs,
            "n_objects": len(detections),
            "target_xy": (None if self.target_xy is None
                          else self.target_xy.tolist()),
            "detections": detections,
        }

    # -- job 2: did the target leave the table --

    def target_on_table(self, frame, exclude_xyz=None, exclude_radius=0.08):
        """Is the target still lying on the table?

        Used with the gripper state to decide success: a grasp counts only
        when the gripper reports holding something AND the target is no
        longer on the table.

        Colour mode asks the exact question: is any red cluster still on
        the table? Track mode can only ask whether anything is detected
        near the target's last known position.

        exclude_xyz: pass the end-effector position when the gripper
        reports a hold. Right after a successful grasp the object is still
        in view, hanging under the gripper, and must not be counted as on
        the table. The test is horizontal distance only, since the held
        object sits directly below the gripper while anything left on the
        table had to be far enough away for the grasp to reach in.
        """
        detections = self.backend.detect(frame)
        if exclude_xyz is not None:
            center = np.asarray(exclude_xyz, dtype=float).reshape(3)[:2]
            detections = [
                d for d in detections
                if np.linalg.norm(np.asarray(d["centroid"][:2]) - center)
                > exclude_radius]
        if self.identity == "color":
            on_table = self._pick_by_label(detections) is not None
        else:
            on_table = (self.target_xy is not None
                        and self._pick_nearest(
                            detections, self.target_xy,
                            self.track_max) is not None)
        return {
            "target_on_table": on_table,
            "n_objects": len(detections),
            "n_objects_start": self.n_objects_start,
            # One fewer object than at the start is what a removed object
            # looks like. Logged, not acted on.
            "objects_removed": self.n_objects_start - len(detections),
            "detections": detections,
        }


# ---------------- graspability rule (geometry only) ----------------

def xy_radius(extent):
    """Half the footprint diagonal: a conservative circle around an object."""
    extent = np.asarray(extent, dtype=float)
    return 0.5 * float(np.hypot(extent[0], extent[1]))


def clearance_to_obstacles(target, obstacles):
    """Shortest distance from any obstacle POINT to the target centroid.

    Draws a circle at the target's xy centroid and reports how far the
    nearest obstacle point is from the circle's centre. The grasp is
    blocked when this distance is smaller than a radius chosen for the
    gripper (see grasp_feasibility). Using raw points instead of each
    obstacle's bounding-box centroid matters when an obstacle is long
    and sticks out toward the target: the centroid can be far while a
    point of the obstacle is right next to the target.

    Returns a value in metres, clipped to [0.0, 0.20]. 0.20 means
    nothing is near. For an obstacle without a point cloud (older
    client, server fallback), its centroid is used."""
    if target is None:
        return 0.0
    if not obstacles:
        return 0.20
    center = np.asarray(target["centroid"][:2], dtype=float)
    min_dist = 0.20
    for obstacle in obstacles:
        points = obstacle.get("points_xy")
        if points:
            pts = np.asarray(points, dtype=float)
            dist = float(np.min(np.linalg.norm(pts - center, axis=1)))
        else:
            dist = float(np.linalg.norm(
                np.asarray(obstacle["centroid"][:2], dtype=float) - center))
        if dist < min_dist:
            min_dist = dist
    return float(np.clip(min_dist, 0.0, 0.20))


# Oct 30: which rule decides "graspable".
#   "circle"  - (until Oct 29) no obstacle point within clearance_min of the
#               target centre, in ANY direction.
#   "fingers" - only the two places where the fingers come down must be
#               free: along the closing direction the grasp will use, on
#               both sides of the target, out to clearance_min from the
#               centre, in a band finger_half_width wide on each side of
#               the closing line. Obstacles in front of, behind or
#               diagonal to the target no longer block the grasp.
GRASP_RULE = {"rule": "fingers", "finger_half_width": 0.015}


def set_grasp_rule(rule="fingers", finger_half_width=0.015):
    if rule not in ("fingers", "circle"):
        raise ValueError("graspable rule must be 'fingers' or 'circle'")
    GRASP_RULE["rule"] = rule
    GRASP_RULE["finger_half_width"] = float(finger_half_width)


def grasp_closing(target):
    """The closing direction (radians, robot frame) the robot server's
    scripted grasp will use for this target. Mirrors the server
    (_closing_candidates + _tilt_plan): an elongated footprint (long/short
    > 1.3) is closed across its short side; a square one across either
    face, and of those the server takes the face whose closing line is
    most ACROSS the line from the robot (the fingertips then stay level
    on the tilted gripper). Without an orientation the server uses its
    canonical roll, which also closes across that line."""
    c = np.asarray(target["centroid"][:2], dtype=float)
    azimuth = float(np.arctan2(c[1], c[0]))
    yaw = target.get("yaw")
    if yaw is None:
        return azimuth + np.pi / 2.0
    yaw = float(yaw)
    long_side, short_side = target.get("rect_long"), target.get("rect_short")
    if (long_side is not None and short_side is not None
            and short_side > 1e-3 and long_side / short_side > 1.3):
        candidates = [yaw + np.pi / 2.0]
    else:
        candidates = [yaw + np.pi / 2.0, yaw]
    return min(candidates, key=lambda a: abs(np.cos(a - azimuth)))


def finger_clearance(target, obstacles, finger_half_width=None):
    """Distance (m) from the target centre, along the closing direction,
    to the nearest obstacle point that lies where a finger comes down:
    within finger_half_width of the closing line and beyond the target's
    own face. Clipped to [0, 0.20]; 0.20 = both finger places are free."""
    if target is None:
        return 0.0
    if finger_half_width is None:
        finger_half_width = GRASP_RULE["finger_half_width"]
    center = np.asarray(target["centroid"][:2], dtype=float)
    closing = grasp_closing(target)
    u = np.array([np.cos(closing), np.sin(closing)])
    v = np.array([-u[1], u[0]])
    # half the target's size along the closing direction (its face)
    own = target.get("points_xy")
    if own:
        face = float(np.max(np.abs(
            (np.asarray(own, dtype=float) - center).dot(u))))
    else:
        extent = np.asarray(target.get("extent", [0.03, 0.03]),
                            dtype=float)[:2]
        face = 0.5 * float(abs(u[0]) * extent[0] + abs(u[1]) * extent[1])
    min_dist = 0.20
    for obstacle in obstacles or []:
        points = obstacle.get("points_xy")
        if points:
            pts = np.asarray(points, dtype=float)
        else:
            pts = np.asarray([obstacle["centroid"][:2]], dtype=float)
        d = pts - center
        along = np.abs(d.dot(u))
        across = np.abs(d.dot(v))
        inside = (across < finger_half_width) & (along >= face - 0.005)
        if np.any(inside):
            min_dist = min(min_dist, float(np.min(along[inside])))
    return float(np.clip(min_dist, 0.0, 0.20))


def grasp_clearance(target, obstacles):
    """The clearance the graspable rule (GRASP_RULE) compares against
    clearance_min; also used for the clearance reward."""
    if GRASP_RULE["rule"] == "circle":
        return clearance_to_obstacles(target, obstacles)
    return finger_clearance(target, obstacles)


def grasp_feasibility(view, clearance_min=0.06, max_radial=0.38,
                      min_radial=0.0):
    """Is the target graspable right now, by rule? Returns a dict with the
    verdict and the numbers behind it.

    The rule: draw a circle of radius `clearance_min` at the target's xy
    centroid; the grasp is graspable only when no obstacle point lies
    inside AND the target centroid is inside the arm's reachable band.
    `clearance_min` default 0.06 m = target half-width (~0.015) +
    gripper half-opening (~0.022) + finger thickness (~0.010) + a 0.013
    safety margin. (The argument is still named clearance_min because
    older callers set it that way; the quantity it compares against is
    the output of clearance_to_obstacles.)"""
    target = view.get("target")
    if target is None:
        return {"graspable": False, "reason": "no target identified",
                "clearance": None, "radial": None}
    radial = float(np.hypot(target["centroid"][0], target["centroid"][1]))
    clearance = grasp_clearance(target, view.get("obstacles", []))
    if clearance < clearance_min:
        if GRASP_RULE["rule"] == "circle":
            reason = ("an obstacle point is %.0f mm from the target centre "
                      "(< %.0f mm)" % (clearance * 1000,
                                       clearance_min * 1000))
        else:
            reason = ("an obstacle point is where a finger comes down, "
                      "%.0f mm from the target centre along the closing "
                      "direction (< %.0f mm)"
                      % (clearance * 1000, clearance_min * 1000))
        return {"graspable": False, "reason": reason,
                "clearance": clearance, "radial": radial}
    if radial > max_radial:
        return {"graspable": False,
                "reason": "radial %.2f beyond reach %.2f" % (radial,
                                                             max_radial),
                "clearance": clearance, "radial": radial}
    if radial < min_radial:
        return {"graspable": False,
                "reason": "radial %.2f closer than the inner limit %.2f "
                          "(near table edge)" % (radial, min_radial),
                "clearance": clearance, "radial": radial}
    return {"graspable": True, "reason": "ok",
            "clearance": clearance, "radial": radial}

# ---------------- read-only perception visualization ----------------
# Invoke this file directly to debug detection without executing robot motions.
def render_perception_debug(frame, view, backend, feasibility=None):
    """Draw YOLO boxes, NanoSAM instance masks, target/obstacle IDs and XYZ.

    Returns a BGR uint8 image suitable for cv2.imshow or cv2.imwrite.
    Requires OpenCV only when visualization is requested.
    """
    import cv2
    rgb = np.asarray(frame["rgb"], dtype=np.uint8)
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = out.copy()
    instances = getattr(backend, "last_instances", [])
    accepted = set(getattr(backend, "last_accepted", []))
    colors = [(0, 210, 255), (255, 170, 0), (220, 60, 220),
              (60, 220, 80), (180, 180, 50), (200, 100, 70)]
    for i, instance in enumerate(instances):
        color = colors[i % len(colors)]
        mask = np.asarray(instance["mask"], dtype=bool)
        if mask.shape == out.shape[:2]:
            overlay[mask] = color
    out = cv2.addWeighted(overlay, 0.30, out, 0.70, 0.0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, instance in enumerate(instances):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = [int(v) for v in instance["box"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        state = "3D OK" if i in accepted else "3D REJECTED"
        rgb_text = (" RGB=%d,%d,%d" % tuple(int(round(v)) for v in instance["mean_rgb"])) \
            if "mean_rgb" in instance else ""
        label = "#%d %s %.2f %s%s" % (i, instance["label"],
                                      instance["score"], state, rgb_text)
        ytext = max(22, y1 - 8)
        cv2.putText(out, label, (max(0, x1), ytext), font, 0.47,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (max(0, x1), ytext), font, 0.47,
                    color, 1, cv2.LINE_AA)
        if i not in accepted:
            reason = str(instance.get("reject_reason") or "unknown")
            cv2.putText(out, reason[:75], (max(0, x1), min(out.shape[0]-8, y2+20)),
                        font, 0.42, (0, 0, 255), 1, cv2.LINE_AA)

    # Project base-frame centroids back to pixels for coordinate labels.
    K = np.asarray(frame["intrinsics"], dtype=float)
    T = np.asarray(frame["cam_to_base"], dtype=float)
    base_to_cam = np.linalg.inv(T)
    objects = []
    if view.get("target") is not None:
        objects.append(("TARGET", view["target"], (0, 0, 255)))
    for i, obstacle in enumerate(view.get("obstacles", [])):
        objects.append(("OBSTACLE %d" % (i + 1), obstacle, (0, 200, 0)))
    for name, obj, color in objects:
        xyz = np.asarray(obj["centroid"], dtype=float)
        cam = base_to_cam.dot(np.r_[xyz, 1.0])[:3]
        if cam[2] <= 0.001:
            continue
        uv = K.dot(cam)
        u, v = int(uv[0] / uv[2]), int(uv[1] / uv[2])
        if 0 <= u < out.shape[1] and 0 <= v < out.shape[0]:
            cv2.drawMarker(out, (u, v), color, cv2.MARKER_CROSS, 18, 2)
            label = "%s (%.3f, %.3f, %.3f)m" % (
                name, xyz[0], xyz[1], xyz[2])
            py = min(out.shape[0] - 8, max(15, v + 23))
            px = max(0, min(u + 12, out.shape[1] - 245))
            cv2.putText(out, label, (px, py), font, 0.43,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, label, (px, py), font, 0.43,
                        color, 1, cv2.LINE_AA)

    found = view.get("target") is not None
    info = "YOLO CPU + NanoSAM GPU: %d boxes | %d 3D objects | target: %s (%s)" % (
        len(instances), len(accepted), "YES" if found else "NO",
        view.get("target_source", "unknown"))
    if feasibility is not None:
        info += " | graspable: " + ("YES" if feasibility["graspable"] else "NO")
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (25, 25, 25), -1)
    cv2.putText(out, info, (8, 20), font, 0.47,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def verify_target_centroid(frame, view, backend, expected_xyz=None, tolerance=0.02):
    """Read-only diagnostics for the red target's reported base-frame centroid.

    This checks camera reprojection against the *independently generated* SAM mask,
    plus depth coverage and optional manually measured base-frame ground truth.
    Reprojection alone is a consistency check, NOT proof of metric accuracy:
    camera-to-base calibration errors can affect both calculations equally.
    """
    target = view.get("target")
    result = {"target_found": target is not None,
              "frame": "base_link", "units": "meters",
              "automatic_check_is_ground_truth": False}
    if target is None:
        result["status"] = "NO_TARGET"
        return result
    xyz = np.asarray(target["centroid"], dtype=np.float64)
    T = np.asarray(frame["cam_to_base"], dtype=np.float64)
    K = np.asarray(frame["intrinsics"], dtype=np.float64)
    depth = np.asarray(frame["depth"], dtype=np.uint16)
    h, w = depth.shape
    result["centroid_base_xyz_m"] = xyz.tolist()
    result["target_extent_xyz_m"] = list(target.get("extent", []))
    result["table_relative_center_height_m"] = float(xyz[2] - backend.config.table_z)
    result["expected_table_surface_z_m"] = float(backend.config.table_z)
    result["target_source"] = view.get("target_source")
    camera = np.linalg.inv(T).dot(np.append(xyz, 1.0))[:3]
    result["centroid_camera_xyz_m"] = camera.tolist()
    if camera[2] <= 0:
        result["status"] = "BEHIND_CAMERA"
        return result
    projection = K.dot(camera)
    u = float(projection[0] / projection[2])
    v = float(projection[1] / projection[2])
    result["projected_pixel_uv"] = [u, v]
    result["projected_centroid_camera_depth_m"] = float(camera[2])
    iu, iv = int(round(u)), int(round(v))
    in_frame = (0 <= iu < w and 0 <= iv < h)
    result["projected_pixel_in_frame"] = in_frame
    if in_frame:
        measured = int(depth[iv, iu])
        result["depth_at_projected_pixel_m"] = (measured / 1000.0 if measured else None)
        result["depth_difference_surface_minus_centroid_m"] = (
            measured / 1000.0 - float(camera[2]) if measured else None)
        # This difference is NOT required to be zero: the image measures a
        # visible surface, while the output is the sampled 3D point centroid.
    idx = target.get("instance_index")  # not present in simplified Route2 target
    dets = view.get("detections", [])
    best_detection = None
    if dets:
        # The target entry is a reduced dictionary; match its centroid back
        # to the full detection carrying the NanoSAM instance index.
        distances = [np.linalg.norm(np.asarray(d["centroid"]) - xyz)
                     for d in dets]
        j = int(np.argmin(distances))
        if distances[j] < 1e-5:
            best_detection = dets[j]
            idx = best_detection.get("instance_index", idx)
    instances = getattr(backend, "last_instances", [])
    if idx is not None and 0 <= idx < len(instances):
        instance = instances[idx]
        mask = np.asarray(instance["mask"], dtype=bool)
        result["sam_instance_index"] = int(idx)
        result["sam_mask_pixels_with_valid_depth"] = int(mask.sum())
        if mask.shape == depth.shape and mask.any():
            result["projected_centroid_inside_sam_mask"] = (
                bool(mask[iv, iu]) if in_frame else False)
            yy, xx = np.nonzero(mask)
            result["distance_to_nearest_mask_pixel_px"] = float(
                np.sqrt(np.min((xx.astype(float) - u) ** 2 +
                               (yy.astype(float) - v) ** 2)))
            depths = depth[mask]
            depths = depths[depths > 0]
            result["sam_mask_depth_median_m"] = (
                float(np.median(depths)) / 1000.0 if len(depths) else None)
    if expected_xyz is not None:
        expected = np.asarray(expected_xyz, dtype=np.float64)
        error = xyz - expected
        distance = float(np.linalg.norm(error))
        result["manually_measured_expected_xyz_m"] = expected.tolist()
        result["error_xyz_m"] = error.tolist()
        result["error_distance_m"] = distance
        result["within_tolerance"] = distance <= tolerance
        result["tolerance_m"] = float(tolerance)
        result["status"] = "PASS" if distance <= tolerance else "CHECK_CALIBRATION_OR_MASK"
    else:
        result["status"] = "VISUAL_CHECK_ONLY_NO_GROUND_TRUTH"
    return result


def add_target_verification_overlay(image, verification):
    """Highlight target projection and display compact read-only verification."""
    import cv2
    out = image.copy()
    h, w = out.shape[:2]
    uv = verification.get("projected_pixel_uv")
    if uv is not None:
        u, v = [int(round(x)) for x in uv]
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(out, (u, v), 13, (255, 255, 255), 3)
            cv2.circle(out, (u, v), 5, (0, 0, 255), -1)
            cv2.putText(out, "RED TARGET 3D CENTER", (max(2, u - 100),
                         min(h - 10, max(45, v - 22))),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255),
                         2, cv2.LINE_AA)
    if verification.get("target_found"):
        x, y, z = verification["centroid_base_xyz_m"]
        line = "TARGET base_link XYZ: %.3f, %.3f, %.3f m" % (x, y, z)
        if "error_distance_m" in verification:
            line += " | ruler error: %.1f mm" % (verification["error_distance_m"] * 1000)
        elif "projected_centroid_inside_sam_mask" in verification:
            line += " | center in SAM mask: %s" % (
                "YES" if verification["projected_centroid_inside_sam_mask"] else "NO")
    else:
        line = "TARGET NOT FOUND - no centroid to verify"
    cv2.rectangle(out, (0, h - 36), (w, h), (20, 20, 20), -1)
    cv2.putText(out, line[:110], (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.53, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _debug_frame_from_server(host, port, timeout=30.0):
    """Request a camera frame; no reset, step, or arm motion commands."""
    import socket
    import json
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(b'{"cmd":"frame"}\n')
        with sock.makefile("rb") as stream:
            response = json.loads(stream.readline().decode("utf-8"))
    if "error" in response:
        raise RuntimeError("robot frame error: %s" % response["error"])
    return decode_frame(response)


def _visualization_main():
    import argparse
    import json
    import os
    import time
    import cv2
    parser = argparse.ArgumentParser(
        description="Visualize YOLO CPU + NanoSAM GPU without moving the robot")
    parser.add_argument("--host", default="127.0.0.1",
                        help="LoCoBot server IP (port 9999)")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--frame-npz", default=None,
                        help="Offline .npz with rgb, depth, intrinsics, cam_to_base")
    parser.add_argument("--output-dir", default="perception_debug")
    parser.add_argument("--show", action="store_true",
                        help="Open an OpenCV window; press q to stop")
    parser.add_argument("--count", type=int, default=1,
                        help="Number of frames; 0 for continuous (use --show)")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Seconds between frames")
    parser.add_argument("--ws-x", type=float, nargs=2, default=[0.28, 0.60])
    parser.add_argument("--ws-y", type=float, nargs=2, default=[-0.25, 0.25])
    parser.add_argument("--table-z", type=float, default=0.225)
    parser.add_argument("--z-max", type=float, default=0.42)
    parser.add_argument("--target-label", default="red_object")
    parser.add_argument("--identity", choices=["color", "track"],
                        default="color")
    parser.add_argument("--expected-target-xyz", type=float, nargs=3,
                        metavar=("X", "Y", "Z"), default=None,
                        help="Optional independently measured red target center in base_link meters")
    parser.add_argument("--centroid-tolerance", type=float, default=0.02,
                        help="Allowed 3D distance from manual ground truth, meters (default 0.02)")
    args = parser.parse_args()
    if args.count < 0 or (args.count == 0 and not args.show):
        parser.error("--count must be positive, or 0 with --show")
    os.makedirs(args.output_dir, exist_ok=True)
    config = PerceptionConfig(ws_x=args.ws_x, ws_y=args.ws_y,
                              table_z=args.table_z, z_max=args.z_max)
    backend = make_backend("segmentation", config)
    perception = Route2Perception(backend, identity=args.identity,
                                  target_label=args.target_label)
    i = 0
    while args.count == 0 or i < args.count:
        if args.frame_npz:
            with np.load(args.frame_npz) as data:
                frame = {k: np.array(data[k]) for k in
                         ("rgb", "depth", "intrinsics", "cam_to_base")}
        else:
            frame = _debug_frame_from_server(args.host, args.port)
        view = perception.perceive(frame)
        feasibility = grasp_feasibility(view)
        annotated = render_perception_debug(frame, view, backend, feasibility)
        verification = verify_target_centroid(
            frame, view, backend, args.expected_target_xyz,
            args.centroid_tolerance)
        annotated = add_target_verification_overlay(annotated, verification)
        stem = "frame_%04d" % i
        png_path = os.path.join(args.output_dir, stem + ".png")
        json_path = os.path.join(args.output_dir, stem + ".json")
        cv2.imwrite(png_path, annotated)
        report = {
            "target_found": view["target_found"],
            "target_source": view["target_source"],
            "target": view["target"],
            "obstacles": view["obstacles"],
            "num_yolo_boxes": len(backend.last_instances),
            "num_accepted_3d_objects": len(backend.last_accepted),
            "instances": [{k: v for k, v in inst.items() if k != "mask"}
                          for inst in backend.last_instances],
            "detections": view.get("detections", []),
            "grasp_feasibility": feasibility,
            "target_centroid_verification": verification,
        }
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        print("[perception] %s | boxes=%d accepted=%d target=%s graspable=%s" % (
            png_path, len(backend.last_instances),
            len(backend.last_accepted), view["target_found"],
            feasibility["graspable"]))
        for obj in view.get("detections", []):
            print("[red ranking] instance=%s mean_RGB=%s red_distance=%s rejection=%s" % (
                obj.get("instance_index"),
                [round(x, 1) for x in obj.get("mean_rgb", [])],
                obj.get("red_distance"), obj.get("target_reject_reason")))
        for idx, inst in enumerate(backend.last_instances):
            if inst.get("reject_reason"):
                print("[3D reject] instance=%d: %s" % (idx, inst["reject_reason"]))
        print("[target center] %s | xyz=%s | uv=%s | mask=%s | error_m=%s" % (
            verification["status"], verification.get("centroid_base_xyz_m"),
            verification.get("projected_pixel_uv"),
            verification.get("projected_centroid_inside_sam_mask"),
            verification.get("error_distance_m")))
        if args.show:
            cv2.imshow("YOLO CPU + NanoSAM GPU (q to quit)", annotated)
            if cv2.waitKey(1 if args.count == 0 else 100) & 0xFF == ord("q"):
                break
        i += 1
        if args.count == 0 or i < args.count:
            time.sleep(max(0.0, args.interval))
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _visualization_main()
