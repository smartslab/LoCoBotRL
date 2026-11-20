"""RGB-D perception, grasp geometry, and debug views for the RL trainer."""

import argparse
import base64
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import traceback
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


def _decode_array(packet):
    if packet["encoding"] != "zlib+base64":
        raise ValueError("unsupported camera array encoding")
    raw = zlib.decompress(base64.b64decode(packet["data"]))
    return np.frombuffer(raw, dtype=np.dtype(packet["dtype"])).reshape(packet["shape"])


def decode_frame(reply):
    """Unpack the robot's full-resolution, aligned RGB-D frame."""
    frame = dict(reply)
    frame["rgb"] = _decode_array(reply["rgb"])
    depth = _decode_array(reply["depth"])
    if (depth.ndim != 2 or frame["rgb"].shape != depth.shape + (3,)
            or frame["rgb"].dtype != np.uint8):
        raise ValueError("expected aligned RGB and depth frames")
    if reply.get("depth_unit", "millimetres") != "millimetres":
        raise ValueError("expected millimetre depth")
    frame["depth_raw"] = depth
    frame["depth"] = depth
    if reply.get("static_mask") is not None:
        static = _decode_array(reply["static_mask"]).astype(bool)
        frame["depth"] = np.where(static, 0, depth)
    frame["intrinsics"] = np.asarray(reply["intrinsics"], dtype=float).reshape(3, 3)
    frame["cam_to_base"] = np.asarray(reply["cam_to_base"], dtype=float).reshape(4, 4)
    return frame


def frame_signature(frame):
    """Identify the actual camera data used for segmentation and geometry."""
    return "rgb=%s depth=%s rgb_stamp=%s depth_stamp=%s" % (
        hashlib.sha256(np.ascontiguousarray(frame["rgb"])).hexdigest()[:16],
        hashlib.sha256(np.ascontiguousarray(frame["depth"])).hexdigest()[:16],
        frame.get("rgb_stamp"), frame.get("depth_stamp"))


def save_debug_frame(path, frame, perception):
    """Save one decoded frame and its workspace and target settings for replay."""
    options = {
        "identity": perception.identity,
        "target_label": perception.target_label,
        "anchor_xy": perception.anchor.tolist(),
        "track_max": perception.track_max,
        "hue_tol": perception.hue_tol,
        "target_max_extent": perception.target_max_extent,
    }
    metadata = {key: frame.get(key) for key in
                ("rgb_stamp", "depth_stamp", "static_pixels")}
    np.savez_compressed(
        path, rgb=frame["rgb"], depth=frame["depth"],
        intrinsics=frame["intrinsics"], cam_to_base=frame["cam_to_base"],
        metadata=json.dumps(metadata),
        config=json.dumps(asdict(perception.backend.config)),
        options=json.dumps(options))


def load_debug_frame(path):
    with np.load(path, allow_pickle=False) as archive:
        frame = json.loads(archive["metadata"].item())
        for key in ("rgb", "depth", "intrinsics", "cam_to_base"):
            frame[key] = archive[key]
        config = PerceptionConfig(**json.loads(archive["config"].item()))
        options = json.loads(archive["options"].item())
    return frame, config, options


def _camera_request(host, port, command):
    """Fetch a read-only JSON response using a short-lived connection."""
    with socket.create_connection((host, port), timeout=30) as conn:
        with conn.makefile("rb") as stream:
            conn.sendall((json.dumps({"cmd": command}) + "\n").encode("utf-8"))
            line = stream.readline()
    if not line:
        raise ConnectionError("LoCoBot closed the camera connection")
    reply = json.loads(line)
    if "error" in reply:
        raise RuntimeError(reply["error"])
    return reply


def _debug_frame_from_server(host, port):
    return decode_frame(_camera_request(host, port, "frame"))


def _project_pixels(depth, intrinsics, cam_to_base, rows, cols):
    z = depth[rows, cols].astype(np.float64) / 1000.0
    x = (cols - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (rows - intrinsics[1, 2]) * z / intrinsics[1, 1]
    return np.column_stack((x, y, z)) @ cam_to_base[:3, :3].T + cam_to_base[:3, 3]


def backproject(depth, intrinsics, cam_to_base, stride=1, rgb=None):
    """Project positive depth pixels into base_link, in metres."""
    rows, cols = np.nonzero(depth[::stride, ::stride] > 0)
    rows, cols = rows * stride, cols * stride
    points = _project_pixels(depth, intrinsics, cam_to_base, rows, cols)
    return points if rgb is None else (points, rgb[rows, cols])


@dataclass
class PerceptionConfig:
    ws_x: tuple = (0.28, 0.60)
    ws_y: tuple = (-0.25, 0.25)
    table_z: float = 0.225
    z_max: float = 0.425
    table_cut: float = 0.008
    min_points: int = 25
    grasp_clearance: float = 0.06

    def crop_mask(self, xyz):
        return (np.isfinite(xyz).all(axis=1)
                & (xyz[:, 0] > self.ws_x[0]) & (xyz[:, 0] < self.ws_x[1])
                & (xyz[:, 1] > self.ws_y[0]) & (xyz[:, 1] < self.ws_y[1])
                & (xyz[:, 2] > self.table_z + self.table_cut)
                & (xyz[:, 2] < self.z_max))


def _model_path(variable, candidates, filename):
    override = os.environ.get(variable)
    paths = [Path(override).expanduser()] if override else list(candidates)
    for path in paths:
        if path.is_file():
            return str(path)
    raise FileNotFoundError(
        "Could not find %s. Set %s to the full path or put %s in --model-dir."
        % (filename, variable, filename))


class SegmentationBackend:
    name = "YOLO + NanoSAM"

    def __init__(self, config, model_dir):
        self.config = config
        self.last_instances = []
        self.last_accepted = []
        home = Path.home()
        roots = ([Path(model_dir).expanduser()] if model_dir else
                 [Path.cwd(), Path.cwd() / "models", home / "nanosam" / "data"])
        weights = _model_path(
            "YOLO_WEIGHTS",
            [root / "best.pt" for root in roots] + [
                home / "runs/block_training/blocks_final/weights/best.pt"],
            "best.pt")
        encoder_name = "resnet18_image_encoder.engine"
        decoder_name = "mobile_sam_mask_decoder.engine"
        encoder = _model_path("NANOSAM_ENCODER",
                              [root / encoder_name for root in roots], encoder_name)
        decoder = _model_path("NANOSAM_DECODER",
                              [root / decoder_name for root in roots], decoder_name)
        from ultralytics import YOLO
        from nanosam.utils.predictor import Predictor
        self.yolo = YOLO(weights)
        self.predictor = Predictor(encoder, decoder)
        self.confidence = float(os.environ.get("YOLO_CONF", "0.50"))
        self.image_size = int(os.environ.get("YOLO_IMGSZ", "1280"))
        print("[perception] module=%s YOLO weights=%s conf=%.3f imgsz=%d" % (
            Path(__file__).resolve(), Path(weights).resolve(),
            self.confidence, self.image_size))
        print("[perception] NanoSAM encoder=%s decoder=%s" %
              (Path(encoder).resolve(), Path(decoder).resolve()))

    def instances(self, frame):
        from PIL import Image
        rgb = frame["rgb"]
        # Ultralytics interprets NumPy inputs as BGR, while NanoSAM takes PIL RGB.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        result = self.yolo.predict(
            bgr, imgsz=self.image_size, conf=self.confidence,
            device="cpu", verbose=False)[0]
        if result.boxes is None or not len(result.boxes):
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        self.predictor.set_image(Image.fromarray(rgb.copy()))
        instances = []
        for box, score in zip(boxes, scores):
            x0, y0, x1, y1 = box
            masks, iou, _ = self.predictor.predict(
                np.array([[x0, y0], [x1, y1]], dtype=np.float32),
                np.array([2, 3], dtype=np.int32))
            if hasattr(masks, "detach"):
                masks = masks.detach().cpu().numpy()
                iou = iou.detach().cpu().numpy()
            masks = np.asarray(masks)
            if masks.shape[-2:] != rgb.shape[:2]:
                raise ValueError("NanoSAM mask shape %r does not match camera %r"
                                 % (masks.shape, rgb.shape[:2]))
            masks = masks.reshape(-1, *rgb.shape[:2])
            mask = masks[0 if len(masks) == 1 else np.argmax(iou)]
            instances.append({"mask": mask > 0, "score": float(score),
                              "bbox": [int(v) for v in box]})
        return instances


class GeometricBackend:
    name = "RGB-D workspace components"

    def __init__(self, config):
        self.config = config
        self.last_instances = []
        self.last_accepted = []

    def instances(self, frame):
        depth = frame["depth"]
        rows, cols = np.nonzero(depth > 0)
        xyz = _project_pixels(depth, frame["intrinsics"], frame["cam_to_base"],
                              rows, cols)
        in_box = self.config.crop_mask(xyz)
        foreground = np.zeros(depth.shape, dtype=np.uint8)
        foreground[rows[in_box], cols[in_box]] = 1
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            foreground, connectivity=8)
        instances = []
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            if area >= self.config.min_points:
                instances.append({"mask": labels == index, "score": 1.0,
                                  "bbox": [int(x), int(y), int(x + w), int(y + h)]})
        return instances


def make_backend(kind, config, model_dir=None):
    if kind == "segmentation":
        return SegmentationBackend(config, model_dir)
    if kind == "geometric":
        return GeometricBackend(config)
    raise ValueError("unknown perception backend: " + kind)


def _object_geometry(points, hue, score):
    centroid = points.mean(axis=0)
    low, high = points.min(axis=0), points.max(axis=0)
    extent = high - low
    xy = points[:, :2] - centroid[:2]
    _, axes = np.linalg.eigh(xy.T @ xy)
    main_axis = axes[:, -1]
    yaw = float(np.arctan2(main_axis[1], main_axis[0]) % np.pi)
    rect = np.ptp(xy @ axes[:, ::-1], axis=0)
    if rect[0] < rect[1]:
        rect = rect[::-1]
        yaw = (yaw + np.pi / 2) % np.pi
    # A compact sample for JSON. Keep the centroid and extents full-resolution.
    xy_sample = points[::max(1, (len(points) + 255) // 256), :2]
    return {"centroid": centroid.tolist(), "extent": extent.tolist(),
            "yaw": yaw, "rect_long": float(rect[0]), "rect_short": float(rect[1]),
            "xy_bounds": [low[:2].tolist(), high[:2].tolist()],
            "points_xy": xy_sample.tolist(), "n": len(points), "score": score,
            "hue": hue, "label": "object"}


class Route2Perception:
    """Build the JSON-safe scene geometry used by training and robot execution."""

    hues = {"red": 0, "orange": 25, "yellow": 60, "green": 120,
            "cyan": 180, "blue": 240, "purple": 285}

    def __init__(self, backend, identity="color", target_label="red_object",
                 anchor_xy=(0.40, 0.03), track_max=0.05, hue_tol=35.0,
                 target_max_extent=0.085):
        self.backend = backend
        self.identity = identity
        self.target_label = target_label
        self.target_hue = self.hues.get(target_label.split("_")[0].lower())
        if self.target_hue is None and identity == "color":
            raise ValueError("unsupported target color: " + target_label)
        self.anchor = np.asarray(anchor_xy, dtype=float)
        self.track_max = float(track_max)
        self.hue_tol = float(hue_tol)
        self.target_max_extent = float(target_max_extent)
        self.target_xy = None

    def _is_target_color(self, hue):
        return (hue is not None and self.target_hue is not None
                and abs((hue - self.target_hue + 180) % 360 - 180) <= self.hue_tol)

    def _detect(self, frame, ee_xyz, exclude_arm=True):
        backend = self.backend
        cfg = backend.config
        backend.last_instances = backend.instances(frame)
        backend.last_accepted = []
        depth = frame["depth"]
        hsv = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2HSV)
        arm_blobs = 0
        detections = []
        for instance in backend.last_instances:
            rows, cols = np.nonzero(instance["mask"] & (depth > 0))
            points = _project_pixels(depth, frame["intrinsics"], frame["cam_to_base"],
                                     rows, cols)
            keep = cfg.crop_mask(points)
            points = points[keep]
            if len(points) < cfg.min_points:
                continue
            centroid = points.mean(axis=0)
            if (exclude_arm and ee_xyz is not None and centroid[2] > cfg.table_z + 0.07
                    and np.linalg.norm(centroid - np.asarray(ee_xyz)) < 0.045):
                arm_blobs += 1
                continue
            colors = hsv[rows[keep], cols[keep]]
            chroma = (colors[:, 1] >= 60) & (colors[:, 2] >= 45)
            hue = None
            if chroma.sum() >= max(5, len(points) * 0.25):
                # Circular mean handles reds on both sides of the hue wrap.
                angles = colors[chroma, 0] * (np.pi / 90.0)
                resultant = np.exp(1j * angles).mean()
                if abs(resultant) >= 0.55:
                    hue = float(np.degrees(np.angle(resultant)) % 360)
            item = _object_geometry(points, hue, instance["score"])
            if self._is_target_color(hue):
                item["label"] = self.target_label
            detections.append(item)
            backend.last_accepted.append(dict(instance, **item))
        return detections, arm_blobs

    def start_episode(self, frame, ee_xyz=None, debug_path=None):
        self.target_xy = None
        view = self.perceive(frame, ee_xyz=ee_xyz)
        if debug_path is not None:
            overlay = render_perception_debug(frame, view, self.backend)
            view["debug_saved"] = cv2.imwrite(debug_path, overlay)
            path = Path(debug_path)
            target_path = path.with_name(path.stem + "_target_rgbd.png")
            view["target_rgbd_saved"] = cv2.imwrite(
                str(target_path), render_target_rgbd_debug(frame, view, self.backend))
            clearance_path = path.with_name(path.stem + "_clearance.png")
            view["clearance_debug_saved"] = cv2.imwrite(
                str(clearance_path),
                render_clearance_debug(frame, view, self.backend))
        return view

    def perceive(self, frame, ee_xyz=None):
        detections, arm_blobs = self._detect(frame, ee_xyz)
        eligible = [d for d in detections
                    if max(d["extent"][:2]) <= self.target_max_extent]
        target = None
        if self.identity == "color":
            color_matches = [d for d in eligible if self._is_target_color(d["hue"])]
            if color_matches:
                target = min(color_matches, key=lambda d: (
                    abs((d["hue"] - self.target_hue + 180) % 360 - 180),
                    -d["score"]))
        elif eligible:
            reference = self.anchor if self.target_xy is None else self.target_xy
            nearest = min(eligible, key=lambda d: np.linalg.norm(
                np.asarray(d["centroid"][:2]) - reference))
            if np.linalg.norm(np.asarray(nearest["centroid"][:2]) - reference) <= self.track_max:
                target = nearest
        if target is not None:
            self.target_xy = np.asarray(target["centroid"][:2])
        reference = self.anchor if self.target_xy is None else self.target_xy
        obstacles = sorted((d for d in detections if d is not target),
                           key=lambda d: (np.linalg.norm(
                               np.asarray(d["centroid"][:2]) - reference),
                               d["centroid"][:2]))
        return {"target": target, "target_found": target is not None,
                "target_xy": None if target is None else target["centroid"][:2],
                "target_source": self.identity if target is not None else "none",
                "obstacles": obstacles[:4], "detections": detections,
                "n_objects": len(detections), "arm_blobs": arm_blobs,
                "proposal_count": len(self.backend.last_instances)}

    def target_on_table(self, frame, exclude_xyz=None):
        detections, _ = self._detect(frame, exclude_xyz)
        if self.identity == "track":
            # With no surviving identity cue, only confirm absence near the
            # last tracked position. A merged detection can still contain it.
            reference = self.anchor if self.target_xy is None else self.target_xy
            confirmed_present = any(
                np.linalg.norm(np.asarray(d["centroid"][:2]) - reference)
                <= self.track_max + xy_radius(d["extent"]) for d in detections)
        else:
            confirmed_present = any(self._is_target_color(d["hue"])
                                    for d in detections)
            # An oversized or missed mask can still contain red tabletop
            # pixels. Use the full aligned frame as a conservative check.
            if not confirmed_present:
                hsv = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2HSV)
                hue_delta = np.abs((2.0 * hsv[..., 0] - self.target_hue
                                    + 180.0) % 360.0 - 180.0)
                depth = frame["depth"]
                rows, cols = np.nonzero((hue_delta <= self.hue_tol)
                                        & (hsv[..., 1] >= 60)
                                        & (hsv[..., 2] >= 45) & (depth > 0))
                xyz = _project_pixels(depth, frame["intrinsics"],
                                      frame["cam_to_base"], rows, cols)
                keep = self.backend.config.crop_mask(xyz)
                if exclude_xyz is not None:
                    close_to_arm = (np.linalg.norm(xyz - np.asarray(exclude_xyz),
                                                   axis=1) < 0.045)
                    keep &= ~(close_to_arm & (xyz[:, 2] >
                                              self.backend.config.table_z + 0.07))
                confirmed_present = bool(keep.sum() >= self.backend.config.min_points)
        return {"target_on_table": bool(confirmed_present)}


class PerceptionWorker(Route2Perception):
    """Run segmentation in a fresh interpreter using a private local socket."""

    def __init__(self, config, model_dir=None, **options):
        super().__init__(SimpleNamespace(config=config, name="YOLO + NanoSAM worker"),
                         **options)
        self.channel, child = socket.socketpair()
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()),
                 "--worker-fd", str(child.fileno())],
                pass_fds=(child.fileno(),))
        except BaseException:
            self.channel.close()
            raise
        finally:
            child.close()
        self.channel.settimeout(120)
        self.stream = self.channel.makefile("rwb")
        try:
            self._request("init", config=asdict(config),
                          model_dir=model_dir, options=options)
        except BaseException:
            self.close()
            raise

    def _request(self, command, frame=None, **fields):
        payload = io.BytesIO()
        if frame is not None:
            np.savez(payload, **{key: frame[key] for key in
                                 ("rgb", "depth", "intrinsics", "cam_to_base")})
        data = payload.getbuffer()
        request = dict(fields, command=command, size=len(data))
        try:
            self.stream.write((json.dumps(request) + "\n").encode("utf-8"))
            self.stream.write(data)
            self.stream.flush()
            line = self.stream.readline()
        except OSError as error:
            raise RuntimeError("perception worker communication failed: %s" % error) from error
        if not line:
            raise RuntimeError("perception worker exited before replying to " + command)
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError("perception worker: " + response["error"])
        return response["result"]

    @staticmethod
    def _xyz(value):
        return None if value is None else np.asarray(value, dtype=float).tolist()

    def start_episode(self, frame, ee_xyz=None, debug_path=None):
        return self._request("start_episode", frame, ee_xyz=self._xyz(ee_xyz),
                             debug_path=debug_path)

    def perceive(self, frame, ee_xyz=None):
        return self._request("perceive", frame, ee_xyz=self._xyz(ee_xyz))

    def target_on_table(self, frame, exclude_xyz=None):
        return self._request("target_on_table", frame, ee_xyz=self._xyz(exclude_xyz))

    def close(self):
        self.stream.close()
        self.channel.close()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def _serve_worker(fd):
    perception = None
    with socket.socket(fileno=fd) as channel, channel.makefile("rwb") as stream:
        for line in iter(stream.readline, b""):
            request = json.loads(line)
            size = request["size"]
            data = stream.read(size)
            if len(data) != size:
                break
            command = request["command"]
            try:
                if command == "init":
                    cfg = PerceptionConfig(**request["config"])
                    backend = make_backend("segmentation", cfg, request["model_dir"])
                    perception = Route2Perception(backend, **request["options"])
                    result = None
                else:
                    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                        frame = {key: archive[key] for key in archive.files}
                    if command == "start_episode":
                        result = perception.start_episode(frame, request["ee_xyz"],
                                                          request["debug_path"])
                    elif command == "perceive":
                        result = perception.perceive(frame, request["ee_xyz"])
                    elif command == "target_on_table":
                        result = perception.target_on_table(frame, request["ee_xyz"])
                    else:
                        raise ValueError("unknown perception command: " + command)
                response = {"result": result}
            except Exception as error:
                traceback.print_exc()
                response = {"error": str(error)}
            stream.write((json.dumps(response) + "\n").encode("utf-8"))
            stream.flush()


def xy_radius(extent):
    return 0.5 * float(np.linalg.norm(np.asarray(extent)[:2]))


_GRASP_RULE = "fingers"
_FINGER_HALF_WIDTH = 0.015


def set_grasp_rule(rule, finger_half_width):
    global _GRASP_RULE, _FINGER_HALF_WIDTH
    if rule not in ("fingers", "circle") or finger_half_width <= 0:
        raise ValueError("invalid grasp rule or finger width")
    _GRASP_RULE = rule
    _FINGER_HALF_WIDTH = float(finger_half_width)


def _closing_direction(target):
    yaw = target.get("yaw")
    if yaw is None:
        return None
    long_side = target.get("rect_long") or 0.0
    short_side = target.get("rect_short") or 0.0
    candidates = [yaw + np.pi / 2]
    if short_side <= 1e-3 or long_side / short_side <= 1.3:
        candidates.append(yaw)
    x, y = target["centroid"][:2]
    azimuth = np.arctan2(y, x)
    closing = min(candidates, key=lambda angle: abs(np.cos(angle - azimuth)))
    return np.array([np.cos(closing), np.sin(closing)])


def grasp_clearance(target, obstacles):
    """Distance from the target center to clutter along the grasp approach."""
    center = np.asarray(target["centroid"][:2])
    axis = _closing_direction(target) if _GRASP_RULE == "fingers" else None
    cross = None if axis is None else np.array([-axis[1], axis[0]])
    clearance = 0.20
    for obstacle in obstacles:
        # Use the whole XY footprint to account for unseen sides of objects.
        # The same box keeps the circle and finger checks conservative.
        if "xy_bounds" in obstacle:
            low, high = np.asarray(obstacle["xy_bounds"])
            half = (high - low) / 2.0
            delta = (low + high) / 2.0 - center
        else:
            half = np.asarray(obstacle["extent"][:2]) / 2.0
            delta = np.asarray(obstacle["centroid"][:2]) - center
        if axis is None:
            distance = np.linalg.norm(np.maximum(np.abs(delta) - half, 0))
        else:
            if abs(delta @ cross) > half @ np.abs(cross) + _FINGER_HALF_WIDTH:
                continue
            distance = max(0.0, abs(delta @ axis) - half @ np.abs(axis))
        clearance = min(clearance, float(distance))
    return clearance


def grasp_feasibility(view, clearance_min, max_radial, min_radial=0.0):
    target = (view or {}).get("target")
    result = {"graspable": False, "clearance": None, "radial": None,
              "reason": "target not detected"}
    if target is None:
        return result
    radial = float(np.linalg.norm(target["centroid"][:2]))
    clearance = grasp_clearance(target, view.get("obstacles") or [])
    if not min_radial <= radial <= max_radial:
        reason = "target outside radial reach"
    elif clearance < clearance_min:
        reason = "insufficient gripper clearance"
    else:
        reason = ""
    return {"graspable": not reason, "clearance": clearance,
            "radial": radial, "reason": reason}


def render_perception_debug(frame, view, backend):
    overlay = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
    for instance in backend.last_instances:
        x0, y0, x1, y1 = instance["bbox"]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 180, 230), 1)
    for instance in backend.last_accepted:
        mask = instance["mask"]
        target = view["target"]
        accepted = target is not None and instance["centroid"] == target["centroid"]
        color = np.array([40, 230, 60] if accepted else [230, 160, 40])
        overlay[mask] = (0.6 * overlay[mask] + 0.4 * color).astype(np.uint8)
        x, y = instance["bbox"][:2]
        cv2.putText(overlay, "target" if accepted else instance["label"],
                    (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, tuple(int(v) for v in color), 1)
    return overlay


def _base_to_rgb_pixels(frame, points):
    base_to_cam = np.linalg.inv(frame["cam_to_base"])
    camera_points = points @ base_to_cam[:3, :3].T + base_to_cam[:3, 3]
    if not np.isfinite(camera_points).all() or (camera_points[:, 2] <= 1e-6).any():
        return None
    homogeneous = camera_points @ frame["intrinsics"].T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    if not np.isfinite(pixels).all() or (np.abs(pixels) >= 2**30).any():
        return None
    return np.rint(pixels).astype(np.int32)


def _debug_instance_pixels(frame, mask, cfg):
    depth = frame["depth"]
    rows, cols = np.nonzero(mask & (depth > 0))
    points = _project_pixels(depth, frame["intrinsics"], frame["cam_to_base"],
                             rows, cols)
    keep = cfg.crop_mask(points)
    return rows[keep], cols[keep], points[keep]


def _target_instance(view, backend):
    target = view.get("target")
    if target is None:
        return None
    # _detect stores a shallow copy, so this identifies the exact detection.
    return next((instance for instance in backend.last_accepted
                 if instance["centroid"] is target["centroid"]), None)


def render_target_rgbd_debug(frame, view, backend):
    """Show only the RGB and depth samples used to compute the target centroid."""
    rgb = np.zeros_like(frame["rgb"])
    depth_image = np.zeros_like(rgb)
    label = "Target not detected"
    depth_label = ""
    instance = _target_instance(view, backend)
    if instance is not None:
        rows, cols, _ = _debug_instance_pixels(frame, instance["mask"],
                                               backend.config)
        rgb[rows, cols] = frame["rgb"][rows, cols, ::-1]
        depth = frame["depth"][rows, cols].astype(float)
        low, high = depth.min(), depth.max()
        scaled = np.rint((depth - low) * 255 / max(high - low, 1)).astype(np.uint8)
        depth_image[rows, cols] = cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)[:, 0]
        centroid = np.asarray(view["target"]["centroid"])
        label = "base_link centroid (m): X %.3f   Y %.3f   Z %.3f" % tuple(centroid)
        depth_label = "Depth: %.0f to %.0f mm" % (low, high)
        pixels = _base_to_rgb_pixels(frame, centroid.reshape(1, 3))
        if pixels is not None:
            for image in (rgb, depth_image):
                cv2.drawMarker(image, tuple(pixels[0]), (0, 255, 255),
                               cv2.MARKER_CROSS, 15, 2, cv2.LINE_AA)
        top, bottom = max(0, rows.min() - 12), rows.max() + 13
        left, right = max(0, cols.min() - 12), cols.max() + 13
        rgb = rgb[top:bottom, left:right]
        depth_image = depth_image[top:bottom, left:right]

    height, width = rgb.shape[:2]
    panel_width = max(230, width)
    canvas = np.zeros((height + 56, 2 * panel_width + 8, 3), dtype=np.uint8)
    for index, (image, heading) in enumerate(((rgb, "Target RGB"),
                                             (depth_image, depth_label or "Target depth"))):
        start = index * (panel_width + 8)
        offset = start + (panel_width - width) // 2
        canvas[56:, offset:offset + width] = image
        cv2.putText(canvas, heading, (start + 8, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def render_clearance_debug(frame, view, backend):
    """Overlay the base_link XY ring and observed obstacle points inside it."""
    overlay = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
    cfg = backend.config
    target = view.get("target")
    label = "Clearance circle: target not detected"
    if target is not None:
        x, y = target["centroid"][:2]
        theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        radius = cfg.grasp_clearance
        target_instance = _target_instance(view, backend)
        obstacle_mask = np.zeros(frame["depth"].shape, dtype=bool)
        for instance in backend.last_accepted:
            if instance is not target_instance:
                obstacle_mask |= instance["mask"]
        if target_instance is not None:
            obstacle_mask &= ~target_instance["mask"]
        rows, cols, points = _debug_instance_pixels(frame, obstacle_mask, cfg)
        inside = (points[:, 0] - x)**2 + (points[:, 1] - y)**2 < radius**2
        overlay[rows[inside], cols[inside]] = (255, 0, 255)
        ring_base = np.column_stack((x + radius * np.cos(theta),
                                     y + radius * np.sin(theta),
                                     np.full(theta.shape, cfg.table_z)))
        polygon = _base_to_rgb_pixels(frame, ring_base)
        label = "Clearance circle: projection unavailable"
        if polygon is not None:
            cv2.polylines(overlay, [polygon], True, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.polylines(overlay, [polygon], True, (0, 255, 255), 2, cv2.LINE_AA)
            label = "Radius %.0f mm | Obstacle depth pixels inside: %d" % (
                1000.0 * radius, np.count_nonzero(inside))
    for color, thickness in [((0, 0, 0), 3), ((0, 255, 255), 1)]:
        cv2.putText(overlay, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, thickness, cv2.LINE_AA)
    return overlay


def draw_workspace(ax, cfg):
    x0, x1 = cfg.ws_x
    y0, y1 = cfg.ws_y
    z0, z1 = cfg.table_z + cfg.table_cut, cfg.z_max
    for x in (x0, x1):
        for y in (y0, y1):
            ax.plot([x, x], [y, y], [z0, z1], color="orange", linewidth=2)
    for z in (z0, z1):
        for x in (x0, x1):
            ax.plot([x, x], [y0, y1], [z, z], color="orange", linewidth=2)
        for y in (y0, y1):
            ax.plot([x0, x1], [y, y], [z, z], color="orange", linewidth=2)


def finite_cloud(depth, frame, stride):
    points, colors = backproject(
        depth, frame["intrinsics"], frame["cam_to_base"],
        stride=stride, rgb=frame["rgb"])
    finite = np.isfinite(points).all(axis=1)
    return points[finite], colors[finite]


def render_frame(frame, perception, backend, cfg, output_dir,
                 show_3d=False, show_workspace=False):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # registers projection on older Matplotlib

    rgb, depth = frame["rgb"], frame["depth"]
    print("\n========== CAMERA DEBUG ==========")
    print("RGB shape:", rgb.shape, "Depth shape:", depth.shape)
    print("[perception] " + frame_signature(frame))
    print("Valid depth pixels:", np.count_nonzero(depth))
    print("Static pixels removed:", frame.get("static_pixels"))
    print("Camera-to-base transform:\n", frame["cam_to_base"])
    print("Intrinsics:\n", frame["intrinsics"])

    view = perception.perceive(frame)
    print("YOLO boxes:", len(backend.last_instances))
    print("Accepted objects:", len(backend.last_accepted))
    print("Target found:", view["target_found"])
    rgb_path = os.path.join(output_dir, "rgb_debug.png")
    if not cv2.imwrite(rgb_path, render_perception_debug(frame, view, backend)):
        raise OSError("Could not write " + rgb_path)
    target_path = os.path.join(output_dir, "target_rgbd.png")
    if not cv2.imwrite(target_path, render_target_rgbd_debug(frame, view, backend)):
        raise OSError("Could not write " + target_path)
    clearance_path = os.path.join(output_dir, "rgb_clearance.png")
    if not cv2.imwrite(clearance_path, render_clearance_debug(frame, view, backend)):
        raise OSError("Could not write " + clearance_path)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    cloud, colors = finite_cloud(depth, frame, stride=8)
    if len(cloud):
        ax.scatter(*cloud.T, c=colors / 255.0, s=0.5, alpha=0.12,
                   depthshade=False)
        print("Full cloud XYZ min:", np.round(cloud.min(axis=0), 3))
        print("Full cloud XYZ max:", np.round(cloud.max(axis=0), 3))
    if show_workspace:
        draw_workspace(ax, cfg)

    red_candidates = []
    print("\n========== OBJECT DEBUG ==========")
    for i, inst in enumerate(backend.last_instances):
        valid_mask = np.asarray(inst["mask"], dtype=bool) & (depth > 0)
        masked_depth = np.where(valid_mask, depth, 0).astype(np.uint16)
        xyz, rgb_points = finite_cloud(masked_depth, frame, stride=2)
        print("\nObject", i, "YOLO confidence:", round(inst["score"], 3))
        print("Mask depth pixels:", int(valid_mask.sum()))
        if not len(xyz):
            print("REJECTED: no usable 3D points")
            continue
        mean_rgb = rgb_points.mean(axis=0)
        centroid = xyz.mean(axis=0)
        print("Mean RGB:", np.round(mean_rgb, 1))
        print("Centroid XYZ:", np.round(centroid, 3))
        print("XYZ min:", np.round(xyz.min(axis=0), 3))
        print("XYZ max:", np.round(xyz.max(axis=0), 3))
        print("Points BEFORE crop:", len(xyz))
        print("Points AFTER crop:", int(cfg.crop_mask(xyz).sum()))
        print("X passed:", int(((xyz[:, 0] > cfg.ws_x[0]) &
                                (xyz[:, 0] < cfg.ws_x[1])).sum()))
        print("Y passed:", int(((xyz[:, 1] > cfg.ws_y[0]) &
                                (xyz[:, 1] < cfg.ws_y[1])).sum()))
        print("Z above table:", int((xyz[:, 2] > cfg.table_z + cfg.table_cut).sum()))
        print("Z below max:", int((xyz[:, 2] < cfg.z_max).sum()))
        step = max(1, len(xyz) // 3000)
        ax.scatter(*xyz[::step].T, c=rgb_points[::step] / 255.0,
                   s=5, alpha=0.9, depthshade=False)

        r, g, b = mean_rgb
        is_red = r >= 45 and r >= 1.25 * max(g, b)
        print("RED CANDIDATE:", "YES" if is_red else "NO")
        if is_red:
            normalized = mean_rgb / max(mean_rgb.sum(), 1.0)
            distance = np.linalg.norm(normalized - np.array([1, 0, 0]))
            red_candidates.append((distance, i, centroid))

    if red_candidates:
        _, index, center = min(red_candidates)
        ax.scatter(*center, s=500, c="red", marker="*", edgecolors="black",
                   linewidths=1.5, label="Red candidate centroid",
                   depthshade=False)
        ax.text(center[0], center[1], center[2] + 0.015,
                "RED #%d\n(%.3f, %.3f, %.3f)" % (index, *center))
        print("\nRED TARGET CANDIDATE:", index, np.round(center, 4))
        print("NOTE: pre-crop diagnostic only")
    else:
        print("\nNo red-dominant object found")

    target = view["target"]
    if view["target_found"] and target is not None:
        ax.scatter(*target["centroid"], s=350, c="lime", marker="D",
                   edgecolors="black", label="Accepted target", depthshade=False)

    ax.set(xlabel="base_link X (m)", ylabel="base_link Y (m)",
           zlabel="base_link Z (m)", title="LoCoBot RGB-D + NanoSAM 3D Debug")
    limits = [list(cfg.ws_x), list(cfg.ws_y),
              [cfg.table_z + cfg.table_cut, cfg.z_max]]
    if len(cloud):
        for axis, (low, high) in enumerate(np.percentile(cloud, [2, 98], axis=0).T):
            limits[axis][0] = min(limits[axis][0], low)
            limits[axis][1] = max(limits[axis][1], high)
    for set_limit, (low, high) in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), limits):
        set_limit(low - 0.05, high + 0.05)

    ax.view_init(elev=25, azim=-60)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    cloud_path = os.path.join(output_dir, "pointcloud_3d.png")
    fig.savefig(cloud_path, dpi=140)
    print("\nSaved 3D image:", cloud_path)
    print("Saved 2D image:", rgb_path)
    print("Saved target RGB-D image:", target_path)
    print("Saved clearance overlay:", clearance_path)
    if show_3d:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--host", default="locobot.local")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--output-dir", default="debug_3d")
    parser.add_argument("--show", "--show-3d", dest="show_3d", action="store_true")
    parser.add_argument("--show-workspace", action="store_true")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--frame-file", help="replay an ik_frame.npz capture offline")
    args = parser.parse_args()
    if args.worker_fd is not None:
        _serve_worker(args.worker_fd)
        return
    if args.count < 1:
        parser.error("--count must be positive")
    if args.frame_file and args.count != 1:
        parser.error("--frame-file takes one saved frame")

    saved_frame = None
    if args.frame_file:
        saved_frame, cfg, options = load_debug_frame(args.frame_file)
    else:
        calibrated = _camera_request(args.host, args.port, "get_config")["config"]
        cfg = PerceptionConfig(
            ws_x=(calibrated["WS_X_MIN"], calibrated["WS_X_MAX"]),
            ws_y=(calibrated["WS_Y_MIN"], calibrated["WS_Y_MAX"]),
            table_z=calibrated["TABLE_Z"], z_max=calibrated["WS_Z_MAX"],
            grasp_clearance=calibrated.get("GRASP_CLEARANCE", 0.06))
        options = {"identity": "color", "target_label": "red_object",
                   "target_max_extent": calibrated["TARGET_MAX_EXTENT"]}

    backend = make_backend("segmentation", cfg, model_dir=args.model_dir)
    perception = Route2Perception(backend, **options)
    for i in range(args.count):
        output_dir = args.output_dir
        if args.count > 1:
            output_dir = os.path.join(output_dir, "frame_%03d" % (i + 1))
        os.makedirs(output_dir, exist_ok=True)
        frame = saved_frame if saved_frame is not None else (
            _debug_frame_from_server(args.host, args.port))
        render_frame(frame, perception, backend, cfg, output_dir,
                     args.show_3d, args.show_workspace)


if __name__ == "__main__":
    main()
