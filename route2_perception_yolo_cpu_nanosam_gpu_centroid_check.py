import argparse
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import route2_perception as p


def draw_workspace(ax, cfg):
    x0, x1 = cfg.ws_x
    y0, y1 = cfg.ws_y
    z0 = cfg.table_z + cfg.table_cut
    z1 = cfg.z_max

    for x in (x0, x1):
        for y in (y0, y1):
            ax.plot(
                [x, x], [y, y], [z0, z1],
                color="orange", linewidth=2
            )

    for z in (z0, z1):
        for x in (x0, x1):
            ax.plot(
                [x, x], [y0, y1], [z, z],
                color="orange", linewidth=2
            )

        for y in (y0, y1):
            ax.plot(
                [x0, x1], [y, y], [z, z],
                color="orange", linewidth=2
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="locobot.local")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--output-dir", default="debug_3d")
    parser.add_argument("--show", "--show-3d", dest="show_3d", action="store_true")
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Existing perception configuration and models
    cfg = p.PerceptionConfig()
    backend = p.make_backend("segmentation", cfg)
    perception = p.Route2Perception(
        backend,
        identity="color",
        target_label="red_object"
    )

    # Read-only camera request
    frame = p._debug_frame_from_server(
        args.host, args.port
    )

    rgb = frame["rgb"]
    depth = frame["depth"]
    K = frame["intrinsics"]
    T = frame["cam_to_base"]

    print("\n========== CAMERA DEBUG ==========")
    print("RGB shape:", rgb.shape)
    print("Depth shape:", depth.shape)
    print("Valid depth pixels:", np.count_nonzero(depth))
    print("Static pixels removed:", frame.get("static_pixels"))
    print("Camera-to-base transform:\n", T)
    print("Intrinsics:\n", K)

    # Run YOLO and NanoSAM normally
    view = perception.perceive(frame)

    print("\nYOLO boxes:", len(backend.last_instances))
    print("Accepted objects:", len(backend.last_accepted))
    print("Target found:", view["target_found"])

    # Original 2D visualization
    annotated = p.render_perception_debug(
        frame, view, backend
    )

    cv2.imwrite(
        os.path.join(args.output_dir, "rgb_debug.png"),
        annotated
    )

    # Create 3D figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Complete RGB-D cloud, before workspace crop
    cloud, colors = p.backproject(
        depth, K, T,
        stride=8,
        rgb=rgb
    )

    finite = np.isfinite(cloud).all(axis=1)
    cloud = cloud[finite]
    colors = colors[finite]

    if len(cloud):
        ax.scatter(
            cloud[:, 0],
            cloud[:, 1],
            cloud[:, 2],
            c=colors / 255.0,
            s=0.5,
            alpha=0.12,
            depthshade=False
        )

        print("\nFull cloud XYZ min:",
              np.round(cloud.min(axis=0), 3))
        print("Full cloud XYZ max:",
              np.round(cloud.max(axis=0), 3))

    # Workspace boundaries
    #draw_workspace(ax, cfg)

    # Collect red candidates independently of rejection
    red_candidates = []

    print("\n========== OBJECT DEBUG ==========")

    for i, inst in enumerate(backend.last_instances):

        mask = np.asarray(inst["mask"], dtype=bool)
        valid_mask = mask & (depth > 0)

        masked_depth = np.where(
            valid_mask, depth, 0
        ).astype(np.uint16)

        xyz, rgb_points = p.backproject(
            masked_depth, K, T,
            stride=2,
            rgb=rgb
        )

        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        rgb_points = rgb_points[finite]

        print("\nObject", i)
        print("YOLO confidence:", round(inst["score"], 3))
        print("Mask depth pixels:", int(valid_mask.sum()))

        if len(xyz) == 0:
            print("REJECTED: no usable 3D points")
            continue

        mean_rgb = rgb_points.mean(axis=0)
        centroid = xyz.mean(axis=0)

        crop = cfg.crop_mask(xyz)

        x_pass = (
            (xyz[:, 0] > cfg.ws_x[0]) &
            (xyz[:, 0] < cfg.ws_x[1])
        )

        y_pass = (
            (xyz[:, 1] > cfg.ws_y[0]) &
            (xyz[:, 1] < cfg.ws_y[1])
        )

        z_min_pass = (
            xyz[:, 2] > cfg.table_z + cfg.table_cut
        )

        z_max_pass = xyz[:, 2] < cfg.z_max

        print("Mean RGB:", np.round(mean_rgb, 1))
        print("Centroid XYZ:", np.round(centroid, 3))
        print("XYZ min:", np.round(xyz.min(axis=0), 3))
        print("XYZ max:", np.round(xyz.max(axis=0), 3))
        print("Points BEFORE crop:", len(xyz))
        print("Points AFTER crop:", int(crop.sum()))
        print("X passed:", int(x_pass.sum()))
        print("Y passed:", int(y_pass.sum()))
        print("Z above table:", int(z_min_pass.sum()))
        print("Z below max:", int(z_max_pass.sum()))

        # Draw each segmented object in its actual RGB color
        step = max(1, len(xyz) // 3000)

        sample_xyz = xyz[::step]
        sample_rgb = rgb_points[::step] / 255.0

        ax.scatter(
            sample_xyz[:, 0],
            sample_xyz[:, 1],
            sample_xyz[:, 2],
            c=sample_rgb,
            s=5,
            alpha=0.9,
            depthshade=False
        )

        # Select red candidate using average RGB
        r, g, b = mean_rgb

        is_red = (
            r >= 45 and
            r >= 1.25 * max(g, b)
        )

        if is_red:
            normalized = mean_rgb / max(
                mean_rgb.sum(), 1.0
            )

            red_distance = np.linalg.norm(
                normalized - np.array([1, 0, 0])
            )

            red_candidates.append(
                (red_distance, i, centroid)
            )

            print("RED CANDIDATE: YES")
        else:
            print("RED CANDIDATE: NO")

    # Large centroid marker for most-red object
    if red_candidates:
        _, idx, center = min(red_candidates)

        ax.scatter(
            [center[0]],
            [center[1]],
            [center[2]],
            s=500,
            c="red",
            marker="*",
            edgecolors="black",
            linewidths=1.5,
            label="Red candidate centroid",
            depthshade=False
        )

        ax.text(
            center[0],
            center[1],
            center[2] + 0.015,
            "RED #%d\n(%.3f, %.3f, %.3f)" %
            (idx, center[0], center[1], center[2])
        )

        print("\nRED TARGET CANDIDATE:", idx)
        print("XYZ:", np.round(center, 4))
        print("NOTE: pre-crop diagnostic only")

    else:
        print("\nNo red-dominant object found")

    # Accepted target, if any
    if view["target_found"] and view["target"] is not None:

        accepted_center = np.array(
            view["target"]["centroid"]
        )

        ax.scatter(
            [accepted_center[0]],
            [accepted_center[1]],
            [accepted_center[2]],
            s=350,
            c="lime",
            marker="D",
            edgecolors="black",
            label="Accepted target",
            depthshade=False
        )

    ax.set_xlabel("base_link X (m)")
    ax.set_ylabel("base_link Y (m)")
    ax.set_zlabel("base_link Z (m)")

    ax.set_title(
        "LoCoBot RGB-D + NanoSAM 3D Debug\n"
        "Orange = workspace | Red star = candidate centroid"
    )

    # Include both the workspace and actual cloud
    limits = [
        [cfg.ws_x[0], cfg.ws_x[1]],
        [cfg.ws_y[0], cfg.ws_y[1]],
        [cfg.table_z + cfg.table_cut, cfg.z_max]
    ]

    if len(cloud):
        for axis in range(3):
            low, high = np.percentile(
                cloud[:, axis], [2, 98]
            )
            limits[axis][0] = min(limits[axis][0], low)
            limits[axis][1] = max(limits[axis][1], high)

    for limits_axis in limits:
        limits_axis[0] -= 0.05
        limits_axis[1] += 0.05

    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])

    ax.view_init(elev=25, azim=-60)

    if red_candidates or view["target_found"]:
        ax.legend()

    fig.tight_layout()

    output = os.path.join(
        args.output_dir, "pointcloud_3d.png"
    )

    fig.savefig(output, dpi=140)

    print("\nSaved 3D image:", output)
    print("Saved 2D image:",
          os.path.join(args.output_dir, "rgb_debug.png"))

    if args.show_3d:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
