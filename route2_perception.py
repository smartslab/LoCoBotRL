1505
1506
1507
1508
1509
1510
1511
1512
1513
1514
1515
1516
1517
1518
1519
1520
1521
1522
1523
1524
1525
1526
1527
1528
1529
1530
1531
1532
1533
1534
1535
1536
1537
1538
1539
1540
1541
1542
1543
1544
1545
1546
1547
1548
1549
1550
1551
1552
1553
1554
1555
1556
1557
1558
1559
1560
1561
1562
1563
1564
1565
1566
1567
1568
1569
1570
1571
1572
1573
1574
1575
1576
1577
1578
1579
1580
1581
1582
1583
1584
1585
1586
1587
1588
1589
1590
1591
1592
1593
1594
1595
1596
1597
1598
1599
1600
1601
1602
1603
1604
1605
1606
1607
1608
1609
1610
1611
1612
1613
1614
1615
1616
1617
1618
1619
1620
1621
1622
1623
1624
1625
1626
1627
1628
1629
1630
1631
1632
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
