# -*- coding: utf-8 -*-
"""set_params.py

Show, and if asked change, the ONE parameter set the robot server keeps.

The server loads ~/locobot_calibration.json at every start (it wins over
server_env.sh). calibrate_all.py writes that file. The training script reads
its geometry from the server. This tool is for looking at the stored numbers
and for changing one or two of them without a full calibration run.

    # show what is stored
    python3 set_params.py --host locobot.local

    # change and save (stays after a server restart)
    python3 set_params.py --host locobot.local TARGET_MAX_EXTENT=0.11 MIN_RADIAL=0.24

Run it on the Jetson. The server checks every value against a safe range and
refuses the whole request if one is out of range; nothing is half-applied.
"""

import argparse
import json
import socket
import sys

SHOWN = [
    ("TABLE_Z", "table height"),
    ("WS_X_MIN", "work area x min"), ("WS_X_MAX", "work area x max"),
    ("WS_Y_MIN", "work area y min"), ("WS_Y_MAX", "work area y max"),
    ("TARGET_X", "anchor x"), ("TARGET_Y", "anchor y"),
    ("MIN_RADIAL", "reach: closest to the robot"),
    ("MAX_RADIAL", "reach: farthest from the robot"),
    ("RETRACT_DZ", "lift before going home (above table)"),
    ("GRASP_PITCH", "gripper pitch (rad)"),
    ("ARM_Z_OFFSET", "fingertip height correction"),
    ("GRASP_DZ", "grasp depth below the block top"),
    ("PUSH_EE_Z", "push height"),
    ("GRASP_CLEARANCE", "grasp circle radius"),
    ("TARGET_MAX_EXTENT", "largest target size"),
]


def send(sock, rfile, msg):
    sock.sendall((json.dumps(msg) + "\n").encode())
    line = rfile.readline()
    if not line:
        raise RuntimeError("the server closed the connection")
    resp = json.loads(line)
    if "error" in resp:
        raise RuntimeError(resp["error"])
    return resp


def show(state):
    config, meta = state["config"], state.get("meta") or {}
    print("stored in %s%s%s" % (
        state.get("calib_file", "?"),
        " (saved %s)" % meta["saved_at"] if meta.get("saved_at") else "",
        "" if (state.get("calib_loaded") or meta.get("saved_at")) else
        "  -- no calibration file yet: the server runs on server_env.sh "
        "values. Run calibrate_all.py"))
    outline = meta.get("table_outline") or []
    print("    table outline: %s" % (
        "%d corners stored (pushes keep blocks on the table)" % len(outline)
        if outline else "NONE stored - run calibrate_all.py --camera_only"))
    for key, text in SHOWN:
        if key in config:
            print("    %-18s %9.3f   %s" % (key, config[key], text))
        else:
            print("    %-18s %9s   %s (old server build)" % (key, "-", text))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="locobot.local")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--hover", type=float, default=None,
                        help="after saving: ONE move, open jaw, to the "
                             "anchor (TARGET_X, TARGET_Y) at this height "
                             "above the table (m). Measure the gap between "
                             "fingertips and table with a ruler, press "
                             "Enter, the arm goes home. Nothing else moves")
    parser.add_argument("changes", nargs="*", help="NAME=VALUE ...")
    args = parser.parse_args(argv)

    values = {}
    for item in args.changes:
        name, sep, value = item.partition("=")
        if not sep:
            parser.error("%r is not NAME=VALUE" % item)
        values[name.strip().upper()] = float(value)

    sock = socket.create_connection((args.host, args.port), timeout=60)
    rfile = sock.makefile("r")
    try:
        if values:
            resp = send(sock, rfile, {"cmd": "set_config", "values": values})
            for name, (old, new) in sorted(resp.get("changed", {}).items()):
                print("changed %-18s %.3f -> %.3f" % (name, old, new))
            state = send(sock, rfile, {"cmd": "get_config"})
            meta = dict(state.get("meta") or {})
            meta.pop("saved_at", None)
            meta["edited_by"] = "set_params.py " + " ".join(args.changes)
            saved = send(sock, rfile, {"cmd": "save_config", "meta": meta})
            print("saved to %s\n" % saved.get("file"))
        state = send(sock, rfile, {"cmd": "get_config"})
        show(state)
        if args.hover is not None:
            c = state["config"]
            xyz = [c["TARGET_X"], c["TARGET_Y"], c["TABLE_Z"] + args.hover]
            # jaw OPEN: a closed jaw held while you measure makes the
            # gripper motor (ID 7) press its fingers together the whole
            # time, and that motor's overload shuts the robot down
            print("\nmoving the open jaw to %s (%.0f mm above the table) "
                  "..." % ([round(v, 3) for v in xyz], args.hover * 1000))
            send(sock, rfile, {"cmd": "gripper", "state": "open"})
            resp = send(sock, rfile, {"cmd": "goto", "xyz": xyz})
            if not resp.get("ok"):
                print("the arm refused this pose; nothing was measured")
            else:
                input("Measure from the TABLE to the lowest fingertip with a "
                      "ruler. It should be %.0f mm. Write it down, then "
                      "press Enter (the arm goes home) ... "
                      % (args.hover * 1000))
                print("If it is not %.0f mm: ARM_Z_OFFSET_new = "
                      "ARM_Z_OFFSET_now - (measured - %.0f mm)/1000 "
                      "(fingertips too high -> smaller value), e.g. "
                      "python3 set_params.py --host %s ARM_Z_OFFSET=..."
                      % (args.hover * 1000, args.hover * 1000, args.host))
            send(sock, rfile, {"cmd": "home"})
    except RuntimeError as error:
        print("the server refused: %s" % error)
        return 1
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
