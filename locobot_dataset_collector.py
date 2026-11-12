
#!/usr/bin/env python3
"""Read-only LoCoBot RGB dataset collector and YOLO box labeler.

Requires numpy, pillow and tkinter.
No YOLO, SAM, ROS or CUDA imports.
Uses only the robot server's read-only frame command.
"""

import argparse
import base64
import json
import queue
import socket
import threading
import time
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox


def unpack(packet):
    if packet.get("encoding") != "zlib+base64":
        raise ValueError("Unknown encoding")
    data = zlib.decompress(base64.b64decode(packet["data"]))
    return np.frombuffer(
        data, dtype=np.dtype(packet["dtype"])
    ).reshape(packet["shape"]).copy()


def fetch_image(host, port, timeout):
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(b'{"cmd":"frame","raw":true}\n')
        with sock.makefile("rb") as inp:
            line = inp.readline()

    if not line:
        raise RuntimeError("Server returned no frame")

    resp = json.loads(line.decode("utf-8"))
    if resp.get("error"):
        raise RuntimeError(resp["error"])

    arr = unpack(resp["rgb"])
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected RGB HxWx3")

    return Image.fromarray(arr.astype("uint8"), "RGB")


def resize_image(img, side):
    w, h = img.size
    if max(w, h) == side:
        return img
    scale = float(side) / max(w, h)
    nw = max(1, round(w * scale))
    nh = max(1, round(h * scale))
    resampling = getattr(Image, "Resampling", Image)
    return img.resize((nw, nh), resampling.LANCZOS)


class Collector:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.base = Path(args.output_dir).expanduser().resolve()
        self.images = self.base / "images"
        self.labels = self.base / "labels"
        self.images.mkdir(parents=True, exist_ok=True)
        self.labels.mkdir(parents=True, exist_ok=True)

        self.image_paths = sorted(self.images.glob("*.jpg"))
        self.index = len(self.image_paths) - 1
        self.img = None
        self.tk_img = None
        self.boxes = []
        self.drag_start = None
        self.scale = 1.0
        self.offset = (0, 0)

        self.running = False
        self.busy = False
        self.next_capture = None
        self.messages = queue.Queue()

        root.title("LoCoBot Dataset Collector")
        root.geometry("1160x900")

        toolbar = ttk.Frame(root, padding=8)
        toolbar.pack(fill="x")

        self.start_button = ttk.Button(
            toolbar, text="Start auto capture", command=self.toggle
        )
        self.start_button.pack(side="left", padx=4)

        buttons = [
            ("Capture now", self.capture),
            ("Previous", lambda: self.navigate(-1)),
            ("Next", lambda: self.navigate(1)),
            ("Undo box", self.undo),
            ("Save labels", self.save_labels),
            ("Mark empty", self.mark_empty),
        ]

        for name, callback in buttons:
            ttk.Button(
                toolbar, text=name, command=callback
            ).pack(side="left", padx=4)

        self.info = tk.StringVar(value="Ready to capture images")
        ttk.Label(
            root, textvariable=self.info, padding=(12, 4)
        ).pack(fill="x")

        self.canvas = tk.Canvas(
            root,
            bg="#202020",
            cursor="crosshair",
            highlightthickness=0
        )
        self.canvas.pack(
            fill="both", expand=True, padx=10, pady=8
        )

        self.canvas.bind("<Configure>", lambda event: self.draw())
        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        self.canvas.bind("<Button-3>", self.on_right_click)

        instructions = (
            "Draw one box per visible block. "
            "Right-click to delete a box. "
            "Save labels when finished. "
            "Automatic capture continues while labeling."
        )
        ttk.Label(
            root, text=instructions, padding=(12, 8)
        ).pack(fill="x")

        root.bind(
            "<Control-s>", lambda event: self.save_labels()
        )
        root.bind(
            "<Left>", lambda event: self.navigate(-1)
        )
        root.bind(
            "<Right>", lambda event: self.navigate(1)
        )
        root.protocol("WM_DELETE_WINDOW", self.close)

        if self.index >= 0:
            self.load(self.index)

        root.after(100, self.poll)

    def toggle(self):
        self.running = not self.running

        if self.running:
            self.start_button.configure(
                text="Stop auto capture"
            )
            self.next_capture = time.monotonic()
        else:
            self.start_button.configure(
                text="Start auto capture"
            )
            self.next_capture = None
            self.info.set("Automatic capture stopped")

    def capture(self):
        if self.busy:
            return

        self.busy = True

        def work():
            try:
                image = fetch_image(
                    self.args.host,
                    self.args.port,
                    self.args.timeout
                )
                image = resize_image(
                    image, self.args.image_size
                )

                stem = datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )
                path = self.images / (stem + ".jpg")

                image.save(
                    str(path), quality=95, subsampling=0
                )
                self.messages.put(("image", str(path)))

            except Exception as exc:
                self.messages.put(("error", str(exc)))

            finally:
                self.messages.put(("done", None))

        threading.Thread(
            target=work, daemon=True
        ).start()

    def poll(self):
        try:
            while True:
                kind, payload = self.messages.get_nowait()

                if kind == "image":
                    path = Path(payload)
                    self.image_paths.append(path)
                    self.image_paths.sort()

                    if self.img is None:
                        self.load(
                            self.image_paths.index(path)
                        )

                    self.info.set(
                        "Captured %s | Total images: %d" %
                        (path.name, len(self.image_paths))
                    )

                elif kind == "error":
                    self.info.set(
                        "Capture failed: " + payload
                    )

                elif kind == "done":
                    self.busy = False

        except queue.Empty:
            pass

        if (
            self.running
            and not self.busy
            and self.next_capture is not None
            and time.monotonic() >= self.next_capture
        ):
            self.next_capture = (
                time.monotonic() + self.args.interval
            )
            self.capture()

        self.root.after(100, self.poll)

    def load(self, index):
        if not 0 <= index < len(self.image_paths):
            return

        self.index = index
        path = self.image_paths[index]

        with Image.open(str(path)) as image:
            self.img = image.convert("RGB")

        self.boxes = []
        label_path = self.labels / (path.stem + ".txt")

        if label_path.exists():
            w, h = self.img.size

            for line in label_path.read_text().splitlines():
                fields = line.split()

                if len(fields) != 5 or fields[0] != "0":
                    continue

                cx, cy, bw, bh = map(
                    float, fields[1:]
                )

                self.boxes.append([
                    (cx - bw / 2) * w,
                    (cy - bh / 2) * h,
                    (cx + bw / 2) * w,
                    (cy + bh / 2) * h
                ])

        self.draw()

        self.info.set(
            "Image %d/%d | %d boxes | %dx%d" %
            (
                index + 1,
                len(self.image_paths),
                len(self.boxes),
                self.img.width,
                self.img.height
            )
        )

    def navigate(self, delta):
        if self.img is not None:
            self.save_labels(silent=True)

        self.load(self.index + delta)

    def bounds(self):
        if self.img is None:
            return 0, 0, 1

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        w, h = self.img.size

        scale = min(cw / w, ch / h, 1.0)

        return (
            (cw - w * scale) / 2,
            (ch - h * scale) / 2,
            scale
        )

    def draw(self):
        self.canvas.delete("all")

        if self.img is None:
            self.canvas.create_text(
                200, 100,
                text="No images captured yet",
                fill="white"
            )
            return

        ox, oy, scale = self.bounds()
        self.offset = (ox, oy)
        self.scale = scale

        display = self.img.resize((
            max(1, round(self.img.width * scale)),
            max(1, round(self.img.height * scale))
        ))

        self.tk_img = ImageTk.PhotoImage(display)

        self.canvas.create_image(
            ox, oy,
            anchor="nw",
            image=self.tk_img
        )

        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box

            px1 = ox + x1 * scale
            py1 = oy + y1 * scale
            px2 = ox + x2 * scale
            py2 = oy + y2 * scale

            self.canvas.create_rectangle(
                px1, py1, px2, py2,
                outline="#00ff70",
                width=3
            )

            self.canvas.create_text(
                px1 + 3,
                max(oy + 12, py1 - 10),
                text="block %d" % (i + 1),
                anchor="w",
                fill="#00ff70"
            )

    def point(self, x, y):
        if self.img is None:
            return None

        ox, oy = self.offset

        px = (x - ox) / self.scale
        py = (y - oy) / self.scale

        if not (
            0 <= px <= self.img.width
            and 0 <= py <= self.img.height
        ):
            return None

        return px, py

    def on_down(self, event):
        self.drag_start = self.point(
            event.x, event.y
        )

    def on_drag(self, event):
        if self.drag_start is None:
            return

        self.draw()
        pt = self.point(event.x, event.y)

        if pt is None:
            return

        a, b = self.drag_start
        ox, oy = self.offset

        self.canvas.create_rectangle(
            ox + a * self.scale,
            oy + b * self.scale,
            ox + pt[0] * self.scale,
            oy + pt[1] * self.scale,
            outline="yellow",
            width=2,
            dash=(5, 3)
        )

    def on_up(self, event):
        if self.drag_start is None:
            return

        pt = self.point(event.x, event.y)

        if pt is not None:
            x1, y1 = self.drag_start
            x2, y2 = pt

            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))

            if x2 - x1 >= 5 and y2 - y1 >= 5:
                self.boxes.append([
                    x1, y1, x2, y2
                ])

        self.drag_start = None
        self.draw()

    def on_right_click(self, event):
        pt = self.point(event.x, event.y)

        if pt is None:
            return

        x, y = pt

        for i in reversed(range(len(self.boxes))):
            x1, y1, x2, y2 = self.boxes[i]

            if x1 <= x <= x2 and y1 <= y <= y2:
                self.boxes.pop(i)
                self.draw()
                return

    def undo(self):
        if self.boxes:
            self.boxes.pop()
            self.draw()

    def save_labels(self, silent=False, force_empty=False):
        if self.img is None:
            return

        path = self.labels / (
            self.image_paths[self.index].stem + ".txt"
        )

        # Leave images unlabeled unless the user drew boxes
        # or explicitly marked the scene empty.
        if (
            not self.boxes
            and not force_empty
            and not path.exists()
        ):
            if not silent:
                self.info.set(
                    "No boxes saved. Use Mark empty "
                    "for scenes without objects."
                )
            return

        w, h = self.img.size
        rows = []

        for x1, y1, x2, y2 in self.boxes:
            cx = (x1 + x2) / (2 * w)
            cy = (y1 + y2) / (2 * h)
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            rows.append(
                "0 %.6f %.6f %.6f %.6f" %
                (cx, cy, bw, bh)
            )

        path.write_text(
            "\n".join(rows) + ("\n" if rows else "")
        )

        if not silent:
            self.info.set(
                "Saved %d boxes to %s" %
                (len(rows), path)
            )

    def mark_empty(self):
        if self.img is None:
            return

        confirmed = messagebox.askyesno(
            "Mark empty?",
            "Are there ZERO visible blocks?"
        )

        if confirmed:
            self.boxes = []
            self.draw()
            self.save_labels(force_empty=True)

    def close(self):
        self.running = False
        self.save_labels(silent=True)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host", required=True,
        help="IP address of the LoCoBot server"
    )
    parser.add_argument(
        "--port", type=int, default=9999
    )
    parser.add_argument(
        "--interval", type=float, default=15.0
    )
    parser.add_argument(
        "--image-size", type=int, default=1280
    )
    parser.add_argument(
        "--output-dir", default="block_dataset"
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0
    )

    args = parser.parse_args()

    if args.interval <= 0 or args.image_size <= 0:
        parser.error("Interval and image size must be positive")

    root = tk.Tk()
    Collector(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
