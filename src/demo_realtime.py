# src/demo_realtime.py
"""
Real-time pest detection + monitoring (YOLOv8 + OpenCV + Matplotlib)

Features
- Webcam or video file input
- YOLOv8 inference (Ultralytics)
- Draw boxes + labels + confidence
- Rolling counts per class over time (Matplotlib bar chart)
- Save annotated video if --save-path is provided
- Works with or without GUI (--no-show)

Usage examples
-------------
# 1) Webcam
python src/demo_realtime.py --weights runs/detect/train/weights/best.pt

# 2) Video file
python src/demo_realtime.py --weights runs/detect/train/weights/best.pt --source demo_video/pests.mp4

# 3) Headless saving (no windows)
python src/demo_realtime.py --weights runs/detect/train/weights/best.pt --source demo_video/pests.mp4 --no-show --save-path out/annotated.mp4

Hotkeys
-------
q     Quit
s     Save current frame as PNG (if window shown)
"""

import argparse
import time
from collections import Counter, deque, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Matplotlib is optional (can be disabled via --no-chart)
import matplotlib
matplotlib.use("TkAgg")  # safe for most desktops; ignored if --no-chart
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to YOLO .pt weights")
    ap.add_argument("--source", default="0", help="0=webcam, or path to video/image/folder/URL")
    ap.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    ap.add_argument("--device", default="0", help="GPU id like 0 or 'cpu'")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--no-show", action="store_true", help="Do not open OpenCV window")
    ap.add_argument("--no-chart", action="store_true", help="Do not show Matplotlib chart")
    ap.add_argument("--class-names", default="", help="Comma-separated override of class names order (optional)")
    ap.add_argument("--save-path", default="", help="Optional path to save annotated video mp4")
    ap.add_argument("--title", default="Pest Detector (YOLOv8)", help="Window/chart title")
    ap.add_argument("--rolling-seconds", type=int, default=10, help="Rolling window (sec) for counts")
    return ap.parse_args()


def open_capture(src: str | int):
    # webcam if "0" or "1" etc
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {src}")
    return cap


def init_video_writer(save_path, cap):
    if not save_path:
        return None
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cv2.VideoWriter(save_path, fourcc, fps, (w, h))


def setup_chart(title, class_names):
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.canvas.manager.set_window_title(title + " – State Engine")
    bars = ax.bar(range(len(class_names)), [0] * len(class_names))
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 10)  # auto-rescaled later
    ax.set_ylabel("detections (rolling)")
    ax.set_title("Detections per class (rolling window)")
    fig.tight_layout()
    return fig, ax, bars


def update_chart(ax, bars, counts, class_names):
    values = [counts.get(name, 0) for name in class_names]
    maxv = max(values) if values else 1
    for bar, v in zip(bars, values):
        bar.set_height(v)
    # autoscale y if needed
    if ax.get_ylim()[1] < maxv + 1:
        ax.set_ylim(0, maxv + 1)
    plt.pause(0.001)


def draw_boxes(frame, results, names):
    # results: list with one element for this frame
    for r in results:
        if r.boxes is None:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()  # (N,4)
        conf = r.boxes.conf.cpu().numpy()  # (N,)
        cls = r.boxes.cls.cpu().numpy().astype(int)  # (N,)
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            label = f"{names[k]} {c:.2f}" if k < len(names) else f"{k} {c:.2f}"
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (255, 200, 0), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    return frame


def main():
    args = parse_args()

    model = YOLO(args.weights)
    model.fuse()  # speed
    names = model.names

    # Optional override of class names order (useful for subset models)
    if args.class_names.strip():
        override = [s.strip() for s in args.class_names.split(",") if s.strip()]
        names = override

    cap = open_capture(args.source)
    writer = init_video_writer(args.save_path, cap)

    # rolling window buffer of (timestamp, class_name)
    window = deque()
    window_seconds = args.rolling_seconds

    # aggregate per-class rolling counts
    rolling_counts = Counter()

    # chart setup
    if args.no_chart:
        fig = ax = bars = None
    else:
        fig, ax, bars = setup_chart(args.title, names)

    # main loop
    last = time.time()
    cv2.namedWindow(args.title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.title, 960, 540)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Inference
            results = model.predict(
                source=frame,  # numpy array
                conf=args.conf,
                device=args.device,
                imgsz=args.imgsz,
                verbose=False
            )

            # Draw
            frame = draw_boxes(frame, results, names)

            # Update rolling counts
            now = time.time()
            # add current detections
            frame_counts = Counter()
            for r in results:
                if r.boxes is None:
                    continue
                for k in (r.boxes.cls.cpu().numpy().astype(int) if r.boxes.cls is not None else []):
                    cname = names[k] if k < len(names) else str(k)
                    frame_counts[cname] += 1
            for cname, cnt in frame_counts.items():
                for _ in range(int(cnt)):
                    window.append((now, cname))
                    rolling_counts[cname] += 1

            # purge old events
            cutoff = now - window_seconds
            while window and window[0][0] < cutoff:
                _, cname = window.popleft()
                rolling_counts[cname] -= 1
                if rolling_counts[cname] <= 0:
                    del rolling_counts[cname]

            # Chart
            if bars is not None:
                update_chart(ax, bars, rolling_counts, names)

            # Show / Save
            if not args.no_show:
                cv2.imshow(args.title, frame)
            if writer is not None:
                writer.write(frame)

            # Hotkeys
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and not args.no_show:
                out = Path("frame_capture.png")
                cv2.imwrite(str(out), frame)
                print(f"[i] Saved frame: {out}")

            # FPS print (optional)
            now2 = time.time()
            if now2 - last > 2.0:
                last = now2
                # quick fps estimate
                # (not exact because ultralytics handles internal batching)
                pass

    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()
        if not args.no_chart:
            plt.ioff()
            plt.close("all")


if __name__ == "__main__":
    main()
