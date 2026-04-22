#!/usr/bin/env python3
"""
J.A.R.V.I.S. Anomaly Detection Backend — Jetson Edition
Real-time object detection with anomaly rules, CSV logging, and WebSocket UI bridge.

Usage:
    python3 anomaly_detector.py [--camera 0] [--threshold 0.5] [--ws-port 8765]

Requirements (Jetson):
    - jetson.inference  (comes with JetPack)
    - jetson.utils
    - websockets        (pip3 install websockets)
    - OpenCV (optional for saving frames)
"""

import argparse
import asyncio
import csv
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# ── Jetson imports (graceful fallback for dev) ──
try:
    import jetson.inference
    import jetson.utils
    JETSON = True
except ImportError:
    JETSON = False
    print("[WARNING] jetson.inference not found — running in DEMO mode")

try:
    import websockets
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False
    print("[WARNING] websockets not installed — UI bridge disabled (pip3 install websockets)")

# ── Optional OpenCV for saving anomaly frames ──
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ════════════════════════════════════════════
#   CONFIG
# ════════════════════════════════════════════
RULES = {
    "max_persons":          3,           # Alert if more than N persons in frame
    "forbidden_classes":    ["cell phone"],  # Any of these → anomaly
    "proximity_threshold":  0.30,        # Fraction of frame width; persons closer → anomaly
    "min_confidence":       0.50,        # Ignore detections below this
    "cooldown_seconds":     2.0,         # Min seconds between logged anomalies (same type)
    "model":               "ssd-mobilenet-v2",
    "labels":              "networks/SSD-Mobilenet-v2/ssd_coco_labels.txt",
}

OUTPUT_DIR = Path("output")
LOG_FILE   = OUTPUT_DIR / "anomaly_log.csv"
IMG_DIR    = OUTPUT_DIR / "anomaly_images"

# ════════════════════════════════════════════
#   STATE
# ════════════════════════════════════════════
last_log_time: dict = defaultdict(float)   # rule_key → last log timestamp
ws_clients: set = set()                    # connected WebSocket clients
stats = {
    "total_frames":    0,
    "total_anomalies": 0,
    "start_time":      time.time(),
}

# ════════════════════════════════════════════
#   SETUP
# ════════════════════════════════════════════
def setup_output():
    OUTPUT_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "frame", "rule", "detail", "objects_in_frame"])


# ════════════════════════════════════════════
#   DETECTION HELPERS
# ════════════════════════════════════════════
def parse_detections(detections, img_width: int, img_height: int) -> List[Dict]:
    """Convert Jetson detection objects into plain dicts."""
    result = []
    for d in detections:
        if d.Confidence < RULES["min_confidence"]:
            continue
        x1 = d.Left   / img_width
        y1 = d.Top    / img_height
        x2 = d.Right  / img_width
        y2 = d.Bottom / img_height
        result.append({
            "class":      d.ClassLabel,
            "confidence": float(d.Confidence),
            "x1": x1, "y1": y1,
            "x2": x2, "y2": y2,
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
        })
    return result


def centroid_distance(a: dict, b: dict) -> float:
    return math.sqrt((a["cx"] - b["cx"]) ** 2 + (a["cy"] - b["cy"]) ** 2)


# ════════════════════════════════════════════
#   ANOMALY RULES
# ════════════════════════════════════════════
def evaluate_rules(detections: List[Dict], frame_idx: int) -> List[Tuple[str, str]]:
    """
    Returns list of (rule_key, detail_string) for each triggered rule.
    """
    violations = []
    now = time.time()

    persons = [d for d in detections if d["class"].lower() == "person"]

    # ── Rule 1: Person count ──────────────────────
    if len(persons) > RULES["max_persons"]:
        key = "overcrowd"
        if now - last_log_time[key] >= RULES["cooldown_seconds"]:
            detail = f"{len(persons)} persons detected (limit {RULES['max_persons']})"
            violations.append((key, detail))
            last_log_time[key] = now

    # ── Rule 2: Forbidden objects ─────────────────
    for d in detections:
        for forbidden in RULES["forbidden_classes"]:
            if forbidden.lower() in d["class"].lower():
                key = f"forbidden_{d['class'].lower().replace(' ','_')}"
                if now - last_log_time[key] >= RULES["cooldown_seconds"]:
                    detail = f"Forbidden object: '{d['class']}' ({d['confidence']:.0%} conf)"
                    violations.append((key, detail))
                    last_log_time[key] = now

    # ── Rule 3: Proximity ─────────────────────────
    if len(persons) >= 2:
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                dist = centroid_distance(persons[i], persons[j])
                if dist < RULES["proximity_threshold"]:
                    key = "proximity"
                    if now - last_log_time[key] >= RULES["cooldown_seconds"]:
                        detail = f"Persons too close: {dist*100:.0f}% frame width apart"
                        violations.append((key, detail))
                        last_log_time[key] = now

    return violations


# ════════════════════════════════════════════
#   LOGGING
# ════════════════════════════════════════════
def log_anomaly(frame_idx: int, rule: str, detail: str, detections: List[Dict]):
    ts = datetime.now().isoformat(timespec="seconds")
    classes = ", ".join(sorted(set(d["class"] for d in detections)))
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([ts, frame_idx, rule, detail, classes])
    print(f"[ANOMALY] {ts} | Frame {frame_idx:06d} | {rule}: {detail}")
    stats["total_anomalies"] += 1


# ════════════════════════════════════════════
#   BOUNDING BOX OVERLAY (Jetson native)
# ════════════════════════════════════════════
def draw_overlays(img, detections: List[Dict], violations: List[Tuple], img_w: int, img_h: int):
    """Draw JARVIS-style bounding boxes using jetson.utils font rendering."""
    if not JETSON:
        return
    font = jetson.utils.cudaFont()

    for d in detections:
        is_anom = any(rule in d["class"].lower() for rule in RULES["forbidden_classes"])
        color = (255, 68, 68, 200) if is_anom else (200, 152, 10, 200)

        x1 = int(d["x1"] * img_w); y1 = int(d["y1"] * img_h)
        x2 = int(d["x2"] * img_w); y2 = int(d["y2"] * img_h)

        # Draw bounding rect
        jetson.utils.cudaDrawRect(img, (x1, y1, x2, y2), color)

        label = f"{d['class'].upper()} {d['confidence']:.0%}"
        font.OverlayText(img, img_w, img_h, label, x1 + 4, y1 + 4,
                         (255, 215, 0, 220), (0, 0, 0, 160))

    # Anomaly banner
    if violations:
        banner = f"[JARVIS] ANOMALY: {violations[0][1][:60]}"
        font.OverlayText(img, img_w, img_h, banner,
                         5, 5, (255, 80, 80, 255), (0, 0, 0, 200))


# ════════════════════════════════════════════
#   WEBSOCKET BRIDGE
# ════════════════════════════════════════════
async def ws_handler(websocket, _path=None):
    ws_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        ws_clients.discard(websocket)


async def broadcast(payload: dict):
    if not ws_clients:
        return
    msg = json.dumps(payload)
    await asyncio.gather(
        *(ws.send(msg) for ws in list(ws_clients)),
        return_exceptions=True
    )


# ════════════════════════════════════════════
#   MAIN LOOP
# ════════════════════════════════════════════
async def run_detection(args):
    setup_output()

    if JETSON:
        net    = jetson.inference.detectNet(RULES["model"], threshold=RULES["min_confidence"])
        camera = jetson.utils.videoSource(f"/dev/video{args.camera}", argv=["--input-flip=none"])
        display = jetson.utils.videoOutput("webrtc://@:8554/jarvis") if args.display else None
    else:
        print("[DEMO] Generating simulated detections (no Jetson hardware)")

    frame_idx = 0
    fps_frames = 0
    fps_t0 = time.time()
    fps = 0.0

    print(f"\n{'='*55}")
    print("  J.A.R.V.I.S. ANOMALY DETECTION — ONLINE")
    print(f"  Rules: max_persons={RULES['max_persons']}, forbidden={RULES['forbidden_classes']}")
    print(f"  Log  : {LOG_FILE}")
    print(f"{'='*55}\n")

    while True:
        t_start = time.perf_counter()

        if JETSON:
            img   = camera.Capture()
            if img is None:
                continue
            w, h  = img.width, img.height
            raw   = net.Detect(img, overlay="none")
            dets  = parse_detections(raw, w, h)
        else:
            # Simulated detections for demo
            import random
            w, h = 1280, 720
            dets = [{"class": "person", "confidence": 0.92,
                     "x1": 0.1, "y1": 0.1, "x2": 0.35, "y2": 0.85,
                     "cx": 0.225, "cy": 0.47}]
            if (frame_idx // 30) % 3 == 0:
                dets.append({"class": "cell phone", "confidence": 0.77,
                             "x1": 0.55, "y1": 0.55, "x2": 0.67, "y2": 0.75,
                             "cx": 0.61, "cy": 0.65})
            await asyncio.sleep(0.033)   # ~30fps

        violations = evaluate_rules(dets, frame_idx)

        # Log anomalies
        for rule, detail in violations:
            log_anomaly(frame_idx, rule, detail, dets)

        # Draw overlays
        if JETSON:
            draw_overlays(img, dets, violations, w, h)
            if display:
                display.Render(img)

        # Broadcast to UI via WebSocket
        latency_ms = round((time.perf_counter() - t_start) * 1000, 1)
        if WS_AVAILABLE:
            await broadcast({
                "detections":   dets,
                "violations":   [list(v) for v in violations],
                "frame":        frame_idx,
                "fps":          round(fps, 1),
                "latency_ms":   latency_ms,
                "stats":        stats,
            })

        # FPS counter
        fps_frames += 1
        elapsed = time.time() - fps_t0
        if elapsed >= 1.0:
            fps = fps_frames / elapsed
            fps_frames = 0
            fps_t0 = time.time()

        stats["total_frames"] = frame_idx
        frame_idx += 1


async def main(args):
    tasks = [run_detection(args)]
    if WS_AVAILABLE:
        ws_server = websockets.serve(ws_handler, "localhost", args.ws_port)
        print(f"[WS] WebSocket server starting on ws://localhost:{args.ws_port}")
        tasks.append(ws_server)
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. Anomaly Detector")
    parser.add_argument("--camera",   type=int, default=0,    help="Camera device index")
    parser.add_argument("--display",  action="store_true",    help="Open WebRTC display output")
    parser.add_argument("--threshold",type=float, default=0.5,help="Detection confidence threshold")
    parser.add_argument("--ws-port",  type=int, default=8765, help="WebSocket server port")
    args = parser.parse_args()

    RULES["min_confidence"] = args.threshold
    asyncio.run(main(args))
