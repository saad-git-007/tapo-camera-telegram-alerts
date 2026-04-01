"""
Porch Security Detector — Telegram Edition (OpenCV DNN face + YOLO11 NCNN)
==========================================================================

Detection triggers:
  1. OpenCV DNN face detector (Caffe SSD)
  2. Ultralytics YOLO11 NCNN for person + backpack + suitcase

Alert delivery:
  - Annotated JPEG saved to SNAPSHOT_DIR
  - Telegram bot uploads the local JPEG directly using sendPhoto
  - 1-minute MP4 video recorded after the first trigger in a burst

Notes:
  - backpack/suitcase are only package proxies, not true parcel detection
  - downloads the OpenCV face model files automatically the first time
"""

from __future__ import annotations

import cv2
import time
import threading
import queue
import logging
import os
import glob
import itertools
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from ultralytics import YOLO

import config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("detector.log"),
    ],
)
log = logging.getLogger(__name__)

FRAME_INTERVAL = 1.0 / config.TARGET_FPS
FACE_DIR = Path(__file__).resolve().parent / "models"
FACE_PROTO = FACE_DIR / "face_deploy.prototxt"
FACE_MODEL = FACE_DIR / "face_detector.caffemodel"
FACE_PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/"
    "face_detector/deploy.prototxt"
)
FACE_MODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/"
    "dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)
YOLO_CLASSES = [0, 24, 28]  # person, backpack, suitcase


def validate_config():
    problems = []

    if "YOURPASSWORD" in str(config.RTSP_URL):
        problems.append("RTSP_URL still contains YOURPASSWORD")

    if "YOUR_BOT_TOKEN_HERE" in str(config.TELEGRAM_BOT_TOKEN):
        problems.append("TELEGRAM_BOT_TOKEN is still a placeholder")

    if str(config.TELEGRAM_CHAT_ID).strip() in {"", "123456789"}:
        problems.append("TELEGRAM_CHAT_ID is still a placeholder")

    if problems:
        for p in problems:
            log.error("Config error: %s", p)
        raise SystemExit("Fix config.py before running detector.py")


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2


class TelegramSender(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._pq = queue.PriorityQueue()
        self._counter = itertools.count()
        self._session = requests.Session()
        self._pending = set()
        self._pending_lock = threading.Lock()

    def enqueue(self, image_path: str, caption: str):
        image_path = os.path.abspath(image_path)
        with self._pending_lock:
            self._pending.add(image_path)
        self._pq.put((time.time(), next(self._counter), image_path, caption, 0))
        log.info("Alert queued → Telegram: %s", caption.replace("\n", " | "))

    def is_pending(self, image_path: str) -> bool:
        image_path = os.path.abspath(image_path)
        with self._pending_lock:
            return image_path in self._pending

    def _mark_done(self, image_path: str):
        image_path = os.path.abspath(image_path)
        with self._pending_lock:
            self._pending.discard(image_path)

    def run(self):
        while True:
            next_try, _, image_path, caption, attempt = self._pq.get()

            wait = next_try - time.time()
            if wait > 0:
                time.sleep(wait)

            if not os.path.exists(image_path):
                log.warning("Snapshot missing before Telegram send, dropping: %s", image_path)
                self._mark_done(image_path)
                continue

            if self._send_once(image_path, caption):
                self._mark_done(image_path)
                continue

            attempt += 1
            backoff = min(5 * (2 ** (attempt - 1)), config.TELEGRAM_MAX_RETRY_BACKOFF_SEC)
            retry_at = time.time() + backoff
            log.warning(
                "Telegram send failed — retry #%d in %ds for %s",
                attempt,
                backoff,
                os.path.basename(image_path),
            )
            self._pq.put((retry_at, next(self._counter), image_path, caption, attempt))

    def _send_once(self, image_path: str, caption: str) -> bool:
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendPhoto"

        data = {
            "chat_id": str(config.TELEGRAM_CHAT_ID),
            "caption": caption,
            "parse_mode": "Markdown",
        }

        if config.TELEGRAM_DISABLE_NOTIFICATION:
            data["disable_notification"] = "true"

        if config.TELEGRAM_MESSAGE_THREAD_ID is not None:
            data["message_thread_id"] = str(config.TELEGRAM_MESSAGE_THREAD_ID)

        try:
            with open(image_path, "rb") as f:
                files = {
                    "photo": (os.path.basename(image_path), f, "image/jpeg"),
                }
                r = self._session.post(url, data=data, files=files, timeout=(10, 90))

            if r.status_code != 200:
                log.warning("Telegram HTTP %s: %s", r.status_code, r.text[:500])
                return False

            payload = r.json()
            if not payload.get("ok"):
                log.warning("Telegram API error: %s", payload)
                return False

            message_id = payload.get("result", {}).get("message_id")
            log.info("Telegram sent ✓ message_id=%s file=%s", message_id, os.path.basename(image_path))
            return True

        except Exception as e:
            log.warning("Telegram request exception: %s", e)
            return False


class MediaCleaner(threading.Thread):
    def __init__(self, sender: TelegramSender):
        super().__init__(daemon=True)
        self.sender = sender

    def run(self):
        while True:
            time.sleep(60)
            try:
                now = time.time()

                snapshot_pattern = os.path.join(config.SNAPSHOT_DIR, "*.jpg")
                for f in glob.glob(snapshot_pattern):
                    f = os.path.abspath(f)
                    if self.sender.is_pending(f):
                        continue
                    age = now - os.path.getmtime(f)
                    if age > config.SNAPSHOT_MAX_AGE_SEC:
                        os.remove(f)
                        log.debug("Deleted old snapshot: %s", f)

                video_pattern = os.path.join(config.VIDEO_DIR, "*.mp4")
                for f in glob.glob(video_pattern):
                    f = os.path.abspath(f)
                    age = now - os.path.getmtime(f)
                    if age > config.VIDEO_MAX_AGE_SEC:
                        os.remove(f)
                        log.debug("Deleted old video: %s", f)

            except Exception as e:
                log.warning("Media cleanup error: %s", e)


class Cooldown:
    def __init__(self):
        self._last = {}

    def ready(self, key: str, seconds: float) -> bool:
        now = time.time()
        if now - self._last.get(key, 0) >= seconds:
            self._last[key] = now
            return True
        return False


class FrameCapture(threading.Thread):
    def __init__(self, rtsp_url: str):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        self._frame = None
        self._lock = threading.Lock()

    def run(self):
        while True:
            log.info("Connecting to camera: %s", self.rtsp_url)
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                log.warning("Cannot open stream — retrying in 5s")
                time.sleep(5)
                continue

            log.info("Camera stream open ✓")
            while True:
                ret, frame = cap.read()
                if not ret:
                    log.warning("Frame read failed — reconnecting in 3s")
                    break
                with self._lock:
                    self._frame = frame

            cap.release()
            time.sleep(3)

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


class VideoRecorder(threading.Thread):
    def __init__(self, capture: FrameCapture):
        super().__init__(daemon=True)
        self.capture = capture
        self._lock = threading.Lock()
        self._writer = None
        self._video_path = None
        self._stop_at = 0.0
        self._active = False

    def is_recording(self) -> bool:
        with self._lock:
            return self._active

    def start_recording(self, reason: str) -> bool:
        with self._lock:
            if self._active:
                log.info(
                    "Video recording already active, skipping new trigger: %s",
                    reason,
                )
                return False

        frame = self.capture.get_frame()
        if frame is None:
            log.warning("Cannot start video recording: no frame available yet")
            return False

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        add_timestamp(frame)

        Path(config.VIDEO_DIR).mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{reason}_{ts}.mp4"
        video_path = os.path.abspath(os.path.join(config.VIDEO_DIR, filename))

        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, float(config.VIDEO_FPS), (w, h))

        if not writer.isOpened():
            log.warning("Failed to open video writer for: %s", video_path)
            return False

        writer.write(frame)

        with self._lock:
            self._writer = writer
            self._video_path = video_path
            self._stop_at = time.time() + float(config.VIDEO_RECORD_SECONDS)
            self._active = True

        log.info("Video recording started: %s (reason=%s)", video_path, reason)
        return True

    def run(self):
        frame_interval = 1.0 / float(config.VIDEO_FPS)

        while True:
            with self._lock:
                active = self._active
                stop_at = self._stop_at
                writer = self._writer

            if not active:
                time.sleep(0.05)
                continue

            if time.time() >= stop_at:
                with self._lock:
                    writer = self._writer
                    done_path = self._video_path
                    self._writer = None
                    self._video_path = None
                    self._stop_at = 0.0
                    self._active = False

                if writer is not None:
                    writer.release()

                log.info("Video recording finished: %s", done_path)
                continue

            frame = self.capture.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            add_timestamp(frame)

            with self._lock:
                writer = self._writer

            if writer is not None:
                writer.write(frame)

            time.sleep(frame_interval)


def draw_box(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    top_y = max(y1 - th - 10, 0)
    cv2.rectangle(frame, (x1, top_y), (x1 + tw + 6, y1), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 3, max(y1 - 5, th + 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )


def add_timestamp(frame):
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(
        frame,
        ts,
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )


def save_snapshot(frame: np.ndarray, prefix: str) -> str:
    Path(config.SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{ts}.jpg"
    path = os.path.abspath(os.path.join(config.SNAPSHOT_DIR, filename))
    ok = cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError(f"Failed to write snapshot: {path}")
    log.info("Snapshot saved: %s", path)
    return path


def ensure_face_model_files():
    FACE_DIR.mkdir(parents=True, exist_ok=True)
    if not FACE_PROTO.exists():
        log.info("Downloading OpenCV face prototxt…")
        urllib.request.urlretrieve(FACE_PROTO_URL, FACE_PROTO)
    if not FACE_MODEL.exists():
        log.info("Downloading OpenCV face caffemodel…")
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL)


def load_face_detector():
    ensure_face_model_files()
    net = cv2.dnn.readNetFromCaffe(str(FACE_PROTO), str(FACE_MODEL))
    return net


def detect_faces_dnn(face_net, frame, min_confidence=0.65):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123))
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []

    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < min_confidence:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
        if x2 <= x1 or y2 <= y1:
            continue

        faces.append((x1, y1, x2, y2, conf))

    return faces


def load_yolo_detector():
    log.info("Loading YOLO model yolo11n_ncnn_model…")
    return YOLO("/home/pi/telegram_porch_detector/yolo11n_ncnn_model", task="detect")


def main():
    validate_config()

    log.info("═" * 60)
    log.info("  Porch Detector starting up (OpenCV DNN face + YOLO11)")
    log.info("═" * 60)

    face_net = load_face_detector()
    yolo = load_yolo_detector()

    sender = TelegramSender()
    sender.start()

    capture = FrameCapture(config.RTSP_URL)
    capture.start()

    recorder = VideoRecorder(capture)
    recorder.start()

    cleaner = MediaCleaner(sender)
    cleaner.start()

    cooldown = Cooldown()

    log.info("Waiting for first camera frame…")
    while capture.get_frame() is None:
        time.sleep(0.5)

    log.info("Detection loop running at %d FPS ✓", config.TARGET_FPS)

    while True:
        t_start = time.time()

        frame = capture.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        # Permanently rotate 90 degrees clockwise before all detection
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        h, w = frame.shape[:2]
        canvas = frame.copy()

        # Smaller frame for speed
        target_w = 640
        scale = target_w / w if w > target_w else 1.0
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale != 1.0 else frame.copy()

        trigger_person = False
        trigger_face = False
        trigger_package = False

        # Face detection on the resized frame (independent of person detection)
        faces = detect_faces_dnn(
            face_net,
            small,
            min_confidence=float(getattr(config, "FACE_CONF", 0.65)),
        )
        for x1s, y1s, x2s, y2s, conf in faces:
            x1 = int(x1s / scale)
            y1 = int(y1s / scale)
            x2 = int(x2s / scale)
            y2 = int(y2s / scale)
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)

            if (x2 - x1) < config.MIN_FACE_SIZE_PX or (y2 - y1) < config.MIN_FACE_SIZE_PX:
                continue

            draw_box(canvas, x1, y1, x2, y2, f"Face {conf:.2f}", (0, 220, 0))
            trigger_face = True
            log.info("Face detected — %dx%d px conf=%.2f", x2 - x1, y2 - y1, conf)

        # YOLO person/package on the resized frame
        results = yolo(
            small,
            classes=YOLO_CLASSES,
            conf=float(getattr(config, "YOLO_CONF", 0.40)),
            imgsz=640,
            verbose=False,
        )

        for result in results:
            boxes = result.boxes
            names = result.names
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1s, y1s, x2s, y2s = [int(v) for v in box.xyxy[0].tolist()]

                x1 = int(x1s / scale)
                y1 = int(y1s / scale)
                x2 = int(x2s / scale)
                y2 = int(y2s / scale)
                x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
                label_name = names.get(cls_id, str(cls_id))

                if cls_id == 0:
                    box_h = y2 - y1
                    height_ratio = box_h / h

                    if height_ratio < config.MIN_PERSON_HEIGHT_RATIO:
                        continue

                    # Ignore person detections whose box center falls in the bottom
                    # configured portion of the rotated upright frame.
                    ignore_bottom_ratio = float(
                        getattr(config, "PERSON_IGNORE_BOTTOM_RATIO", 0.20)
                    )
                    ignore_y = int(h * (1.0 - ignore_bottom_ratio))
                    center_y = int((y1 + y2) / 2)

                    if center_y >= ignore_y:
                        log.info(
                            "Rejected person candidate — center in bottom blind spot "
                            "center_y=%d ignore_y=%d conf=%.2f",
                            center_y,
                            ignore_y,
                            conf,
                        )
                        continue

                    draw_box(canvas, x1, y1, x2, y2, f"Person {conf:.2f}", (255, 80, 0))
                    trigger_person = True
                    log.info(
                        "Person detected — box height %.0f%% conf=%.2f center_y=%d",
                        height_ratio * 100,
                        conf,
                        center_y,
                    )

                else:
                    # backpack/suitcase are only rough package proxies
                    area_ratio = ((x2 - x1) * (y2 - y1)) / float(w * h)
                    if area_ratio < 0.01:
                        continue

                    draw_box(canvas, x1, y1, x2, y2, f"{label_name.title()} {conf:.2f}", (0, 140, 255))
                    trigger_package = True
                    log.info("Package-proxy detected — %s conf=%.2f", label_name, conf)

        add_timestamp(canvas)

        video_reason = None
        if trigger_face:
            video_reason = "face"
        elif trigger_person:
            video_reason = "person"
        elif trigger_package:
            video_reason = "package"

        if video_reason is not None:
            recorder.start_recording(video_reason)

        if trigger_face and cooldown.ready("face", config.FACE_COOLDOWN_SEC):
            ts_str = datetime.now().strftime("%d %b %Y  %I:%M:%S %p")
            caption = f"🙂 *Face detected at porch!*\n{ts_str}"
            image_path = save_snapshot(canvas, "face")
            sender.enqueue(image_path, caption)

        if trigger_person and cooldown.ready("person", config.PERSON_COOLDOWN_SEC):
            ts_str = datetime.now().strftime("%d %b %Y  %I:%M:%S %p")
            caption = f"🚨 *Person detected at porch!*\n{ts_str}"
            image_path = save_snapshot(canvas, "person")
            sender.enqueue(image_path, caption)

        if trigger_package and cooldown.ready("package", config.PACKAGE_COOLDOWN_SEC):
            ts_str = datetime.now().strftime("%d %b %Y  %I:%M:%S %p")
            caption = (
                "📦 *Package-like item detected at porch!*\n"
                "_(backpack/suitcase proxy)_\n"
                f"{ts_str}"
            )
            image_path = save_snapshot(canvas, "package")
            sender.enqueue(image_path, caption)

        elapsed = time.time() - t_start
        time.sleep(max(0.0, FRAME_INTERVAL - elapsed))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Stopped by user.")
