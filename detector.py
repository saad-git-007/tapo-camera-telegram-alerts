"""
Porch Security Detector — Telegram Edition
==========================================

Detection triggers:
  1. OpenCV DNN face detector (Caffe SSD)
  2. Ultralytics YOLO11 NCNN for person
  3. Ultralytics YOLO11 NCNN custom package model

Alert delivery:
  - Annotated JPEG saved to SNAPSHOT_DIR
  - Telegram bot uploads the local JPEG directly using sendPhoto
  - 1-minute MP4 video recorded after the first trigger in a burst
"""

from __future__ import annotations

from dataclasses import dataclass
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
PERSON_CLASSES = [0]
Box = tuple[int, int, int, int]


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


@dataclass
class RememberedBox:
    box: Box
    alerted_at: float


class AlertCooldown:
    def __init__(self):
        self._last_sent: dict[str, float] = {}

    def ready(self, key: str, seconds: float, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        last_sent = self._last_sent.get(key, 0.0)
        if now - last_sent < float(seconds):
            return False
        self._last_sent[key] = now
        return True


class StationaryAlertMemory:
    def __init__(self, label: str, tolerance_px: int, stationary_cooldown_sec: float):
        self.label = label
        self.tolerance_px = int(tolerance_px)
        self.stationary_cooldown_sec = float(stationary_cooldown_sec)
        self._remembered: list[RememberedBox] = []

    def _prune_expired(self, now: float):
        self._remembered = [
            entry
            for entry in self._remembered
            if now - entry.alerted_at < self.stationary_cooldown_sec
        ]

    def _same_position(self, box_a: Box, box_b: Box) -> bool:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        aw = ax2 - ax1
        ah = ay2 - ay1
        bw = bx2 - bx1
        bh = by2 - by1

        acx = ax1 + (aw / 2.0)
        acy = ay1 + (ah / 2.0)
        bcx = bx1 + (bw / 2.0)
        bcy = by1 + (bh / 2.0)

        tol = self.tolerance_px
        return (
            abs(acx - bcx) <= tol
            and abs(acy - bcy) <= tol
            and abs(aw - bw) <= tol
            and abs(ah - bh) <= tol
        )

    def new_boxes(self, boxes: list[Box], now: float | None = None) -> list[Box]:
        now = time.time() if now is None else now
        self._prune_expired(now)

        fresh_boxes = []
        for box in boxes:
            if any(self._same_position(box, entry.box) for entry in self._remembered):
                continue
            if any(self._same_position(box, fresh_box) for fresh_box in fresh_boxes):
                continue
            fresh_boxes.append(box)
        return fresh_boxes

    def remember_boxes(self, boxes: list[Box], now: float | None = None):
        now = time.time() if now is None else now
        for box in self.new_boxes(boxes, now):
            self._remembered.append(RememberedBox(box=box, alerted_at=now))


class TelegramSender(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._pq = queue.PriorityQueue()
        self._counter = itertools.count()
        self._session = requests.Session()
        self._pending = set()
        self._pending_lock = threading.Lock()

    def enqueue_photo(self, image_path: str, caption: str):
        self._enqueue_media("photo", image_path, caption)

    def enqueue_video(self, video_path: str, caption: str):
        self._enqueue_media("video", video_path, caption)

    def _enqueue_media(self, media_kind: str, media_path: str, caption: str):
        media_path = os.path.abspath(media_path)
        with self._pending_lock:
            self._pending.add(media_path)
        self._pq.put((time.time(), next(self._counter), media_kind, media_path, caption, 0))
        log.info(
            "%s queued → Telegram: %s",
            media_kind.title(),
            caption.replace("\n", " | "),
        )

    def is_pending(self, media_path: str) -> bool:
        media_path = os.path.abspath(media_path)
        with self._pending_lock:
            return media_path in self._pending

    def _mark_done(self, media_path: str):
        media_path = os.path.abspath(media_path)
        with self._pending_lock:
            self._pending.discard(media_path)

    def run(self):
        while True:
            next_try, _, media_kind, media_path, caption, attempt = self._pq.get()

            wait = next_try - time.time()
            if wait > 0:
                time.sleep(wait)

            if not os.path.exists(media_path):
                log.warning(
                    "%s missing before Telegram send, dropping: %s",
                    media_kind.title(),
                    media_path,
                )
                self._mark_done(media_path)
                continue

            if self._send_once(media_kind, media_path, caption):
                self._mark_done(media_path)
                continue

            attempt += 1
            backoff = min(5 * (2 ** (attempt - 1)), config.TELEGRAM_MAX_RETRY_BACKOFF_SEC)
            retry_at = time.time() + backoff
            log.warning(
                "Telegram send failed — retry #%d in %ds for %s",
                attempt,
                backoff,
                os.path.basename(media_path),
            )
            self._pq.put(
                (retry_at, next(self._counter), media_kind, media_path, caption, attempt)
            )

    def _send_once(self, media_kind: str, media_path: str, caption: str) -> bool:
        if media_kind == "photo":
            endpoint = "sendPhoto"
            field_name = "photo"
            mime_type = "image/jpeg"
            timeout = (10, 90)
        elif media_kind == "video":
            endpoint = "sendVideo"
            field_name = "video"
            mime_type = "video/mp4"
            timeout = (10, 300)
        else:
            log.warning("Unknown Telegram media kind: %s", media_kind)
            return False

        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/{endpoint}"

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
            with open(media_path, "rb") as f:
                files = {field_name: (os.path.basename(media_path), f, mime_type)}
                r = self._session.post(url, data=data, files=files, timeout=timeout)

            if r.status_code != 200:
                log.warning("Telegram HTTP %s: %s", r.status_code, r.text[:500])
                return False

            payload = r.json()
            if not payload.get("ok"):
                log.warning("Telegram API error: %s", payload)
                return False

            message_id = payload.get("result", {}).get("message_id")
            log.info(
                "Telegram sent ✓ kind=%s message_id=%s file=%s",
                media_kind,
                message_id,
                os.path.basename(media_path),
            )
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
                    if now - os.path.getmtime(f) > config.SNAPSHOT_MAX_AGE_SEC:
                        os.remove(f)

                video_pattern = os.path.join(config.VIDEO_DIR, "*.mp4")
                for f in glob.glob(video_pattern):
                    f = os.path.abspath(f)
                    if self.sender.is_pending(f):
                        continue
                    if now - os.path.getmtime(f) > config.VIDEO_MAX_AGE_SEC:
                        os.remove(f)

            except Exception as e:
                log.warning("Media cleanup error: %s", e)


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
    def __init__(self, capture: FrameCapture, sender: TelegramSender):
        super().__init__(daemon=True)
        self.capture = capture
        self.sender = sender
        self._lock = threading.Lock()
        self._writer = None
        self._video_path = None
        self._stop_at = 0.0
        self._active = False
        self._telegram_caption = None

    def start_recording(self, reason: str, telegram_caption: str) -> bool:
        with self._lock:
            if self._active:
                log.info("Video recording already active, skipping new trigger: %s", reason)
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
            self._telegram_caption = telegram_caption

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
                    telegram_caption = self._telegram_caption
                    self._writer = None
                    self._video_path = None
                    self._stop_at = 0.0
                    self._active = False
                    self._telegram_caption = None
                if writer is not None:
                    writer.release()
                log.info("Video recording finished: %s", done_path)
                if done_path and telegram_caption:
                    self.sender.enqueue_video(done_path, telegram_caption)
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
    path = os.path.abspath(os.path.join(config.SNAPSHOT_DIR, f"{prefix}_{ts}.jpg"))
    ok = cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError(f"Failed to write snapshot: {path}")
    log.info("Snapshot saved: %s", path)
    return path


def ensure_face_model_files():
    FACE_DIR.mkdir(parents=True, exist_ok=True)
    if not FACE_PROTO.exists():
        urllib.request.urlretrieve(FACE_PROTO_URL, FACE_PROTO)
    if not FACE_MODEL.exists():
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL)


def load_face_detector():
    ensure_face_model_files()
    return cv2.dnn.readNetFromCaffe(str(FACE_PROTO), str(FACE_MODEL))


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
        if x2 > x1 and y2 > y1:
            faces.append((x1, y1, x2, y2, conf))

    return faces


def load_person_detector():
    log.info("Loading person model yolo11n_ncnn_model…")
    return YOLO("/home/pi/telegram_porch_detector/yolo11n_ncnn_model", task="detect")


def load_package_detector():
    log.info("Loading package model %s…", config.PACKAGE_MODEL_PATH)
    return YOLO(config.PACKAGE_MODEL_PATH, task="detect")


def build_photo_caption(kind: str, ts_str: str) -> str:
    if kind == "face":
        return f"🙂 *Face detected at porch!*\n{ts_str}"
    if kind == "person":
        return f"🚨 *Person detected at porch!*\n{ts_str}"
    if kind == "package":
        return f"📦 *Package detected at porch!*\n{ts_str}"
    raise ValueError(f"Unknown detection kind: {kind}")


def build_video_caption(kind: str, ts_str: str) -> str:
    if kind == "face":
        return f"🎥 *Face detection video at porch!*\n{ts_str}"
    if kind == "person":
        return f"🎥 *Person detection video at porch!*\n{ts_str}"
    if kind == "package":
        return f"🎥 *Package detection video at porch!*\n{ts_str}"
    raise ValueError(f"Unknown detection kind: {kind}")


def main():
    validate_config()

    log.info("═" * 60)
    log.info("  Porch Detector starting up (OpenCV DNN face + YOLO11 custom package)")
    log.info("═" * 60)

    face_net = load_face_detector()
    person_yolo = load_person_detector()
    package_yolo = load_package_detector()

    sender = TelegramSender()
    sender.start()

    capture = FrameCapture(config.RTSP_URL)
    capture.start()

    recorder = VideoRecorder(capture, sender)
    recorder.start()

    cleaner = MediaCleaner(sender)
    cleaner.start()

    alert_cooldown = AlertCooldown()
    position_tolerance_px = int(getattr(config, "ALERT_POSITION_TOLERANCE_PX", 40))
    face_memory = StationaryAlertMemory(
        "face",
        tolerance_px=position_tolerance_px,
        stationary_cooldown_sec=float(
            getattr(config, "FACE_STATIONARY_COOLDOWN_SEC", 24 * 3600)
        ),
    )
    person_memory = StationaryAlertMemory(
        "person",
        tolerance_px=position_tolerance_px,
        stationary_cooldown_sec=float(
            getattr(config, "PERSON_STATIONARY_COOLDOWN_SEC", 24 * 3600)
        ),
    )
    package_memory = StationaryAlertMemory(
        "package",
        tolerance_px=position_tolerance_px,
        stationary_cooldown_sec=float(
            getattr(config, "PACKAGE_STATIONARY_COOLDOWN_SEC", 24 * 3600)
        ),
    )

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

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        h, w = frame.shape[:2]
        canvas = frame.copy()

        target_w = 640
        scale = target_w / w if w > target_w else 1.0
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale != 1.0 else frame.copy()

        trigger_person = False
        trigger_face = False
        trigger_package = False
        face_boxes: list[Box] = []
        person_boxes: list[Box] = []
        package_boxes: list[Box] = []

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

            face_boxes.append((x1, y1, x2, y2))
            draw_box(canvas, x1, y1, x2, y2, f"Face {conf:.2f}", (0, 220, 0))
            trigger_face = True
            log.info("Face detected — %dx%d px conf=%.2f", x2 - x1, y2 - y1, conf)

        person_results = person_yolo(
            small,
            classes=PERSON_CLASSES,
            conf=float(getattr(config, "YOLO_CONF", 0.40)),
            imgsz=640,
            verbose=False,
        )

        for result in person_results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                conf = float(box.conf[0].item())
                x1s, y1s, x2s, y2s = [int(v) for v in box.xyxy[0].tolist()]

                x1 = int(x1s / scale)
                y1 = int(y1s / scale)
                x2 = int(x2s / scale)
                y2 = int(y2s / scale)
                x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)

                box_h = y2 - y1
                height_ratio = box_h / h
                if height_ratio < config.MIN_PERSON_HEIGHT_RATIO:
                    continue

                ignore_bottom_ratio = float(getattr(config, "PERSON_IGNORE_BOTTOM_RATIO", 0.20))
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

                person_boxes.append((x1, y1, x2, y2))
                draw_box(canvas, x1, y1, x2, y2, f"Person {conf:.2f}", (255, 80, 0))
                trigger_person = True
                log.info(
                    "Person detected — box height %.0f%% conf=%.2f center_y=%d",
                    height_ratio * 100,
                    conf,
                    center_y,
                )

        package_results = package_yolo(
            small,
            conf=float(getattr(config, "PACKAGE_CONF", 0.35)),
            imgsz=640,
            verbose=False,
        )

        for result in package_results:
            boxes = result.boxes
            names = result.names
            if boxes is None:
                continue

            for box in boxes:
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                x1s, y1s, x2s, y2s = [int(v) for v in box.xyxy[0].tolist()]

                x1 = int(x1s / scale)
                y1 = int(y1s / scale)
                x2 = int(x2s / scale)
                y2 = int(y2s / scale)
                x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)

                if (x2 - x1) < config.MIN_PACKAGE_SIZE_PX or (y2 - y1) < config.MIN_PACKAGE_SIZE_PX:
                    continue

                label_name = names.get(cls_id, "package")
                package_boxes.append((x1, y1, x2, y2))
                draw_box(canvas, x1, y1, x2, y2, f"{label_name.title()} {conf:.2f}", (0, 140, 255))
                trigger_package = True
                log.info("Package detected — %s conf=%.2f", label_name, conf)

        add_timestamp(canvas)

        new_face_boxes = face_memory.new_boxes(face_boxes, now=t_start)
        new_person_boxes = person_memory.new_boxes(person_boxes, now=t_start)
        new_package_boxes = package_memory.new_boxes(package_boxes, now=t_start)

        send_face_alert = bool(new_face_boxes) and alert_cooldown.ready(
            "face",
            float(getattr(config, "FACE_ALERT_COOLDOWN_SEC", 60)),
            now=t_start,
        )
        send_person_alert = bool(new_person_boxes) and alert_cooldown.ready(
            "person",
            float(getattr(config, "PERSON_ALERT_COOLDOWN_SEC", 60)),
            now=t_start,
        )
        send_package_alert = bool(new_package_boxes) and alert_cooldown.ready(
            "package",
            float(getattr(config, "PACKAGE_ALERT_COOLDOWN_SEC", 60)),
            now=t_start,
        )

        video_reason = None
        video_caption = None
        video_ts_str = datetime.now().strftime("%d %b %Y  %I:%M:%S %p")
        if send_face_alert:
            video_reason = "face"
        elif send_person_alert:
            video_reason = "person"
        elif send_package_alert:
            video_reason = "package"

        if video_reason is not None:
            video_caption = build_video_caption(video_reason, video_ts_str)
            recorder.start_recording(video_reason, video_caption)

        if send_face_alert:
            ts_str = datetime.now().strftime("%d %b %Y  %I:%M:%S %p")
            caption = build_photo_caption("face", ts_str)
            image_path = save_snapshot(canvas, "face")
            sender.enqueue_photo(image_path, caption)
            face_memory.remember_boxes(new_face_boxes, now=t_start)

        if send_person_alert:
            ts_str = datetime.now().strftime("%d %b %Y  %I:%M:%S %p")
            caption = build_photo_caption("person", ts_str)
            image_path = save_snapshot(canvas, "person")
            sender.enqueue_photo(image_path, caption)
            person_memory.remember_boxes(new_person_boxes, now=t_start)

        if send_package_alert:
            ts_str = datetime.now().strftime("%d %b %Y  %I:%M:%S %p")
            caption = build_photo_caption("package", ts_str)
            image_path = save_snapshot(canvas, "package")
            sender.enqueue_photo(image_path, caption)
            package_memory.remember_boxes(new_package_boxes, now=t_start)

        elapsed = time.time() - t_start
        time.sleep(max(0.0, FRAME_INTERVAL - elapsed))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Stopped by user.")
