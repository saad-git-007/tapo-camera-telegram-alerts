# ═══════════════════════════════════════════════════════════════
#  config.py — fill in ALL values before running
# ═══════════════════════════════════════════════════════════════

# ── Camera ──────────────────────────────────────────────────────
RTSP_URL = 'rtsp://username:password@camera_static_ip_address:554/stream2'

# ── Telegram bot alerts ─────────────────────────────────────────
# Create a bot with @BotFather and paste the token here.
TELEGRAM_BOT_TOKEN = ""

# Numeric chat ID to receive alerts.
# Use get_chat_id.py after messaging your bot once.
TELEGRAM_CHAT_ID = "12345678"

# Optional: topic/thread ID if sending into a forum topic in a group.
# Set to None for a normal private chat or normal group chat.
TELEGRAM_MESSAGE_THREAD_ID = None

# True = silent notification, False = normal notification sound.
TELEGRAM_DISABLE_NOTIFICATION = False

# Max retry backoff when internet/Telegram is temporarily unavailable.
TELEGRAM_MAX_RETRY_BACKOFF_SEC = 300

# ── Detection tuning ─────────────────────────────────────────────
# FACE detection — OpenCV DNN (Caffe SSD)
# Minimum face width/height in pixels to trigger an alert.
MIN_FACE_SIZE_PX = 50

# Minimum confidence for face detection.
FACE_CONF = 0.68

# PERSON detection — YOLO11n NCNN (class 0)
# Minimum fraction of frame height the person bounding box must occupy.
MIN_PERSON_HEIGHT_RATIO = 0.25

# Blind spot: ignore person detections whose box center falls in the bottom
# 20% of the rotated upright frame.
PERSON_IGNORE_BOTTOM_RATIO = 0.20

# YOLO confidence for person / backpack / suitcase detections.
YOLO_CONF = 0.40

# ── Alert logic ─────────────────────────────────────────────────
FACE_COOLDOWN_SEC = 60
PERSON_COOLDOWN_SEC = 60
PACKAGE_COOLDOWN_SEC = 60

# ── Performance ────────────────────────────────────────────────
TARGET_FPS = 3

# ── Snapshots ──────────────────────────────────────────────────
SNAPSHOT_DIR = "/home/pi/porch_snapshots"
SNAPSHOT_MAX_AGE_SEC = 3600

# ── Videos ─────────────────────────────────────────────────────
VIDEO_DIR = "/home/pi/porch_videos"
VIDEO_RECORD_SECONDS = 60
VIDEO_FPS = 30 #15 fine
VIDEO_MAX_AGE_SEC = 140 * 3600
