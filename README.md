# TP-LINK Tapo Camera Telegram Alerts

Local porch monitoring for TP-LINK Tapo cameras with RTSP enabled, designed for a Raspberry Pi 4 and above and Telegram-based photo and video alert delivery.

This project watches a local RTSP camera stream, runs local face and object detection, saves annotated snapshots, records post-detection video clips, and sends photo, video, and stream-health alerts to Telegram without any paid cloud service.

Tested setup:
- Raspberry Pi 4 with 4 GB RAM
- Raspberry Pi OS Bookworm or newer
- Tapo camera models that support RTSP streaming
- Stable operation tested up to 5 FPS; above that, Raspberry Pi 4 thermal throttling became an issue during longer runs, can be solved with a fan cooling case.

## What It Does

- Connects to a Tapo RTSP stream with OpenCV
- Detects faces with the OpenCV DNN face detector
- Detects people with YOLO11n NCNN
- Detects delivery packages with a custom YOLO11n package model
- Sends annotated photo alerts and detection videos to Telegram
- Supports standard chats and Telegram forum topic threads
- Keeps a configurable in-memory pre-roll buffer so alert videos can include a few seconds before the trigger
- Records a configurable MP4 clip after the first trigger in a burst
- Uploads the finished MP4 detection video to Telegram after recording completes
- Applies separate per-category 60-second alert cooldowns for face, person, and package alerts
- Keeps separate 24-hour remembered-box position memory for face, person, and package detections
- Uses a configurable pixel tolerance to decide whether a new box is really a new event or the same stationary detection
- Keeps the single-recording behavior: once recording starts, additional triggers do not start another video until the current one finishes
- Adds timestamps to saved images and recorded video
- Retries Telegram uploads automatically with backoff if the network is down
- Reconnects to the camera stream automatically if RTSP drops
- Sends a delayed Telegram text alert if the camera stream stays offline for too long
- Can send a Telegram recovery message when the RTSP stream comes back
- Cleans up old snapshots and videos automatically
- Includes helper scripts to test Telegram delivery and fetch your chat ID

## Detection Types

- Face detection
- Person detection
- Package detection using a custom one-class model

## Detection Models

- Face detection uses the OpenCV DNN Caffe SSD face detector
- Person detection uses YOLO11n exported in NCNN format
- Package detection uses a custom YOLO11n model trained from pretrained weights on the Roboflow `package-at-front-door` dataset in YOLO format:
  https://universe.roboflow.com/package-detection/package-at-front-door
- The dataset YAML defines a single class, `package`
- Training improved steadily before early stopping at epoch 65, with the best checkpoint saved at epoch 50
- Validation for the best package model reached approximately Precision `0.882`, Recall `0.863`, mAP@50 `0.921`, and mAP@50-95 `0.739`
- NCNN is a good fit for Raspberry Pi because it is lightweight and optimized for edge-style inference workloads

## Project Files

- `detector.py`: Main detector and alerting loop
- `config.py`: Camera, Telegram, cooldown, and storage settings
- `start.sh`: Launcher script intended for Raspberry Pi deployment
- `telegram_test.py`: Sends a test message and optional test photo to Telegram
- `get_chat_id.py`: Prints chat IDs seen by your Telegram bot
- `temp_logger.sh`: Optional Pi temperature logger
- `yolo11n_ncnn_model/`: YOLO11 NCNN export used for person detection at runtime
- `package_yolo11n_ncnn_model/`: Custom YOLO11 NCNN export used for package detection at runtime

## Deployment Path

The current scripts are written for this install path:

```text
/home/pi/telegram_porch_detector
```

If you want to use the code exactly as it is in this repository, clone it to that location on the Pi.

The detector also writes media to:

```text
/home/pi/porch_snapshots
/home/pi/porch_videos
```

## Requirements

- Python 3
- A Telegram bot token from `@BotFather`
- Your Telegram chat ID
- RTSP enabled on the Tapo camera
- `cv2` available to the Python environment used for the detector

`requirements.txt` contains the Python packages used directly by the project:

```text
requests
ultralytics
```

You also need OpenCV available for Python. On Raspberry Pi OS Bookworm, there are two common approaches:

1. Install `opencv-python` into the virtual environment
2. Install `python3-opencv` on the system and create the venv with access to system site packages

## Camera Preparation

Before using the detector, create a Camera Account in the Tapo app and assign a username and password for the camera's RTSP access.

Those same camera-account credentials are what you place in the RTSP URL in `config.py`, for example:

```text
rtsp://username:password@camera_ip_address:554/stream2
```

Tapo cameras typically expose multiple RTSP streams. In most setups:

- `stream1` is the higher-quality stream
- `stream2` is the lower-quality stream and is lighter on the Raspberry Pi

This code has been tested with `stream2`.

## Setup

Clone the repository on the Pi to the expected path:

```bash
cd /home/pi
git clone https://github.com/saad-git-007/tapo-camera-telegram-alerts.git telegram_porch_detector
cd telegram_porch_detector
```

Create the virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Make sure OpenCV is available in that same Python environment before running the detector.

## Configure

Edit `config.py` and set:

- `RTSP_URL`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `TELEGRAM_MESSAGE_THREAD_ID` if you use a Telegram topic

You can also tune:

- Camera disconnect alert delay and optional recovery alert
- Face confidence and minimum face size
- YOLO confidence
- Package model path
- Package confidence and minimum package size
- Face, person, and package 60-second alert cooldowns
- Face, person, and package stationary-memory cooldowns
- Alert position tolerance in pixels
- Person height threshold
- Person bottom blind-spot filter
- Snapshot and video retention
- Detection loop FPS, pre-roll length, and post-trigger video recording length

The default detector loop in `config.py` is conservative for Raspberry Pi use. This project has been tested up to 5 FPS on a Pi 4 with 4 GB RAM. Higher values increased heat and led to thermal throttling in extended runs.

## Camera Stream Health Alerts

- If the RTSP stream stays offline longer than `CAMERA_DISCONNECT_ALERT_DELAY_SEC`, the detector sends a Telegram text alert about the outage
- Only one offline message is sent for each continuous disconnect event
- If `CAMERA_SEND_RECOVERY_ALERT` is enabled, the detector sends a recovery message after the stream comes back
- These stream-health alerts are separate from face, person, and package detection alerts

## Alert Suppression Logic

- Face, person, and package detections are tracked independently, so one category does not block alerts in another
- Each category has its own short alert cooldown, currently 60 seconds by default
- Each category also keeps a remembered list of recently alerted bounding boxes for 24 hours by default
- If the same category appears again in roughly the same position and size, within the configured pixel tolerance, it is treated as the same stationary event and a new Telegram alert is suppressed
- If that category appears in a meaningfully different position, it can trigger again immediately as long as its short cooldown has expired
- The remembered-box logic helps reduce repeated nuisance alerts from stationary faces, people, or packages while still allowing new activity elsewhere in the frame to alert normally

## Telegram Setup

1. Open Telegram and search for `@BotFather`
2. Start a chat with `@BotFather` and run the `/newbot` command
3. Follow the prompts to give your bot a display name and a unique bot username
4. BotFather will return a bot token that looks similar to:

```text
123456789:AAExampleTokenHere
```

5. Copy that token into `TELEGRAM_BOT_TOKEN` in `config.py`
6. Send a message to your new bot from the Telegram chat where you want alerts to arrive
7. Run:

```bash
python3 get_chat_id.py
```

8. Copy the numeric chat ID printed by the script into `TELEGRAM_CHAT_ID` in `config.py`
9. If you are sending alerts into a Telegram forum topic, also set `TELEGRAM_MESSAGE_THREAD_ID`
10. Optionally test messaging with:

```bash
python3 telegram_test.py
```

## Run

Start the detector with:

```bash
./start.sh
```

The launcher will:

- Create the snapshot and video directories if needed
- Stop an older detector instance
- Set OpenMP limits to reduce CPU heat on the Pi
- Start `detector.py` and append logs to `detector.log`

## Notes

- The camera frame is rotated 90 degrees clockwise before detection and recording. This was done for a Tapo camera mounted in portrait orientation. If your camera is mounted in landscape orientation, comment out the two `cv2.rotate(..., cv2.ROTATE_90_CLOCKWISE)` calls in `detector.py`
- Face model files are downloaded automatically on first run if they are missing
- Package detection uses a custom one-class YOLO11n model trained for front-door package detection
- Detection videos are sent to Telegram after the recording finishes, not at the instant the recording starts
- Video pre-roll is kept in memory, not continuously written to disk, so the detector can prepend a few seconds before the trigger without recording full-time video files
- Camera outage alerts are delayed and one-time per continuous disconnect, so a long Wi-Fi dropout does not spam Telegram repeatedly
- `temp_logger.sh` is optional and intended for Raspberry Pi systems that provide `vcgencmd`

## Suggested Pi Usage

If you want the detector to start automatically on boot, run it from a systemd service or a supervised startup script on the Pi. The included `start.sh` is a good entry point for that setup.
