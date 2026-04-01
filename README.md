# TP-LINK Tapo Camera Telegram Alerts

Local porch monitoring for TP-LINK Tapo cameras with RTSP enabled, designed for a Raspberry Pi 4 and above and Telegram-based photo alert delivery.

This project watches a local RTSP camera stream, runs local face and object detection, saves annotated snapshots, records after detection video clips, and sends photo alerts to Telegram without any paid cloud service.

Tested setup:
- Raspberry Pi 4 with 4 GB RAM
- Raspberry Pi OS Bookworm or newer
- Tapo camera models that support RTSP streaming
- Stable operation tested up to 5 FPS; above that, Raspberry Pi 4 thermal throttling became an issue during longer runs, can be solved with a fan cooling case.

## What It Does

- Connects to a Tapo RTSP stream with OpenCV
- Detects faces with the OpenCV DNN face detector
- Detects people and delivery packages with YOLO11n NCNN
- Uses backpack and suitcase detections as package-like proxies
- Sends annotated photo alerts to Telegram
- Supports standard chats and Telegram forum topic threads
- Records a configurable MP4 clip after the first trigger in a burst
- Applies separate cooldowns for face, person, and package alerts
- Adds timestamps to saved images and recorded video
- Retries Telegram uploads automatically with backoff if the network is down
- Reconnects to the camera stream automatically if RTSP drops
- Cleans up old snapshots and videos automatically
- Includes helper scripts to test Telegram delivery and fetch your chat ID

## Detection Types

- Face detection
- Person detection
- Package-like detection using `backpack` and `suitcase` classes as practical proxies

## Detection Models

- Face detection uses the OpenCV DNN Caffe SSD face detector
- Person and package-like detection use YOLO11n exported in NCNN format
- NCNN is a good fit for Raspberry Pi because it is lightweight and optimized for edge-style inference workloads

## Project Files

- `detector.py`: Main detector and alerting loop
- `config.py`: Camera, Telegram, cooldown, and storage settings
- `start.sh`: Launcher script intended for Raspberry Pi deployment
- `telegram_test.py`: Sends a test message and optional test photo to Telegram
- `get_chat_id.py`: Prints chat IDs seen by your Telegram bot
- `temp_logger.sh`: Optional Pi temperature logger
- `yolo11n_ncnn_model/`: YOLO11 NCNN export used by the detector at runtime

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

- Face confidence and minimum face size
- YOLO confidence
- Person height threshold
- Person bottom blind-spot filter
- Cooldowns
- Snapshot and video retention
- Detection loop FPS and video recording length

The default detector loop in `config.py` is conservative for Raspberry Pi use. This project has been tested up to 5 FPS on a Pi 4 with 4 GB RAM. Higher values increased heat and led to thermal throttling in extended runs.

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

- The camera frame is rotated 90 degrees clockwise before detection and recording
- Face model files are downloaded automatically on first run if they are missing
- Package alerts are heuristic alerts, not parcel-specific recognition
- `temp_logger.sh` is optional and intended for Raspberry Pi systems that provide `vcgencmd`

## Suggested Pi Usage

If you want the detector to start automatically on boot, run it from a systemd service or a supervised startup script on the Pi. The included `start.sh` is a good entry point for that setup.
