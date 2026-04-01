# Tapo Camera Telegram Alerts

Local porch monitoring for Tapo cameras with RTSP enabled, designed for a Raspberry Pi 4 and Telegram-based alert delivery.

This project watches an RTSP stream, runs local face and object detection, saves annotated snapshots, records follow-up video clips, and sends photo alerts to Telegram without any paid cloud service.

Tested setup:
- Raspberry Pi 4 with 4 GB RAM
- Raspberry Pi OS Bookworm
- Tapo camera models that support RTSP streaming

## What It Does

- Connects to a Tapo RTSP stream with OpenCV
- Detects faces with the OpenCV DNN face detector
- Detects people, backpacks, and suitcases with YOLO11 NCNN
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

## Setup

Clone the repository on the Pi to the expected path:

```bash
cd /home/pi
git clone https://github.com/YOUR_USERNAME/tapo-camera-telegram-alerts.git telegram_porch_detector
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

## Telegram Setup

1. Create a Telegram bot with `@BotFather`
2. Send a message to the bot from the chat where you want alerts
3. Run:

```bash
python3 get_chat_id.py
```

4. Copy the numeric chat ID into `config.py`
5. Optionally test messaging with:

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
