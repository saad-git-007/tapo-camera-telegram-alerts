#!/bin/bash
set -e

SCRIPT_DIR="/home/pi/telegram_porch_detector"
PYTHON="$SCRIPT_DIR/venv/bin/python3"
SNAPSHOT_DIR="/home/pi/porch_snapshots"
VIDEO_DIR="/home/pi/porch_videos"

echo "──────────────────────────────────────────────"
echo "  Porch Detector — Starting (Telegram edition)"
echo "──────────────────────────────────────────────"

if [ ! -x "$PYTHON" ]; then
    echo "Python venv not found at: $PYTHON"
    echo
    echo "Create it first with:"
    echo "  cd $SCRIPT_DIR"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

mkdir -p "$SNAPSHOT_DIR" "$VIDEO_DIR"

cd "$SCRIPT_DIR"

# Reduce NCNN/OpenMP CPU heat a bit on Pi 4
export OMP_NUM_THREADS=2
export OMP_THREAD_LIMIT=2
export OMP_WAIT_POLICY=PASSIVE

# Stop any old detector instance first
pkill -f "$SCRIPT_DIR/detector.py" 2>/dev/null || true
sleep 1

echo "Starting detector..."
echo "──────────────────────────────────────────────"

exec "$PYTHON" "$SCRIPT_DIR/detector.py"
