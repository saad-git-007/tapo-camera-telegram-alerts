#!/bin/bash
LOGFILE="/home/pi/telegram_porch_detector/temp_log.txt"

TEMP_RAW=$(vcgencmd measure_temp | grep -o '[0-9]*\.[0-9]*')
TEMP_INT=${TEMP_RAW%.*}

if [ "$TEMP_INT" -ge 80 ]; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') | Temp: ${TEMP_RAW}C" >> "$LOGFILE"
fi
