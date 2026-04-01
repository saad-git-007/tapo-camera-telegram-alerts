"""
Sends a test Telegram text message and an optional photo to verify your setup.
"""

import os
import requests
import config

base = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}"

# 1) text test
r = requests.post(
    f"{base}/sendMessage",
    data={
        "chat_id": str(config.TELEGRAM_CHAT_ID),
        "text": "✅ Porch detector Telegram test message",
    },
    timeout=20,
)
print("sendMessage:", r.status_code, r.text)

# 2) optional image test
test_image = os.path.join(config.SNAPSHOT_DIR, "telegram_test.jpg")
if os.path.exists(test_image):
    with open(test_image, "rb") as f:
        r = requests.post(
            f"{base}/sendPhoto",
            data={
                "chat_id": str(config.TELEGRAM_CHAT_ID),
                "caption": "✅ Porch detector Telegram photo test",
            },
            files={"photo": ("telegram_test.jpg", f, "image/jpeg")},
            timeout=(10, 60),
        )
    print("sendPhoto:", r.status_code, r.text)
else:
    print(f"No test image found at {test_image}")
