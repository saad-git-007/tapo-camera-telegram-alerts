"""
Message your bot first in Telegram, then run this script.

It fetches recent updates and prints any chat IDs it sees so you can copy the
right TELEGRAM_CHAT_ID into config.py.
"""

import requests
import config

url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/getUpdates"

r = requests.get(url, params={"timeout": 10}, timeout=20)
r.raise_for_status()

payload = r.json()
print("Telegram API ok:", payload.get("ok"))
print()

if not payload.get("result"):
    print("No updates found.")
    print("Open Telegram, search for your bot, press Start or send 'hi', then run this again.")
    raise SystemExit(0)

seen = set()

for upd in payload["result"]:
    msg = upd.get("message") or upd.get("edited_message") or upd.get("channel_post")
    if not msg:
        continue

    chat = msg.get("chat", {})
    chat_id = chat.get("id")
    title = chat.get("title") or chat.get("username") or chat.get("first_name") or "(unknown)"
    chat_type = chat.get("type")
    text = msg.get("text", "")

    key = (chat_id, chat_type, title)
    if key in seen:
        continue
    seen.add(key)

    print(f"chat_id={chat_id}   type={chat_type}   name={title}")
    if text:
        print(f"  sample text: {text}")
    print()
