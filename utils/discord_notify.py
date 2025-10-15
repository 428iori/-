import requests, os

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

def send_discord_message(content):
    if not DISCORD_WEBHOOK_URL:
        print("[discord] webhook not set, skipping")
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": content})
        print("[discord] sent")
    except Exception as e:
        print(f"[discord] error: {e}")
