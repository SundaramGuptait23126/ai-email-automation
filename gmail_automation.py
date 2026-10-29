"""
Autonomous Gmail Automation Engine powered by Custom SLM.
Improvements:
1. Start-Time Filter: Processes ONLY emails that arrived AFTER the bot started (Ignores all older unread emails).
2. Professional Typography & HTML Formatting:
   - High-quality styling (clean line-height, proper paragraphs, beautiful fonts)
   - Proper salutations and signature formatting
   - Both Multipart HTML and Plain-Text fallback for all email clients
3. Thread-safe in-reply-to delivery
4. Prevents duplicate replies with 'AI-Replied' tag
"""

import os
import sys
import base64
import time
import re
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from engine.email_engine import EmailAutomationEngine

# ==============================================================================
# CONFIGURATION
# ==============================================================================
AUTO_SEND = True
SAFETY_GUARD_ENABLED = False  # Set to True if high-risk emails should be Drafts instead

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send"
]

CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
TOKEN_FILE = PROJECT_ROOT / "token.json"


def get_gmail_service():
    """Authenticates and returns the Gmail API client."""
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    f"\n❌ 'credentials.json' not found at {CREDENTIALS_FILE}!\n"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def get_or_create_label(service, label_name="AI-Replied"):
    """Finds or creates a custom Gmail label."""
    try:
        results = service.users().labels().list(userId="me").execute()
        labels = results.get("labels", [])
        for l in labels:
            if l["name"].lower() == label_name.lower():
                return l["id"]

        label_body = {
            "name": label_name,
            "labelListVisibility": "labelShow",
            "messageListVisibility": "show"
        }
        created = service.users().labels().create(userId="me", body=label_body).execute()
        return created["id"]
    except Exception as e:
        print(f"Warning on label creation: {e}")
        return None


def parse_email_message(service, msg_id):
    """Fetches and parses email headers, timestamp, sender, subject, and body."""
    msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
    payload = msg.get("payload", {})
    headers = payload.get("headers", [])

    header_dict = {h["name"].lower(): h["value"] for h in headers}
    sender = header_dict.get("from", "")
    subject = header_dict.get("subject", "No Subject")
    message_id_header = header_dict.get("message-id", "")
    thread_id = msg.get("threadId", "")
    
    # Internal date in epoch seconds (exact time email reached Google servers)
    internal_date = int(msg.get("internalDate", 0)) // 1000

    # Extract body text
    body = ""
    if "parts" in payload:
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain" and "data" in part.get("body", {}):
                data = part["body"]["data"]
                body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                break
    elif "data" in payload.get("body", {}):
        data = payload["body"]["data"]
        body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")

    return {
        "id": msg_id,
        "thread_id": thread_id,
        "internal_date": internal_date,
        "sender": sender,
        "subject": subject,
        "message_id_header": message_id_header,
        "body": body.strip()
    }


def format_professional_email(raw_text):
    """
    Cleans raw text, organizes it into structured paragraphs,
    and returns both clean Plain-Text and beautifully styled HTML.
    """
    # 1. Clean up stray quotes, JSON residues, and escaped characters
    clean = raw_text.replace('\\n', '\n').replace('\\"', '"').strip('"\' \n\r\t')
    clean = re.sub(r'^[{\\"\'\s]*(Tribe Response:\s*|Response:\s*|Body:\s*)?', '', clean)
    clean = re.sub(r'[\\"\'\}]+\s*$', '', clean)

    # 2. Split into distinct logical paragraphs
    lines = [line.strip() for line in clean.split('\n') if line.strip()]
    
    paragraphs = []
    current_p = []

    for line in lines:
        # Detect greeting or signoff lines
        if re.match(r'^(Dear|Hi|Hello|Respected|Hey)\b', line, re.IGNORECASE):
            if current_p:
                paragraphs.append(" ".join(current_p))
                current_p = []
            paragraphs.append(line)
        elif re.match(r'^(Best regards|Warm regards|Regards|Sincerely|Thanks & Regards|Thank you)\b', line, re.IGNORECASE):
            if current_p:
                paragraphs.append(" ".join(current_p))
                current_p = []
            current_p.append(line)
        elif line.startswith("The ") and any(term in line.lower() for term in ["team", "hr", "support", "office"]):
            current_p.append(line)
            paragraphs.append("\n".join(current_p))
            current_p = []
        else:
            current_p.append(line)

    if current_p:
        paragraphs.append(" ".join(current_p))

    # 3. Create Plain Text version with double spacing
    plain_text = "\n\n".join(paragraphs)

    # 4. Create Styled HTML version
    html_paragraphs = []
    for p in paragraphs:
        if "\n" in p:
            # Multi-line sign-off (e.g. Best regards,\nHR Team)
            parts = p.split("\n")
            html_p = f'<p style="margin-top: 16px; margin-bottom: 4px; color: #374151;">{parts[0]}<br><strong style="color: #111827;">{parts[1] if len(parts) > 1 else ""}</strong></p>'
        elif re.match(r'^(Dear|Hi|Hello|Respected)\b', p, re.IGNORECASE):
            html_p = f'<p style="margin-bottom: 14px; font-weight: 500; color: #111827;">{p}</p>'
        else:
            html_p = f'<p style="margin-bottom: 14px; text-align: justify; line-height: 1.65; color: #374151;">{p}</p>'
        html_paragraphs.append(html_p)

    html_content = "".join(html_paragraphs)
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
          font-size: 14px;
          line-height: 1.65;
          color: #202124;
          background-color: #ffffff;
          margin: 0;
          padding: 12px;
        }}
      </style>
    </head>
    <body>
      <div style="max-width: 600px; padding: 4px 8px;">
        {html_content}
      </div>
    </body>
    </html>
    """

    return plain_text, html_template.strip()


def send_reply(service, email_info, raw_reply_text, auto_send=True):
    """Sends an in-thread, beautifully formatted Multipart (HTML + Plain) email."""
    plain_text, html_text = format_professional_email(raw_reply_text)

    # Create Multipart Container
    message = MIMEMultipart("alternative")
    message["to"] = email_info["sender"]
    message["subject"] = (
        email_info["subject"] if email_info["subject"].lower().startswith("re:")
        else f"Re: {email_info['subject']}"
    )

    if email_info["message_id_header"]:
        message["In-Reply-To"] = email_info["message_id_header"]
        message["References"] = email_info["message_id_header"]

    # Attach Plain-Text and HTML parts
    part1 = MIMEText(plain_text, "plain", "utf-8")
    part2 = MIMEText(html_text, "html", "utf-8")
    message.attach(part1)
    message.attach(part2)

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
    body_payload = {
        "raw": raw,
        "threadId": email_info["thread_id"]
    }

    if auto_send:
        sent = service.users().messages().send(userId="me", body=body_payload).execute()
        print(f"🚀 [AUTO-SENT] Professional HTML reply delivered to: {email_info['sender']}")
        return sent
    else:
        draft = service.users().drafts().create(userId="me", body={"message": body_payload}).execute()
        print(f"📝 [DRAFT CREATED] Saved formatted draft for: {email_info['sender']}")
        return draft


def process_inbox(service, engine, label_id, bot_start_time):
    """Scans for unread emails arrived strictly AFTER bot start time."""
    query = f"is:unread label:INBOX after:{bot_start_time - 30}"
    res = service.users().messages().list(userId="me", q=query, maxResults=10).execute()
    messages = res.get("messages", [])

    if not messages:
        return 0

    processed_count = 0

    for m in messages:
        msg_id = m["id"]
        email_info = parse_email_message(service, msg_id)

        # STRICT TIMESTAMP FILTER: Ignore any email that arrived before bot started
        if email_info["internal_date"] < bot_start_time:
            continue

        processed_count += 1
        received_time_str = datetime.fromtimestamp(email_info["internal_date"]).strftime("%I:%M:%S %p")

        print("\n" + "=" * 65)
        print(f"📨 New Email Received at: {received_time_str}")
        print(f"👤 From: {email_info['sender']}")
        print(f"📌 Subject: {email_info['subject']}")
        print("=" * 65)

        # 1. Run local SLM
        print("🤖 Running Custom Email SLM...")
        analysis = engine.process_email(
            from_email=email_info["sender"],
            subject=email_info["subject"],
            body=email_info["body"]
        )

        triage = analysis.get("parsed", {}).get("triage", {})
        draft_reply = analysis.get("parsed", {}).get("draft_reply", "")

        category = triage.get("category", "General Inquiry")
        priority = triage.get("priority", "MEDIUM")
        approval_needed = triage.get("requires_human_approval", False)

        print(f"🏷️ Category: {category} | Priority: {priority} | Approval Needed: {approval_needed}")

        # 2. Decide Auto-Send vs Draft
        should_send = AUTO_SEND
        if SAFETY_GUARD_ENABLED and approval_needed:
            print("⚠️ High-risk email: Falling back to Draft for safety review.")
            should_send = False

        send_reply(service, email_info, draft_reply, auto_send=should_send)

        # 3. Mark as Read and tag with label
        modify_body = {
            "removeLabelIds": ["UNREAD"],
            "addLabelIds": [label_id] if label_id else []
        }
        service.users().messages().modify(userId="me", id=msg_id, body=modify_body).execute()
        print(f"🏷️ Marked email #{msg_id} as Read and tagged '{label_id}'.")

    return processed_count


def run_continuous_watcher(poll_interval_seconds=45):
    """Continuously monitors inbox for new emails arriving AFTER start."""
    bot_start_time = int(time.time())
    start_str = datetime.fromtimestamp(bot_start_time).strftime("%d-%b-%Y %I:%M:%S %p")

    print("=" * 65)
    print("🤖 AUTONOMOUS AI GMAIL BOT ACTIVATED")
    print(f"📅 Bot Start Timestamp: {start_str}")
    print("🛡️ Filter: ONLY processing emails arriving AFTER this start time!")
    print(f"⚡ Auto-Send Mode: {'ENABLED (100% Zero-Touch)' if AUTO_SEND else 'DISABLED (Drafts only)'}")
    print(f"⏱️ Polling Frequency: Every {poll_interval_seconds} seconds")
    print("=" * 65)

    service = get_gmail_service()
    label_id = get_or_create_label(service, "AI-Replied")
    engine = EmailAutomationEngine()

    try:
        while True:
            n = process_inbox(service, engine, label_id, bot_start_time)
            if n == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 💤 Inbox checked. No new emails since start. Waiting {poll_interval_seconds}s...", end="\r")
            time.sleep(poll_interval_seconds)
    except KeyboardInterrupt:
        print("\n\n🛑 AI Gmail Bot stopped by user.")


if __name__ == "__main__":
    run_continuous_watcher(poll_interval_seconds=30)
