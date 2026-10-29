"""
Production FastAPI Web Service for AI Email Automation.
Deployable on Render.com, Railway, or any Cloud Provider.

Endpoints:
- GET  /                  : Health Check & System Status
- POST /api/triage        : Analyzes raw email text and generates Triage + Professional Reply
- POST /api/process-inbox : Triggers Gmail inbox scan & auto-replies
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.email_engine import EmailAutomationEngine

# Initialize FastAPI App
app = FastAPI(
    title="AI Email Automation API",
    description="Autonomous Multi-Sector Email Triage and Professional Drafting Engine",
    version="1.0.0"
)

# Enable CORS for external dashboards / webhooks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load engine on startup
engine: Optional[EmailAutomationEngine] = None

@app.on_event("startup")
def startup_event():
    global engine
    print("🚀 Initializing AI Email Automation Engine...")
    engine = EmailAutomationEngine()
    print("✅ Engine ready for requests!")


# ------------------------------------------------------------------------------
# Request & Response Schemas
# ------------------------------------------------------------------------------
class EmailTriageRequest(BaseModel):
    from_email: str = Field(..., example="aman.verma@college.edu")
    subject: str = Field(..., example="Application for AI/ML Intern position")
    body: str = Field(..., example="Hi Hiring Team, I am a 3rd year CS student passionate about PyTorch and NLP. Please find my resume attached.")
    temperature: float = Field(0.2, ge=0.0, le=1.0)

class TriageMetadata(BaseModel):
    category: str
    priority: str
    sentiment: str
    requires_human_approval: bool
    action_items: List[str] = []
    candidate_details: Optional[Dict[str, Any]] = None

class EmailTriageResponse(BaseModel):
    engine: str
    sender: str
    subject: str
    triage: TriageMetadata
    draft_reply: str
    raw_generation: Optional[str] = None


# Helper to locate secret files across Local and Cloud (Render /etc/secrets)
def resolve_secret_file(filename: str) -> Optional[Path]:
    """Finds a secret file either in PROJECT_ROOT, Render /etc/secrets, or env vars."""
    custom_env = os.environ.get(filename.upper().replace(".", "_") + "_PATH")
    candidates = [
        Path(custom_env) if custom_env else None,
        PROJECT_ROOT / filename,
        Path("/etc/secrets") / filename,
        Path.home() / ".credentials" / filename
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None

# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------
@app.get("/")
def health_check():
    global engine
    engine_name = "Not Loaded"
    if engine:
        engine_name = f"Ollama ({engine.model_name})" if engine.use_ollama else "Local Custom PyTorch SLM (2.23M)"

    token_path = resolve_secret_file("token.json")
    creds_path = resolve_secret_file("credentials.json")
    has_token_env = bool(os.environ.get("GMAIL_TOKEN_JSON"))

    return {
        "status": "online",
        "service": "AI Email Automation Backend",
        "active_engine": engine_name,
        "checkpoint_exists": (PROJECT_ROOT / "checkpoints" / "best_email_slm.pt").exists(),
        "gmail_configured": (token_path is not None) or (creds_path is not None) or has_token_env,
        "token_location": str(token_path) if token_path else ("env:GMAIL_TOKEN_JSON" if has_token_env else None)
    }


@app.post("/api/triage", response_model=EmailTriageResponse)
def triage_email(req: EmailTriageRequest):
    """
    Analyzes an incoming email and returns structured categorization,
    priority, sentiment, action items, and a ready-to-send draft response.
    """
    global engine
    if not engine:
        raise HTTPException(status_code=503, detail="AI Engine is still initializing. Please retry in a few seconds.")

    try:
        result = engine.process_email(
            from_email=req.from_email,
            subject=req.subject,
            body=req.body,
            temperature=req.temperature
        )

        parsed = result.get("parsed", {})
        triage_data = parsed.get("triage", {})
        draft = parsed.get("draft_reply", "")

        return EmailTriageResponse(
            engine=result.get("engine", "AI Engine"),
            sender=req.from_email,
            subject=req.subject,
            triage=TriageMetadata(
                category=triage_data.get("category", "General Inquiry"),
                priority=str(triage_data.get("priority", "MEDIUM")).upper(),
                sentiment=triage_data.get("sentiment", "Neutral"),
                requires_human_approval=triage_data.get("requires_human_approval", False),
                action_items=triage_data.get("action_items", []),
                candidate_details=triage_data.get("candidate_details")
            ),
            draft_reply=draft,
            raw_generation=result.get("raw_generation")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/process-inbox")
def trigger_inbox_processing(background_tasks: BackgroundTasks):
    """
    Triggers an inbox scan via Gmail API in the background to auto-reply to unread emails.
    """
    token_path = resolve_secret_file("token.json")
    creds_path = resolve_secret_file("credentials.json")
    has_token_env = bool(os.environ.get("GMAIL_TOKEN_JSON"))

    if not token_path and not creds_path and not has_token_env:
        raise HTTPException(
            status_code=400,
            detail="Gmail credentials (token.json) not found on server (/etc/secrets/token.json or local). Please configure Gmail authentication first."
        )

    def run_gmail_worker():
        try:
            import gmail_automation
            service = gmail_automation.get_gmail_service()
            label_id = gmail_automation.get_or_create_label(service, "AI-Replied")
            bot_start = int(os.environ.get("BOT_START_TIMESTAMP", 0))
            count = gmail_automation.process_inbox(service, engine, label_id, bot_start)
            print(f"✅ Background inbox scan complete. Processed {count} email(s).")
        except Exception as e:
            print(f"❌ Error during background inbox processing: {e}")

    background_tasks.add_task(run_gmail_worker)
    return {"message": "Inbox scan scheduled in background task."}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
