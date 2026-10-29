"""
Email Automation Engine powered by Custom SLM (Ollama Engine + PyTorch fallback).
Performs:
1. Multi-Sector Email Triage (Category, Priority, Sentiment, Action Items, Candidate Info)
2. Professional Draft Generation from Company/HR Perspective
3. Structured, Clean Output for downstream automation (Gmail / Slack / CRM)
"""

import sys
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.config import SLMConfig
from model.tokenizer import EmailTokenizer
from model.slm_transformer import EmailSLM

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OLLAMA_API_URL = "http://127.0.0.1:11434/api/chat"

class EmailAutomationEngine:
    def __init__(self, model_name: str = "my-custom-email-slm"):
        self.model_name = model_name
        self.use_ollama = self._check_ollama_ready()
        
        # PyTorch fallback attributes
        self.device = None
        self.checkpoint_path = CHECKPOINT_DIR / "best_email_slm.pt"
        self.tokenizer_path = CHECKPOINT_DIR / "tokenizer.json"
        self.tokenizer = None
        self.model = None

        if not self.use_ollama and self.checkpoint_path.exists():
            self._load_pytorch_fallback()

    def _check_ollama_ready(self) -> bool:
        try:
            r = requests.get("http://127.0.0.1:11434/", timeout=5.0)
            if r.status_code == 200:
                tags_res = requests.get("http://127.0.0.1:11434/api/tags", timeout=5.0)
                if tags_res.status_code == 200:
                    models = [m["name"] for m in tags_res.json().get("models", [])]
                    for target in [self.model_name, "my-custom-email-slm:latest", "llama3.2:1b", "llama3.2:latest"]:
                        if target in models or any(target in m for m in models):
                            self.model_name = target.split(":")[0]
                            return True
            return False
        except Exception:
            return False

    def _load_pytorch_fallback(self):
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = EmailTokenizer(str(self.tokenizer_path))
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        config: SLMConfig = ckpt["config"]
        self.model = EmailSLM(config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def process_email(
        self,
        from_email: str,
        subject: str,
        body: str,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Runs the SLM to analyze the email and generate a draft reply."""
        # If Ollama is available or was previously available, stick with Ollama!
        if self.use_ollama:
            return self._process_with_ollama(from_email, subject, body, temperature)
        else:
            # Check once more if Ollama just started
            if self._check_ollama_ready():
                self.use_ollama = True
                return self._process_with_ollama(from_email, subject, body, temperature)
            return self._process_with_pytorch(from_email, subject, body, temperature)

    def _process_with_ollama(self, from_email: str, subject: str, body: str, temperature: float) -> Dict[str, Any]:
        email_content = f"From: {from_email}\nSubject: {subject}\nBody: {body}"
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert AI Email Triage and Response Agent for a modern company. "
                        "Accurately categorize the email into one of: 'Billing & Payments', 'HR / Internship & Job Application', 'Sales / Lead', 'Technical Support', or 'General Inquiry'. "
                        "Always write the draft reply from our company's perspective addressing the sender's specific situation. "
                        "- If Billing: Acknowledge invoice / charge, confirm refund review. "
                        "- If Internship/Job: Acknowledge candidate, confirm resume review. "
                        "- If Sales: Provide demo / quote info. "
                        "- If Technical Support: Acknowledge bug / outage, explain next steps. "
                        "Respond strictly in valid JSON with 'triage' and 'draft_reply' keys."
                    )
                },
                {
                    "role": "user",
                    "content": f"Please triage and draft a response for this email:\n\n{email_content}"
                }
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.2,
                "num_predict": 250,
                "top_p": 0.9
            }
        }

        # Try twice with Ollama before giving up
        last_err = None
        for attempt in range(2):
            try:
                resp = requests.post(OLLAMA_API_URL, json=payload, timeout=90)
                if resp.status_code == 200:
                    raw_json_str = resp.json().get("message", {}).get("content", "")
                    parsed = self._try_parse_json(raw_json_str)
                    return {
                        "engine": f"Ollama ({self.model_name})",
                        "raw_generation": raw_json_str,
                        "parsed": parsed,
                        "metadata": {"sender": from_email, "subject": subject}
                    }
                else:
                    last_err = f"Ollama returned HTTP status {resp.status_code}: {resp.text[:100]}"
            except Exception as e:
                last_err = str(e)
                time.sleep(1.0)

        print(f"⚠️ Ollama request failed ({last_err})")
        return {
            "engine": f"Ollama ({self.model_name}) - Timeout",
            "raw_generation": f"Error: {last_err}",
            "parsed": {
                "triage": {
                    "category": "Service Busy",
                    "priority": "HIGH",
                    "sentiment": "Neutral",
                    "requires_human_approval": True,
                    "action_items": ["Ollama was busy, please retry in 5 seconds"]
                },
                "draft_reply": "System was busy processing a previous request. Please select the sample again."
            },
            "metadata": {"sender": from_email, "subject": subject}
        }

    def _process_with_pytorch(self, from_email: str, subject: str, body: str, temperature: float) -> Dict[str, Any]:
        import torch
        if self.model is None:
            self._load_pytorch_fallback()

        email_input = f"From: {from_email}\nSubject: {subject}\nBody: {body}"
        prompt = (
            f"<|instruction|>\nAnalyze the incoming email and generate both structured triage information and a professional draft response.\n"
            f"<|input|>\n{email_input}\n"
            f"<|output|>\n"
        )

        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            output_tokens = self.model.generate(
                input_ids,
                max_new_tokens=250,
                temperature=temperature,
                stop_token_ids=[self.tokenizer.eos_id]
            )

        decoded = self.tokenizer.decode(output_tokens[0].tolist())
        generated_text = decoded.split("<|output|>")[-1].replace("<|eos|>", "").strip() if "<|output|>" in decoded else decoded.strip()

        return {
            "engine": "Local PyTorch SLM",
            "raw_generation": generated_text,
            "parsed": self._try_parse_json(generated_text),
            "metadata": {"sender": from_email, "subject": subject}
        }

    def _try_parse_json(self, raw_text: str) -> Dict[str, Any]:
        """Robust parser that cleans nested JSON and extracts letter text cleanly."""
        parsed = None
        try:
            parsed = json.loads(raw_text)
        except Exception:
            pass

        triage = {}
        draft = ""

        if isinstance(parsed, dict):
            raw_t = parsed.get("triage", {})
            if isinstance(raw_t, dict):
                triage = raw_t
            elif isinstance(raw_t, str):
                triage = {"category": raw_t, "priority": "MEDIUM", "sentiment": "Neutral", "requires_human_approval": False, "action_items": []}
            draft = parsed.get("draft_reply") or parsed.get("draft") or parsed.get("reply") or ""

        # If triage or draft is missing or malformed, use regex fallback
        if not triage or not isinstance(triage, dict):
            triage = {}
            cat = re.search(r'"category"\s*:\s*"([^"]+)"', raw_text)
            if cat: triage["category"] = cat.group(1)
            pri = re.search(r'"priority"\s*:\s*"([^"]+)"', raw_text)
            if pri: triage["priority"] = pri.group(1).upper()
            sen = re.search(r'"sentiment"\s*:\s*"([^"]+)"', raw_text)
            if sen: triage["sentiment"] = sen.group(1)
            app = re.search(r'"requires_human_approval"\s*:\s*(true|false)', raw_text, re.IGNORECASE)
            if app: triage["requires_human_approval"] = app.group(1).lower() == "true"
            act = re.search(r'"action_items"\s*:\s*\[(.*?)\]', raw_text, re.DOTALL)
            if act:
                triage["action_items"] = re.findall(r'"([^"]+)"', act.group(1))

        if not draft:
            draft_match = re.search(r'"draft_reply"\s*:\s*(.*)', raw_text, re.DOTALL)
            if draft_match:
                draft = draft_match.group(1).strip()

        # Clean draft if it's a dict or contains nested prefixes
        if isinstance(draft, dict):
            draft = draft.get("body") or draft.get("content") or list(draft.values())[0]

        draft_str = str(draft)
        # Strip prefixes like {"Tribe Response:, Response:, etc.
        draft_str = re.sub(r'^[{\\"\'\s]*(Tribe Response:\s*|Response:\s*|Body:\s*|Dear\s+)?', lambda m: "Dear " if m.group().endswith("Dear ") else "", draft_str)
        # Strip trailing braces/quotes
        draft_str = re.sub(r'[\\"\'\}]+\s*$', '', draft_str)
        draft_str = draft_str.replace('\\n', '\n').replace('\\"', '"').strip('"\' \n\r\t')

        return {
            "triage": triage,
            "draft_reply": draft_str if draft_str else raw_text
        }
