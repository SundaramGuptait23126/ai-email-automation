"""
Interactive CLI Runner for Email Automation SLM
Provides an intuitive, beautiful terminal interface to:
1. Train the SLM
2. Test on sample emails
3. Enter your own custom incoming email
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import json
import re
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

from data.dataset_generator import generate_synthetic_dataset
from training.train_slm import train
from engine.email_engine import EmailAutomationEngine

console = Console()

SAMPLE_EMAILS = [
    {
        "title": "Internship Application (AI / Python Intern)",
        "from": "aman.verma@college.edu",
        "subject": "Application for AI/ML Intern position - Aman Verma",
        "body": "Respected Hiring Team, I am a 3rd year Computer Science student at DTU passionate about Generative AI, PyTorch, and NLP. I have built an email triage automation project. I have attached my resume and would love to contribute to your team for a 6-month summer internship."
    },
    {
        "title": "Disputed Invoice / Double Charge",
        "from": "karan.malhotra@zenith.in",
        "subject": "Urgent: Double charged for invoice #INV-7712",
        "body": "Hi Support team, I see two deductions of $99 on my bank statement for invoice #INV-7712. Kindly check your records and issue a refund for the duplicate transaction."
    },
    {
        "title": "Enterprise Sales & Pricing Inquiry",
        "from": "sarah.connor@cybertech.com",
        "subject": "Evaluating automation for 300 agents",
        "body": "Hello Sales, We are planning to replace our legacy ticket responder with your AI workflow. We have around 300 agents. Can we schedule an intro call and discuss volume pricing this Wednesday?"
    },
    {
        "title": "System Outage / API Crash",
        "from": "alex@devops.co",
        "subject": "CRITICAL: 502 Bad Gateway on Webhooks endpoint",
        "body": "Team, All our webhook payloads are failing with 502 Bad Gateway since 10 minutes ago. Please look into this urgently, our pipeline is blocked."
    }
]

def display_result(res: dict):
    engine_name = res.get("engine", "Local SLM")
    parsed = res.get("parsed", {})
    triage = parsed.get("triage", {})
    
    # Handle draft_reply if it's a dict or string
    raw_draft = parsed.get("draft_reply") or parsed.get("draft") or parsed.get("reply") or res.get("raw_generation", "")
    if isinstance(raw_draft, dict):
        if "body" in raw_draft:
            draft = raw_draft["body"]
        elif "content" in raw_draft:
            draft = raw_draft["content"]
        elif "draft_reply" in raw_draft:
            draft = raw_draft["draft_reply"]
        else:
            first_key = list(raw_draft.keys())[0] if raw_draft else ""
            if len(first_key) > 40 and ("Dear" in first_key or "Hi" in first_key or "Thank" in first_key):
                draft = first_key
            else:
                draft = json.dumps(raw_draft, indent=2)
    else:
        draft = str(raw_draft)

    # Clean escaped characters and json residue
    draft = draft.replace("\\n", "\n").replace('\\"', '"').strip()
    draft = re.sub(r'^[{\\"\'\s]*(Tribe Response:\s*|Response:\s*|Body:\s*)?', '', draft)
    draft = re.sub(r'[\\"\'\}]+\s*$', '', draft)
    draft = draft.strip('"\' \n\r\t')

    # Format action items safely (handles list of strings, list of dicts, or single string)
    action_items_raw = triage.get("action_items", [])
    if isinstance(action_items_raw, list):
        formatted_actions = []
        for item in action_items_raw:
            if isinstance(item, dict):
                # Try common keys
                val = item.get("action") or item.get("task") or item.get("description") or item.get("item") or list(item.values())[0]
                formatted_actions.append(str(val))
            else:
                formatted_actions.append(str(item))
        action_items_str = "\n• " + "\n• ".join(formatted_actions) if formatted_actions else "None"
    else:
        action_items_str = str(action_items_raw)

    # Table for triage
    table = Table(title=f"🏷️ AI Email Triage Result [{engine_name}]", show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan", width=25)
    table.add_column("Value", style="green")

    table.add_row("Category", str(triage.get("category", "General Inquiry")))
    priority_val = str(triage.get("priority", "MEDIUM")).upper()
    priority_color = "red" if priority_val in ["HIGH", "CRITICAL"] else "yellow"
    table.add_row("Priority", f"[{priority_color}]{priority_val}[/{priority_color}]")
    table.add_row("Sentiment", str(triage.get("sentiment", "Neutral")))
    
    needs_approval = triage.get("requires_human_approval", True)
    table.add_row("Human Approval Needed?", "⚠️ YES" if needs_approval else "✅ NO (Safe to Auto-Send)")
    table.add_row("Action Items", action_items_str)

    # Candidate details if present
    candidate = triage.get("candidate_details")
    if candidate and isinstance(candidate, dict):
        cand_str = ", ".join([f"{k}: {v}" for k, v in candidate.items()])
        table.add_row("Candidate Info", cand_str)

    console.print(table)

    # Panel for Draft Reply
    console.print(Panel(
        draft.strip(),
        title="✉️ AI Generated Professional Email Draft",
        border_style="green",
        subtitle="Ready for Gmail / Outlook Dispatch"
    ))

def main():
    console.print(Panel.fit(
        "[bold cyan]🤖 AI EMAIL AUTOMATION SLM (Small Language Model)[/bold cyan]\n"
        "[dim]100% Private • Ultra-Fast • Custom Fine-Tuned for Email Triage & Drafting[/dim]",
        border_style="blue"
    ))

    ckpt_file = Path("checkpoints/best_email_slm.pt")

    while True:
        console.print("\n[bold yellow]Menu Options:[/bold yellow]")
        console.print("1. [bold green]Train / Retrain Custom SLM[/bold green] (on local dataset)")
        console.print("2. [bold cyan]Test with Sample Incoming Emails[/bold cyan]")
        console.print("3. [bold magenta]Type Your Own Custom Email[/bold magenta]")
        console.print("4. [bold red]Exit[/bold red]")

        choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4"], default="2")

        if choice == "1":
            console.print("[cyan]Generating training dataset...[/cyan]")
            generate_synthetic_dataset(150)
            console.print("[green]Starting SLM training...[/green]")
            train(epochs=10, batch_size=4)
            console.print("[bold green]✅ Training completed and checkpoint saved![/bold green]")

        elif choice == "2":
            if not ckpt_file.exists():
                console.print("[yellow]No trained model checkpoint found! Let's train a fast model first (1-2 mins)...[/yellow]")
                generate_synthetic_dataset(150)
                train(epochs=8, batch_size=4)

            engine = EmailAutomationEngine()
            console.print("\nSelect a sample email to simulate:")
            for idx, s in enumerate(SAMPLE_EMAILS, 1):
                console.print(f"  {idx}. [bold]{s['title']}[/bold] - [dim]{s['subject']}[/dim]")

            email_idx = int(Prompt.ask("Choose sample", choices=["1", "2", "3", "4"], default="1")) - 1
            selected = SAMPLE_EMAILS[email_idx]

            console.print(Panel(
                f"[bold]From:[/bold] {selected['from']}\n"
                f"[bold]Subject:[/bold] {selected['subject']}\n\n"
                f"{selected['body']}",
                title=f"📥 Incoming Email: {selected['title']}",
                border_style="yellow"
            ))

            with console.status("[bold green]SLM is processing email (Inferencing)...[/bold green]"):
                res = engine.process_email(
                    from_email=selected["from"],
                    subject=selected["subject"],
                    body=selected["body"]
                )

            display_result(res)

        elif choice == "3":
            if not ckpt_file.exists():
                console.print("[yellow]No trained checkpoint found! Training first...[/yellow]")
                generate_synthetic_dataset(150)
                train(epochs=8, batch_size=4)

            engine = EmailAutomationEngine()
            sender = Prompt.ask("Sender Email", default="customer@domain.com")
            subject = Prompt.ask("Subject", default="Help with my subscription")
            console.print("Enter Email Body (Press Enter when done):")
            body = input("> ")

            with console.status("[bold green]SLM is processing email...[/bold green]"):
                res = engine.process_email(
                    from_email=sender,
                    subject=subject,
                    body=body
                )

            display_result(res)

        elif choice == "4":
            console.print("[bold cyan]Goodbye![/bold cyan]")
            break

if __name__ == "__main__":
    main()
