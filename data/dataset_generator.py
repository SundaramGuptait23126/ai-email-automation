"""
Dataset Generator for Email Automation SLM (Small Language Model)
Generates high-quality instruction-tuning pairs for:
1. Intent & Category Classification
2. Sentiment & Priority Triage
3. Contextual, Professional Email Drafting
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent
OUTPUT_FILE = DATA_DIR / "email_slm_dataset.jsonl"

TEMPLATES = [
    # 1. Billing / Refund Request
    {
        "instruction": "Analyze the incoming email and generate both structured triage information and a professional draft response.",
        "input": "From: rahul.sharma@example.com\nSubject: Charged twice for my monthly subscription\nBody: Hi Team, I was charged twice ($49 x 2) on my credit card for invoice #INV-9821 yesterday. Please refund the extra amount immediately. This is frustrating.",
        "output": json.dumps({
            "triage": {
                "category": "Billing & Payments",
                "priority": "HIGH",
                "sentiment": "Frustrated",
                "requires_human_approval": True,
                "action_items": ["Verify charge on invoice #INV-9821", "Process duplicate refund of $49"]
            },
            "draft_reply": "Hi Rahul,\n\nThank you for reaching out, and we sincerely apologize for the inconvenience caused by the duplicate charge.\n\nWe have located your account and invoice #INV-9821. Our billing team is currently reviewing the transaction to reverse the additional $49 charge back to your original payment method. You should see the refund reflected in your account within 3 to 5 business days.\n\nWe will update you as soon as the refund reference ID is generated. Please let us know if you need any further assistance in the meantime.\n\nBest regards,\nCustomer Support Team"
        }, indent=2)
    },
    # 2. Sales / Demo Inquiry
    {
        "instruction": "Analyze the incoming email and generate both structured triage information and a professional draft response.",
        "input": "From: priya.n@techcorp.io\nSubject: Enterprise demo inquiry for 150 users\nBody: Hello, We are evaluating workflow automation tools for our engineering team of 150 members. Can we set up a quick 30-minute demo this Thursday around 3 PM IST?",
        "output": json.dumps({
            "triage": {
                "category": "Sales / Lead",
                "priority": "HIGH",
                "sentiment": "Positive",
                "requires_human_approval": False,
                "action_items": ["Log lead in CRM (150 seats)", "Coordinate demo for Thursday 3:00 PM IST"]
            },
            "draft_reply": "Hi Priya,\n\nThank you for your interest in our workflow automation platform!\n\nWe would be delighted to host a tailored demo for your engineering team. Thursday at 3:00 PM IST works perfectly for our solutions architect. \n\nI have attached a calendar invite with the Google Meet link. Please feel free to forward it to your team members, and let us know if there are specific use cases or integrations you would like us to focus on.\n\nLooking forward to speaking with you!\n\nBest regards,\nSales & Solutions Team"
        }, indent=2)
    },
    # 3. Technical Support / Bug Report
    {
        "instruction": "Analyze the incoming email and generate both structured triage information and a professional draft response.",
        "input": "From: david.miller@acme.org\nSubject: Error 500 when exporting CSV report\nBody: Hi, Since this morning, whenever we click 'Export CSV' on the analytics dashboard, it throws an internal server error (500). Our weekly report is due today. Please help.",
        "output": json.dumps({
            "triage": {
                "category": "Technical Support",
                "priority": "CRITICAL",
                "sentiment": "Urgent",
                "requires_human_approval": False,
                "action_items": ["Investigate CSV export service 500 errors", "Notify DevOps/Engineering on-call"]
            },
            "draft_reply": "Hi David,\n\nThank you for alerting us to this issue. We understand that your weekly report is due and apologize for the disruption.\n\nOur engineering team has been notified and is actively investigating the 500 error on the analytics export service. In the meantime, our support engineer can manually generate and send your export if you share your account workspace ID.\n\nWe will provide a status update within the next 45 minutes.\n\nBest regards,\nTechnical Support Team"
        }, indent=2)
    },
    # 4. Partnership / Vendor Pitch (Low Priority / Polite Decline or Review)
    {
        "instruction": "Analyze the incoming email and generate both structured triage information and a professional draft response.",
        "input": "From: sam@outsourceseo.biz\nSubject: Boost your website SEO ranking #1 on Google\nBody: Dear Founder, We offer guaranteed top 3 ranking on Google within 30 days. Let me know if you have 10 mins to discuss our affordable monthly packages.",
        "output": json.dumps({
            "triage": {
                "category": "Cold Outreach / Vendor",
                "priority": "LOW",
                "sentiment": "Neutral",
                "requires_human_approval": False,
                "action_items": ["Archive or polite decline"]
            },
            "draft_reply": "Hi Sam,\n\nThank you for reaching out. At this time, we have our organic marketing and SEO strategies covered internally and are not looking to onboard external agencies.\n\nWe will keep your details on file should our requirements change in the future. Wishing you continued success.\n\nBest regards,\nOperations Team"
        }, indent=2)
    },
    # 5. Account Cancellation / Retention
    {
        "instruction": "Analyze the incoming email and generate both structured triage information and a professional draft response.",
        "input": "From: anita.desai@startup.co\nSubject: Cancel our Pro plan\nBody: Hello support, we have decided to shut down our subsidiary project and will no longer need the Pro subscription. Please cancel our account before the next billing cycle on Nov 1st.",
        "output": json.dumps({
            "triage": {
                "category": "Churn / Cancellation",
                "priority": "MEDIUM",
                "sentiment": "Neutral",
                "requires_human_approval": True,
                "action_items": ["Process account cancellation effective Nov 1st", "Stop auto-renewal", "Send data export instructions"]
            },
            "draft_reply": "Hi Anita,\n\nThank you for letting us know. We have processed the cancellation request for your Pro subscription, effective at the end of your current billing cycle on November 1st.\n\nYou will continue to have full access to your workspace until that date, and no further charges will occur. If you need to export any historical project data or reports, you can do so directly from Settings > Data Export before Nov 1st.\n\nWe truly appreciate your journey with us and hope to work together again on your future ventures!\n\nWarm regards,\nCustomer Success Team"
        }, indent=2)
    },
    # 6. Meeting Reschedule Request
    {
        "instruction": "Analyze the incoming email and generate both structured triage information and a professional draft response.",
        "input": "From: mark.z@partnercapital.com\nSubject: Rescheduling our strategy sync\nBody: Hey, Something urgent came up on my calendar for our 2 PM call today. Could we push it to tomorrow morning between 10 AM and 12 PM?",
        "output": json.dumps({
            "triage": {
                "category": "Meeting / Scheduling",
                "priority": "HIGH",
                "sentiment": "Neutral",
                "requires_human_approval": False,
                "action_items": ["Cancel 2 PM invite today", "Send updated invite for tomorrow 11:00 AM"]
            },
            "draft_reply": "Hi Mark,\n\nNo problem at all, thanks for the heads-up! \n\nTomorrow morning at 11:00 AM works well on our end. I have updated the calendar invitation accordingly with the meeting link.\n\nTalk soon!\n\nBest regards,\nExecutive Office"
        }, indent=2)
    }
]

# Variations and expansions
COMPANIES = ["FinTech Global", "CloudScale Labs", "Nexus Retail", "HealthSync", "Apex Logistics", "VentureBridge"]
NAMES = ["Amit Verma", "Sara Khan", "Michael Chang", "Neha Patel", "Elena Rostova", "Kevin O'Connor"]
CATEGORIES_TEMPLATES = [
    ("Billing & Payments", "URGENT: Invoice dispute for order #{id}", "Hi, we notice an extra charge of ${amount} on our account #{id}. Please clarify."),
    ("Technical Support", "API Rate Limiting issue on endpoint /v1/{endpoint}", "Hello, our production system is hitting unexpected 429 errors on the /v1/{endpoint} route. We are well within tier limits."),
    ("Sales / Lead", "Request for custom quote - {seats} licenses", "Hi Sales, We are looking to deploy your software across {seats} team members. Can you share custom tier pricing?"),
    ("Feedback & Praise", "Loved the latest update!", "Hey team, just wanted to say the new automation flow saved us 10 hours this week. Amazing work!"),
]

def generate_synthetic_dataset(num_samples: int = 150):
    samples = []
    # Add base high quality hand-crafted templates
    for item in TEMPLATES:
        samples.append(item)

    # Generate synthetic variations
    for i in range(num_samples - len(TEMPLATES)):
        cat, sub_tpl, body_tpl = random.choice(CATEGORIES_TEMPLATES)
        name = random.choice(NAMES)
        company = random.choice(COMPANIES)
        rid = random.randint(1000, 9999)
        amount = random.choice([29, 49, 99, 199, 499])
        seats = random.choice([25, 50, 100, 500])
        endpoint = random.choice(["messages", "search", "webhooks", "analytics"])

        subject = sub_tpl.format(id=rid, amount=amount, seats=seats, endpoint=endpoint)
        body = body_tpl.format(id=rid, amount=amount, seats=seats, endpoint=endpoint)
        email_input = f"From: {name.lower().replace(' ', '.')}@{company.lower().replace(' ', '')}.com\nSubject: {subject}\nBody: {body}"

        if cat == "Billing & Payments":
            priority = "HIGH"
            sentiment = "Concerned"
            triage_action = [f"Check transaction #{rid}", f"Verify dispute of ${amount}"]
            reply = f"Hi {name.split()[0]},\n\nThank you for contacting our finance team regarding order #{rid}.\n\nWe are reviewing the charge of ${amount} against your account records right now. Our team will verify the billing ledger and update you with a detailed breakdown or refund confirmation within 24 hours.\n\nBest regards,\nAccounts Team"
        elif cat == "Technical Support":
            priority = "CRITICAL"
            sentiment = "Urgent"
            triage_action = [f"Investigate /v1/{endpoint} rate limits", "Check API gateway telemetry"]
            reply = f"Hi {name.split()[0]},\n\nThank you for reaching out. We have flagged this rate limit alert with our API infrastructure team.\n\nWe are actively verifying the throttling metrics on /v1/{endpoint} for your tier. We will deploy a fix or elevate your temporary quota within the next hour.\n\nBest regards,\nDeveloper Support Team"
        elif cat == "Sales / Lead":
            priority = "HIGH"
            sentiment = "Positive"
            triage_action = [f"Prepare quote for {seats} seats", "Schedule introductory call"]
            reply = f"Hi {name.split()[0]},\n\nThank you for considering our platform for {company}!\n\nFor a deployment of {seats} licenses, we offer enterprise volume pricing, dedicated customer success, and custom SLA terms. I would love to share our pricing sheet and set up a brief call with your procurement lead.\n\nBest regards,\nEnterprise Sales Team"
        else:
            priority = "LOW"
            sentiment = "Delighted"
            triage_action = ["Share with product & engineering teams", "Log testimonial"]
            reply = f"Hi {name.split()[0]},\n\nThank you so much for the kind words! Messages like yours make our entire team's day.\n\nWe have shared your feedback with our product squad. Please let us know if there is anything else we can do to make your experience even better.\n\nBest regards,\nCustomer Success Team"

        output_json = {
            "triage": {
                "category": cat,
                "priority": priority,
                "sentiment": sentiment,
                "requires_human_approval": priority in ["HIGH", "CRITICAL"],
                "action_items": triage_action
            },
            "draft_reply": reply
        }

        samples.append({
            "instruction": "Analyze the incoming email and generate both structured triage information and a professional draft response.",
            "input": email_input,
            "output": json.dumps(output_json, indent=2)
        })

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Generated {len(samples)} high-quality training pairs at: {OUTPUT_FILE}")
    return len(samples)

if __name__ == "__main__":
    generate_synthetic_dataset(200)
