"""
Bulk Multi-Sector Dataset Generator for Email Automation SLM.
Generates thousands of diverse, realistic training pairs across:
1. HR & Recruitment (Internships, Senior Engineering, Design, Product roles)
2. Billing & Finance (Disputes, Duplicate charges, Invoices, Refunds, Tax receipts)
3. Enterprise Sales (Quotes, Volume pricing, Demos, POCs)
4. Technical Support (Crashes, 500/502 errors, Rate limits, Security advisories)
5. Corporate & Partnerships (University sponsorships, PR, Vendor contracts)
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = DATA_DIR / "email_slm_dataset.jsonl"

ROLES = [
    "AI/ML Engineer Intern", "Full-Stack Developer", "DevOps Engineer",
    "Data Scientist", "Product Manager", "Frontend React Developer",
    "Cybersecurity Analyst", "Cloud Architect", "UI/UX Designer",
    "Python Backend Developer", "Salesforce Specialist", "QA Automation Engineer"
]

COLLEGES = [
    "DTU", "IIT Delhi", "BITS Pilani", "IIT Bombay", "NIT Trichy",
    "VIT Vellore", "SRM University", "Manipal Institute", "IIT Madras",
    "Thapar University", "IIIT Hyderabad", "PEC Chandigarh"
]

SKILLS_MAP = {
    "AI/ML Engineer Intern": ["PyTorch", "NLP", "LLMs", "FastAPI", "Hugging Face", "Vector DBs"],
    "Full-Stack Developer": ["React", "Node.js", "PostgreSQL", "Docker", "TypeScript", "Tailwind"],
    "DevOps Engineer": ["Kubernetes", "AWS", "Terraform", "CI/CD", "Docker", "Prometheus"],
    "Data Scientist": ["Pandas", "Scikit-Learn", "SQL", "Tableau", "TensorFlow", "Statistics"],
    "UI/UX Designer": ["Figma", "Design Systems", "Wireframing", "User Research", "Adobe XD"],
    "Python Backend Developer": ["Django", "FastAPI", "Redis", "Celery", "PostgreSQL", "GraphQL"]
}

FIRST_NAMES = [
    "Aman", "Rohan", "Priya", "Sneha", "Karan", "Anjali", "Vikram", "Neha",
    "Rahul", "Divya", "Arjun", "Kavita", "Siddharth", "Pooja", "Varun", "Rhea",
    "David", "Sarah", "Michael", "Elena", "Alex", "Emily", "Daniel", "Sophia"
]

LAST_NAMES = [
    "Verma", "Sharma", "Patel", "Gupta", "Singh", "Reddy", "Iyer", "Nair",
    "Malhotra", "Mehta", "Chopra", "Deshmukh", "Miller", "Connor", "Zhang", "Smith"
]

COMPANIES = [
    "Zenith Tech", "Apex Cloud", "Nexus Global", "CyberTech", "FinScale Labs",
    "VentureBridge", "InnoPulse", "Quantum Dynamics", "HealthSync", "LogiChain"
]

def generate_bulk_dataset(target_count: int = 2500):
    print(f"Generating {target_count} multi-sector training pairs...")
    samples = []

    for i in range(target_count):
        fname = random.choice(FIRST_NAMES)
        lname = random.choice(LAST_NAMES)
        full_name = f"{fname} {lname}"
        company = random.choice(COMPANIES)
        email_addr = f"{fname.lower()}.{lname.lower()}@{company.lower().replace(' ', '')}.com"

        sector = random.choices(
            ["HR", "BILLING", "SALES", "TECH_SUPPORT", "PARTNERSHIP"],
            weights=[30, 25, 25, 15, 5],
            k=1
        )[0]

        if sector == "HR":
            role = random.choice(ROLES)
            college = random.choice(COLLEGES)
            skills = ", ".join(random.sample(SKILLS_MAP.get(role, ["Python", "Git", "Problem Solving"]), 3))
            duration = random.choice(["3-month", "6-month", "summer", "immediate"])

            subject = f"Application for {role} - {full_name}"
            body = (
                f"Respected Hiring Team, I am a final/pre-final year student at {college} passionate about {skills}. "
                f"I have built relevant projects and would love to contribute to your team for a {duration} internship/role. "
                f"Please find my resume attached."
            )

            output_obj = {
                "triage": {
                    "category": "HR / Internship & Job Application",
                    "priority": "MEDIUM",
                    "sentiment": "Positive",
                    "requires_human_approval": False,
                    "action_items": [f"Review {full_name}'s resume", "Share with Hiring Lead for initial screening"],
                    "candidate_details": {
                        "name": full_name,
                        "role": role,
                        "institution": college,
                        "skills": skills
                    }
                },
                "draft_reply": (
                    f"Dear {full_name},\n\n"
                    f"Thank you for your interest in the {role} position at our organization. "
                    f"We have received your resume and application. Our recruitment team is currently reviewing your profile and skills in {skills}. "
                    f"We evaluate applications on a rolling basis and will reach out with the next steps regarding the interview schedule within 3 to 5 business days.\n\n"
                    f"Best regards,\nThe HR & Talent Acquisition Team"
                )
            }

        elif sector == "BILLING":
            inv_num = f"INV-{random.randint(1000, 9999)}"
            amt = random.choice([49, 99, 149, 299, 499])
            issue = random.choice(["duplicate charge", "unauthorized billing", "refund request", "tax receipt missing"])

            if issue == "duplicate charge":
                subject = f"Urgent: Double charged ${amt} for invoice #{inv_num}"
                body = f"Hi Accounts Team, My bank statement shows two deductions of ${amt} yesterday for invoice #{inv_num}. Kindly verify and refund the extra amount."
                actions = [f"Verify transaction ledger for #{inv_num}", f"Initiate ${amt} duplicate reversal"]
                reply_text = (
                    f"Dear {full_name},\n\n"
                    f"Thank you for contacting our finance team regarding invoice #{inv_num}.\n\n"
                    f"We sincerely apologize for the inconvenience. We have verified your account records and identified the duplicate charge of ${amt}. "
                    f"Our accounts department has already initiated the refund process back to your original payment method. "
                    f"You should see the amount credited to your account within 3 to 5 business days.\n\n"
                    f"Please let us know if we can assist you with anything else.\n\n"
                    f"Best regards,\nAccounts & Billing Department"
                )
            else:
                subject = f"Refund request for subscription #{inv_num}"
                body = f"Hello support, We discontinued our project earlier this week and would like to request a prorated refund of ${amt} for invoice #{inv_num}."
                actions = [f"Check refund eligibility for #{inv_num}", "Send cancellation confirmation"]
                reply_text = (
                    f"Dear {full_name},\n\n"
                    f"Thank you for reaching out regarding subscription invoice #{inv_num}.\n\n"
                    f"We have received your refund request. Our billing team is reviewing the account usage and refund eligibility in accordance with our terms of service. "
                    f"We will update you with the approval status and credit receipt within 24 to 48 hours.\n\n"
                    f"Best regards,\nCustomer Success & Finance Team"
                )

            output_obj = {
                "triage": {
                    "category": "Billing & Payments",
                    "priority": "HIGH",
                    "sentiment": "Frustrated" if issue == "duplicate charge" else "Neutral",
                    "requires_human_approval": True,
                    "action_items": actions
                },
                "draft_reply": reply_text
            }

        elif sector == "SALES":
            seats = random.choice([25, 50, 100, 250, 500, 1000])
            day = random.choice(["Tuesday", "Wednesday", "Thursday", "Friday"])
            time_slot = random.choice(["2:00 PM", "3:30 PM", "11:00 AM", "4:00 PM"])

            subject = f"Enterprise pricing and demo inquiry for {seats} licenses"
            body = (
                f"Hello Sales, We are actively evaluating AI workflow solutions for our team of {seats} members at {company}. "
                f"Could you share enterprise volume pricing and schedule a 30-minute product demo this {day} around {time_slot}?"
            )

            output_obj = {
                "triage": {
                    "category": "Sales / Lead",
                    "priority": "HIGH",
                    "sentiment": "Positive",
                    "requires_human_approval": False,
                    "action_items": [f"Log {seats}-seat lead for {company}", f"Schedule demo for {day} at {time_slot}"]
                },
                "draft_reply": (
                    f"Dear {full_name},\n\n"
                    f"Thank you for considering our platform for {company}!\n\n"
                    f"We would be delighted to host a personalized product demo for your team. {day} at {time_slot} works perfectly for our enterprise solutions team. "
                    f"I have coordinated a calendar invitation with the meeting link. We will also prepare a tailored volume pricing schedule for {seats} licenses to discuss during the call.\n\n"
                    f"Looking forward to speaking with you!\n\n"
                    f"Best regards,\nEnterprise Sales Team"
                )
            }

        elif sector == "TECH_SUPPORT":
            err_code = random.choice(["500 Internal Server Error", "502 Bad Gateway", "429 Too Many Requests", "Crash on export"])
            endpoint = random.choice(["/api/v1/webhooks", "/api/v1/auth", "/dashboard/analytics", "/export/csv"])

            subject = f"URGENT: Encountering {err_code} on {endpoint}"
            body = f"Hi Support, Our production systems are facing {err_code} when calling {endpoint} since the last 15 minutes. Our pipeline is blocked. Please investigate immediately."

            output_obj = {
                "triage": {
                    "category": "Technical Support",
                    "priority": "CRITICAL",
                    "sentiment": "Urgent",
                    "requires_human_approval": False,
                    "action_items": [f"Notify DevOps on-call for {err_code}", f"Inspect server metrics on {endpoint}"]
                },
                "draft_reply": (
                    f"Dear {full_name},\n\n"
                    f"Thank you for alerting us to this issue. We understand the urgency and apologize for the impact on your production pipeline.\n\n"
                    f"Our engineering and DevOps teams have been immediately notified and are actively investigating the telemetry on {endpoint}. "
                    f"We are treating this with top priority and will provide a status update as soon as root cause analysis is confirmed, within the next 30 minutes.\n\n"
                    f"Best regards,\nTechnical Support & Operations"
                )
            }

        else: # PARTNERSHIP
            subject = f"Partnership & Collaboration Proposal - {company}"
            body = f"Hi Team, We represent {company} and would like to explore a strategic partnership regarding AI automation and joint webinars. Can we connect next week?"

            output_obj = {
                "triage": {
                    "category": "Partnership",
                    "priority": "MEDIUM",
                    "sentiment": "Positive",
                    "requires_human_approval": True,
                    "action_items": [f"Review partnership proposal from {company}", "Forward to Strategic Alliances Lead"]
                },
                "draft_reply": (
                    f"Dear {full_name},\n\n"
                    f"Thank you for reaching out and for your interest in partnering with us.\n\n"
                    f"We are always excited to explore strategic alliances and synergistic collaborations. I have forwarded your note to our Strategic Partnerships lead, who will review your proposal and get in touch to coordinate an introductory discussion.\n\n"
                    f"Best regards,\nStrategic Alliances & Partnerships"
                )
            }

        samples.append({
            "instruction": "Analyze the incoming email and generate both structured triage information and a professional draft response.",
            "input": f"From: {email_addr}\nSubject: {subject}\nBody: {body}",
            "output": json.dumps(output_obj, indent=2)
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    file_size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"Successfully generated {len(samples)} training pairs!")
    print(f"Dataset path: {OUTPUT_FILE} ({file_size_kb:.1f} KB)")
    return len(samples)

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    generate_bulk_dataset(count)
