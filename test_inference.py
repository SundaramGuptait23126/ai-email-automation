import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from engine.email_engine import EmailAutomationEngine

def run_test():
    print("Initializing Email Automation Engine...")
    engine = EmailAutomationEngine()

    test_emails = [
        {
            "from": "john.smith@acme.com",
            "subject": "Charged twice on order #4891",
            "body": "Hi, I was billed $99 two times yesterday on invoice #4891. Please refund the extra amount."
        },
        {
            "from": "lisa@techcorp.io",
            "subject": "Demo request for 100 enterprise licenses",
            "body": "Hello, we want to try your automation platform for our 100 team members. When can we talk?"
        }
    ]

    for idx, em in enumerate(test_emails, 1):
        print(f"\n--- Testing Email {idx} ---")
        print(f"From: {em['from']}")
        print(f"Subject: {em['subject']}")
        print(f"Body: {em['body']}")

        res = engine.process_email(
            from_email=em["from"],
            subject=em["subject"],
            body=em["body"],
            temperature=0.5
        )

        print("\n[AI SLM Triage & Draft Output]:")
        print(res["raw_generation"])
        print("-" * 50)

if __name__ == "__main__":
    run_test()
