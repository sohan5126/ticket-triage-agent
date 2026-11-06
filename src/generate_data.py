"""Generates a synthetic helpdesk ticket log. Replace data/tickets.csv with a real export
(columns: ticket_id, created_at, customer_tier, subject, body, category, priority)."""
import random, pandas as pd
from datetime import datetime, timedelta

random.seed(42)
TEMPLATES = {
    "Billing": ["I was charged twice for my subscription", "Invoice amount is wrong this month",
                "Need a copy of my receipt", "Why did my plan price increase?"],
    "Technical Issue": ["The app crashes when I open reports", "Error 500 when uploading a file",
                        "Dashboard is not loading", "Sync keeps failing on my phone"],
    "Account Access": ["I cannot log in to my account", "Password reset email never arrives",
                       "My account got locked", "Two-factor code is not working"],
    "Shipping": ["My order has not arrived yet", "Tracking shows no update for a week",
                 "Package delivered to the wrong address", "Item arrived damaged"],
    "Feature Request": ["It would be great to have dark mode", "Please add export to Excel",
                        "Can you support bulk editing?", "Suggestion: add calendar integration"],
    "Refund/Cancellation": ["I want to cancel my subscription", "Please refund my last payment",
                            "How do I close my account?", "Requesting refund for defective product"],
}
URGENT = ["urgent", "asap", "immediately", "production is down", "losing money", "critical"]
CALM = ["when you get a chance", "no rush", "just a question", "whenever possible"]
ANGRY = ["this is unacceptable", "very frustrated", "worst service", "I am furious"]
TIERS = ["Free", "Pro", "Enterprise"]

rows = []
start = datetime(2025, 1, 1)
for i in range(3000):
    cat = random.choice(list(TEMPLATES))
    body = random.choice(TEMPLATES[cat])
    tier = random.choices(TIERS, [0.5, 0.35, 0.15])[0]
    score = {"Billing": 1, "Technical Issue": 2, "Account Access": 2, "Shipping": 1,
             "Feature Request": -2, "Refund/Cancellation": 1}[cat]
    score += {"Free": 0, "Pro": 1, "Enterprise": 2}[tier]
    r = random.random()
    if r < 0.25: body += ". " + random.choice(URGENT); score += 3
    elif r < 0.45: body += ". " + random.choice(ANGRY); score += 2
    elif r < 0.65: body += ". " + random.choice(CALM); score -= 2
    score += random.choice([-1, 0, 0, 1])
    priority = "Critical" if score >= 6 else "High" if score >= 4 else "Medium" if score >= 2 else "Low"
    rows.append(dict(ticket_id=f"T{i+1:05d}",
                     created_at=start + timedelta(hours=random.randint(0, 24 * 300)),
                     customer_tier=tier, subject=body.split(".")[0][:50], body=body,
                     category=cat, priority=priority))
pd.DataFrame(rows).to_csv("data/tickets.csv", index=False)
print("Wrote data/tickets.csv", len(rows))
