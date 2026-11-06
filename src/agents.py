"""Multi-step agentic triage pipeline.
Each agent does one job and appends to a shared trace so the dashboard can show reasoning."""
import re, os, joblib
import numpy as np, pandas as pd
from dataclasses import dataclass, field
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

URGENT_WORDS = {"urgent", "asap", "immediately", "critical", "down", "losing", "emergency"}
NEG_WORDS = {"unacceptable", "frustrated", "worst", "furious", "angry", "terrible", "damaged"}
TIER_MAP = {"Free": 0, "Pro": 1, "Enterprise": 2}

ROUTING = {  # category -> team
    "Billing": "Billing Team", "Technical Issue": "Tier-2 Engineering",
    "Account Access": "Security & Access", "Shipping": "Logistics",
    "Feature Request": "Product Team", "Refund/Cancellation": "Retention Team"}
SLA_HOURS = {"Critical": 1, "High": 4, "Medium": 24, "Low": 72}


def clean_text(t: str) -> str:
    t = re.sub(r"http\S+|\S+@\S+", " ", str(t).lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9!\s']", " ", t)).strip()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agent 1 - Intake: clean text + engineer features."""
    df = df.copy()
    df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).map(clean_text)
    toks = df["text"].str.split()
    df["urgent_hits"] = toks.map(lambda w: sum(x in URGENT_WORDS for x in w))
    df["neg_hits"] = toks.map(lambda w: sum(x in NEG_WORDS for x in w))
    df["exclaims"] = df["text"].str.count("!")
    df["length"] = toks.map(len)
    df["tier_num"] = df["customer_tier"].map(TIER_MAP).fillna(0)
    return df


NUM = ["urgent_hits", "neg_hits", "exclaims", "length", "tier_num"]


def build_category_model():
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                     ("clf", LogisticRegression(max_iter=1000))])


def build_priority_model():
    pre = ColumnTransformer([("txt", TfidfVectorizer(ngram_range=(1, 2), min_df=2), "text"),
                             ("num", StandardScaler(), NUM),
                             ("tier", OneHotEncoder(handle_unknown="ignore"), ["customer_tier"])])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])


@dataclass
class TriageResult:
    category: str
    category_conf: float
    priority: str
    priority_conf: float
    team: str
    sla_hours: int
    escalate: bool
    draft_reply: str
    trace: list = field(default_factory=list)


class TriageAgent:
    """Orchestrator: Intake -> Classifier -> Prioritizer -> Router -> Responder."""

    def __init__(self, cat_model, pri_model):
        self.cat_model, self.pri_model = cat_model, pri_model

    @classmethod
    def load(cls, path="models/triage.joblib"):
        d = joblib.load(path)
        return cls(d["cat"], d["pri"])

    def run(self, subject, body, tier="Free") -> TriageResult:
        trace = []
        df = add_features(pd.DataFrame([{"subject": subject, "body": body, "customer_tier": tier}]))
        trace.append(f"Intake: cleaned text; urgent_hits={int(df.urgent_hits[0])}, "
                     f"negative_hits={int(df.neg_hits[0])}, tier={tier}")
        cp = self.cat_model.predict_proba(df["text"])[0]
        cat = self.cat_model.classes_[cp.argmax()]
        trace.append(f"Classifier: category={cat} (conf {cp.max():.0%})")
        pp = self.pri_model.predict_proba(df)[0]
        pri = self.pri_model.classes_[pp.argmax()]
        trace.append(f"Prioritizer: priority={pri} (conf {pp.max():.0%})")
        team, sla = ROUTING[cat], SLA_HOURS[pri]
        low_conf = cp.max() < 0.5 or pp.max() < 0.5
        escalate = pri == "Critical" or low_conf
        trace.append(f"Router: -> {team}, SLA {sla}h" + (" | ESCALATED to human review" if escalate else ""))
        reply = self.draft_reply(cat, pri, body)
        trace.append("Responder: drafted acknowledgement")
        return TriageResult(cat, float(cp.max()), pri, float(pp.max()), team, sla, escalate, reply, trace)

    def draft_reply(self, cat, pri, body):
        """Uses an LLM if ANTHROPIC_API_KEY is set, otherwise a template (so the app works offline)."""
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                msg = anthropic.Anthropic().messages.create(
                    model="claude-sonnet-4-5", max_tokens=250,
                    messages=[{"role": "user", "content":
                               f"Write a short, empathetic support reply (max 80 words). "
                               f"Category: {cat}. Priority: {pri}. Ticket: {body}"}])
                return msg.content[0].text
            except Exception:
                pass
        return (f"Hi, thanks for contacting us. We've logged your {cat.lower()} request as "
                f"{pri.lower()} priority and routed it to the right team. "
                f"You'll hear back within {SLA_HOURS[pri]} hour(s).")
