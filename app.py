import sys, json
sys.path.insert(0, "src")
import pandas as pd, plotly.express as px, streamlit as st
from agents import TriageAgent, add_features

st.set_page_config(page_title="Support Triage Agent", layout="wide")
st.title("🎫 Support Operations Dashboard")


@st.cache_resource
def load_agent():
    return TriageAgent.load()


@st.cache_data
def load_data():
    return pd.read_csv("data/tickets.csv", parse_dates=["created_at"])


agent, df = load_agent(), load_data()
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🤖 Triage a Ticket", "📁 Batch Triage", "🧪 Model Performance"])

with tab1:
    c = st.columns(4)
    c[0].metric("Total tickets", f"{len(df):,}")
    c[1].metric("Critical", int((df.priority == "Critical").sum()))
    c[2].metric("Enterprise share", f"{(df.customer_tier == 'Enterprise').mean():.0%}")
    c[3].metric("Categories", df.category.nunique())
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(df, x="category", color="priority", title="Tickets by category & priority"),
                   use_container_width=True)
    daily = df.set_index("created_at").resample("W").size().reset_index(name="tickets")
    daily = daily.iloc[:-1]  
    b.plotly_chart(px.line(daily, x="created_at", y="tickets", title="Weekly ticket volume"),
                   use_container_width=True)

with tab2:
    subject = st.text_input("Subject", "Payment failed")
    body = st.text_area("Ticket body", "I was charged twice and this is unacceptable. Need a refund asap.")
    tier = st.selectbox("Customer tier", ["Free", "Pro", "Enterprise"])
    if st.button("Run triage agent", type="primary"):
        r = agent.run(subject, body, tier)
        m = st.columns(4)
        m[0].metric("Category", r.category, f"{r.category_conf:.0%}")
        m[1].metric("Priority", r.priority, f"{r.priority_conf:.0%}")
        m[2].metric("Route to", r.team)
        m[3].metric("SLA", f"{r.sla_hours}h")
        if r.escalate:
            st.warning("Escalated to human review (critical or low confidence).")
        st.subheader("Agent trace")
        for step in r.trace:
            st.write("•", step)
        st.subheader("Draft reply")
        st.info(r.draft_reply)

with tab3:
    st.caption("Upload a CSV with columns: subject, body, customer_tier")
    up = st.file_uploader("Tickets CSV", type="csv")
    if up:
        inp = pd.read_csv(up)
        missing = {"subject", "body"} - set(inp.columns)
        if missing:
            st.error(f"Missing columns: {', '.join(sorted(missing))}")
            st.stop()
        # rename any existing result columns so they don't clash with predictions
        clash = ["category", "priority", "team", "sla_hours", "escalate"]
        inp = inp.rename(columns={c: f"given_{c}" for c in clash if c in inp.columns})
        if "customer_tier" not in inp.columns:
            inp["customer_tier"] = "Free"
        out = []
        for _, row in inp.iterrows():
            r = agent.run(row["subject"], row["body"], row["customer_tier"])
            out.append(dict(category=r.category, priority=r.priority, team=r.team,
                            sla_hours=r.sla_hours, escalate=r.escalate))
        res = pd.concat([inp.reset_index(drop=True), pd.DataFrame(out)], axis=1)
        if "given_priority" in res.columns:
            st.metric("Priority matches the file's labels", f"{(res.priority == res.given_priority).mean():.0%}")
        st.dataframe(res, use_container_width=True)
        st.download_button("Download results", res.to_csv(index=False), "triaged.csv")

with tab4:
    met = json.load(open("models/metrics.json"))
    for name in ["category", "priority"]:
        m = met[name]
        st.subheader(f"{name.title()} model")
        st.metric("Accuracy", f"{m['report']['accuracy']:.1%}")
        st.plotly_chart(px.imshow(m["confusion"], x=m["labels"], y=m["labels"], text_auto=True,
                                  labels=dict(x="Predicted", y="Actual"), title="Confusion matrix"),
                        use_container_width=True)
