import json, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from agents import add_features, build_category_model, build_priority_model

df = add_features(pd.read_csv("data/tickets.csv"))
tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["category"])

cat = build_category_model().fit(tr["text"], tr["category"])
pri = build_priority_model().fit(tr, tr["priority"])

metrics = {}
for name, model, X, y in [("category", cat, te["text"], te["category"]), ("priority", pri, te, te["priority"])]:
    pred = model.predict(X)
    rep = classification_report(y, pred, output_dict=True, zero_division=0)
    labels = list(model.classes_)
    metrics[name] = dict(report=rep, labels=labels, confusion=confusion_matrix(y, pred, labels=labels).tolist())
    print(f"\n== {name} ==\n", classification_report(y, pred, zero_division=0))

joblib.dump({"cat": cat, "pri": pri}, "models/triage.joblib")
json.dump(metrics, open("models/metrics.json", "w"))
print("Saved models/triage.joblib")
