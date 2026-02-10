import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not found in environment")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "gpt-4o-mini"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

df_full = pd.read_csv("data_training_selected_clusters_comments_and_rules.csv")
sample_df = df_full.sample(50, random_state=2).reset_index(drop=True)

df_reduced = sample_df[["body", "subreddit_id", "label"]]
df_clusters_reduced = sample_df[
    ["body", "subreddit_id", "assigned_rule_cluster", "label"]
]

def build_prompt(body, subreddit_id):
    return f"""
You are a moderation classifier.

Given the Reddit comment below, infer the MOST LIKELY rule cluster it violates.
If it does not violate any rule, output: NONE.

Return ONLY the cluster name or NONE.

Comment:
\"\"\"{body}\"\"\"

Subreddit ID: {subreddit_id}
"""

def get_llm_prediction(body, subreddit_id):
    messages = [
        {"role": "system", "content": "You classify moderation rule clusters."},
        {"role": "user", "content": build_prompt(body, subreddit_id)}
    ]
    data = {"model": MODEL, "messages": messages, "temperature": 0}

    try:
        response = requests.post(API_URL, headers=HEADERS, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"ERROR: {e}"

predicted_clusters = []
print("Running LLM classification...")

for i, row in enumerate(df_reduced.itertuples(), start=1):
    prediction = get_llm_prediction(row.body, row.subreddit_id)
    predicted_clusters.append(prediction)
    print(f"[{i}/{len(df_reduced)}] → {prediction}")

df_eval = df_clusters_reduced.copy()
df_eval["predicted_rule_cluster"] = predicted_clusters

def normalize(x):
    return str(x).strip().lower()

df_eval["gold_norm"] = df_eval["assigned_rule_cluster"].map(normalize)
df_eval["pred_norm"] = df_eval["predicted_rule_cluster"].map(normalize)

accuracy = (df_eval["gold_norm"] == df_eval["pred_norm"]).mean()
print("\nExact-match accuracy:", f"{accuracy:.2%}")

print("\nConfusion matrix:")
print(pd.crosstab(df_eval["assigned_rule_cluster"], df_eval["predicted_rule_cluster"]))


df_eval.to_csv("llm_cluster_eval_results.csv", index=False)
print("\nSaved results to llm_cluster_eval_results.csv")
