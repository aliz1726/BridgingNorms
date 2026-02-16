import os
import requests
import pandas as pd
import json
import re
import argparse
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = input("Enter your OpenRouter API key: ").strip()
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not provided")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "gpt-4o-mini"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

df = pd.read_csv(
    "data_training_selected_clusters_comments_and_rules.csv",
    usecols=['body', 'subreddit_id', 'assigned_rule_cluster'],
    index_col=False
)

df = df.reset_index().rename(columns={'index': 'comment_id'})

sample_size = 120
eval_df = df.sample(120, random_state=2)
few_shot_pool = df.drop(eval_df.index)

few_shot_examples = few_shot_pool[['body', 'assigned_rule_cluster']] \
    .sample(20, random_state=1)
parser = argparse.ArgumentParser()
parser.add_argument(
    "mode",
    choices=["zero", "few"],
    help="Classification mode"
)
args = parser.parse_args()

mode = args.mode

if mode not in ["zero", "few"]:
    print("Invalid input, defaulting to zero-shot")
    mode = "zero"

if mode == "zero":
    few_shot_examples = None
# llm input
LLM_INPUT_LOG = []
LLM_OUTPUT_LOG = []

def llm_cluster_classification(comment, comment_id=None, mode="zero", few_shot_examples=None):
    if mode == "few" and few_shot_examples is not None:
        few_shot_prompt = ""
        for _, row in few_shot_examples.iterrows():
            few_shot_prompt += f'Comment: "{row["body"]}"\nCluster: {row["assigned_rule_cluster"]}\n\n'
        prompt = f"""
You are a moderation assistant. Predict which moderation rule cluster a Reddit comment belongs to.
Return ONLY the cluster name.

{few_shot_prompt}
Classify these comments:
\"\"\"{comment}\"\"\"
Cluster:
"""
    else:
        prompt = f"""
You are a moderation assistant. Predict which moderation rule cluster a Reddit comment belongs to.
Return ONLY the cluster name.

Comment:
\"\"\"{comment}\"\"\"
Cluster:
"""

    messages = [
        {"role": "system", "content": "You classify Reddit comments into moderation rule clusters."},
        {"role": "user", "content": prompt}
    ]

    data = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0
    }
    LLM_INPUT_LOG.append({
        "comment_id": comment_id,
        "request": data
    })
    try:
        response = requests.post(API_URL, headers=HEADERS, json=data)
        response.raise_for_status()
        cluster = response.json()['choices'][0]['message']['content'].strip()

        LLM_OUTPUT_LOG.append({
            "comment_id": comment_id,
            "comment": comment,
            "predicted_cluster": cluster
        })


        return cluster
    except Exception as e:
        print(f"LLM error: {e}")
        return "error"

eval_df["predicted_cluster"] = eval_df.apply(
    lambda row: llm_cluster_classification(
        row["body"],
        comment_id=row["comment_id"],
        mode=mode,
        few_shot_examples=few_shot_examples
    ),
    axis=1
)

output = []
for cluster, group in eval_df.groupby('predicted_cluster'):
    output.append({
        "cluster": cluster,
        "ids": group['comment_id'].tolist()
    })

eval_df['assigned_cluster_norm'] = eval_df['assigned_rule_cluster'].astype(str).str.strip()
def normalize_cluster(cluster):
    if not cluster:
        return "error"
    match = re.search(r'C\s*-\s*\d+', cluster)
    if match:
        return match.group(0)
    match = re.search(r'\b\d+\b', cluster)
    if match:
        return f"C - {match.group(0)}"
    return "error"

eval_df['predicted_cluster_norm'] = eval_df['predicted_cluster'].apply(normalize_cluster)
eval_df['assigned_cluster_norm'] = eval_df['assigned_rule_cluster'].astype(str).str.strip()

accuracy = (eval_df['assigned_cluster_norm'] == eval_df['predicted_cluster_norm']).mean()
print(f"\nExact-match accuracy: {accuracy:.2%}")

confusion = pd.crosstab(
    eval_df['assigned_cluster_norm'],
    eval_df['predicted_cluster_norm'],
    rownames=['Actual'],
    colnames=['Predicted'],
    dropna=False
)
pd.DataFrame({
    "metric": ["accuracy"],
    "value": [accuracy]
}).to_csv("cluster_accuracy.csv", index=False)

confusion.to_csv("cluster_confusion_matrix.csv")

confusion_pct = confusion.div(confusion.sum(axis=1), axis=0)
confusion_pct.to_csv("cluster_confusion_matrix_percent.csv")
print("\nConfusion Matrix:")
print(confusion)

eval_df.to_csv("cluster_predictions_comparison.csv", index=False)
print("\nSaved full comparison to cluster_predictions_comparison.csv")

print(json.dumps(output, indent=2))

with open("cluster_classification_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nSaved to cluster_classification_results.json")

with open("llm_inputs.json", "w") as f:
    json.dump(LLM_INPUT_LOG, f, indent=2)

print("Saved LLM inputs → llm_inputs.json")

with open("llm_outputs.json", "w") as f:
    json.dump(LLM_OUTPUT_LOG, f, indent=2)

print("Saved LLM outputs → llm_outputs.json")
