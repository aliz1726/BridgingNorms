import os
import requests
import pandas as pd
import json
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
    usecols=['body', 'subreddit_id', 'assigned_rule_cluster', 'target_reason'],
    index_col=False
)

df = df.reset_index().rename(columns={'index': 'comment_id'})

sample_size = 120
sample_df = df.sample(sample_size, random_state=2).reset_index(drop=True)

mode = input("Choose classification mode ('zero' or 'few'): ").strip().lower()
if mode not in ["zero", "few"]:
    print("Invalid input, defaulting to zero-shot")
    mode = "zero"

few_shot_examples = None
if mode == "few":
    few_shot_examples = sample_df[['body', 'assigned_rule_cluster']].sample(min(20, len(sample_df)), random_state=1)

def llm_cluster_classification(comment, mode="zero", few_shot_examples=None):
    if mode == "few" and few_shot_examples is not None:
        few_shot_prompt = ""
        for _, row in few_shot_examples.iterrows():
            few_shot_prompt += f'Comment: "{row["body"]}"\nCluster: {row["assigned_rule_cluster"]}\n\n'
        prompt = f"""
You are a moderation assistant. Predict which moderation rule cluster a Reddit comment belongs to.
Return ONLY the cluster name.

{few_shot_prompt}
Classify this comment:
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

    try:
        response = requests.post(API_URL, headers=HEADERS, json=data)
        response.raise_for_status()
        cluster = response.json()['choices'][0]['message']['content'].strip()
        return cluster
    except Exception as e:
        print(f"LLM error: {e}")
        return "error"

print(f"Classifying comments into clusters using {mode}-shot LLM...")
sample_df['predicted_cluster'] = sample_df['body'].apply(
    lambda x: llm_cluster_classification(x, mode=mode, few_shot_examples=few_shot_examples)
)

output = []
for cluster, group in sample_df.groupby('predicted_cluster'):
    output.append({
        "cluster": cluster,
        "ids": group['comment_id'].tolist(),
        "reasoning": group['target_reason'].tolist()
    })

sample_df['assigned_cluster_norm'] = sample_df['assigned_rule_cluster'].astype(str).str.strip()
sample_df['predicted_cluster_norm'] = sample_df['predicted_cluster'].astype(str).str.strip()

accuracy = (sample_df['assigned_cluster_norm'] == sample_df['predicted_cluster_norm']).mean()
print(f"\nExact-match accuracy: {accuracy:.2%}")

confusion = pd.crosstab(
    sample_df['assigned_cluster_norm'],
    sample_df['predicted_cluster_norm'],
    rownames=['Actual'],
    colnames=['Predicted'],
    dropna=False
)
print("\nConfusion Matrix:")
print(confusion)

sample_df.to_csv("cluster_predictions_comparison.csv", index=False)
print("\nSaved full comparison to cluster_predictions_comparison.csv")

print(json.dumps(output, indent=2))

with open("cluster_classification_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nSaved to cluster_classification_results.json")
