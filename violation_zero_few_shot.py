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
    usecols=['body', 'subreddit_id', 'label', 'assigned_rule_cluster', 'target_reason'],
    index_col=False
)

df = df.reset_index().rename(columns={'index': 'comment_id'})

sample_df = df.sample(120, random_state=2).reset_index(drop=True)

# ask user if they want zero or few shot, default to zero
mode = input("Choose classification mode ('zero' for zero-shot, 'few' for few-shot): ").strip().lower()
if mode not in ["zero", "few"]:
    print("Invalid input, defaulting to zero-shot")
    mode = "zero"

# if few, get a few examples
few_shot_examples = None
if mode == "few":
    few_shot_examples = sample_df[['body', 'label']].sample(min(20, len(sample_df)), random_state=1)


def llm_violation_classification(comment, mode="zero", few_shot_examples=None):
    if mode == "few" and few_shot_examples is not None:
        few_shot_prompt = ""
        for _, row in few_shot_examples.iterrows():
            few_shot_prompt += f'Comment: "{row["body"]}"\nLabel: {row["label"]}\n\n'

        prompt = f"""
You are a moderation assistant. Decide if a Reddit comment violates community rules.
Return ONLY "violation" or "non_violation".

{few_shot_prompt}
Now classify this comment:
\"\"\"{comment}\"\"\"
Label:
"""
    else: 
        prompt = f"""
You are a moderation assistant.

Decide whether the following Reddit comment violates the community rules.
Return ONLY "violation" or "non_violation".

Comment:
\"\"\"{comment}\"\"\"
"""

    messages = [
        {"role": "system", "content": "You classify Reddit comments as violation or non_violation."},
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
        label = response.json()['choices'][0]['message']['content'].strip().lower()
        if label not in ["violation", "non_violation"]:
            return "error"
        return label
    except Exception as e:
        print(f"LLM error: {e}")
        return "error"


print(f"Classifying comments using {mode}-shot LLM...")
sample_df['predicted_label'] = sample_df['body'].apply(
    lambda x: llm_violation_classification(x, mode=mode, few_shot_examples=few_shot_examples)
)

violations_df = sample_df[sample_df['predicted_label'] == 'violation']

output = []
for theme, group in violations_df.groupby('assigned_rule_cluster'):
    output.append({
        "theme": theme,
        "ids": group['comment_id'].tolist(),
        "reasoning": group['target_reason'].tolist()
    })


print(json.dumps(output, indent=2))

with open("violations_with_reasoning_and_llm_classification.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nSaved to violations_with_reasoning_and_llm_classification.json")
