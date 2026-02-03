import os
import requests
import pandas as pd
import json
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

df = pd.read_csv(
    "data_training_selected_clusters_comments_and_rules.csv",
    usecols=['body', 'subreddit_id', 'label', 'assigned_rule_cluster', 'target_reason'],
    index_col=False
)
df = df.reset_index().rename(columns={'index': 'comment_id'})
sample_df = df.sample(120, random_state=2)


violations_df = sample_df[sample_df['label'] == 'violation']

# ask llm to rate explicitness from 0-1, could add more (weigh heavily on language? context?)
def llm_explicitness(comment):
    prompt = f"""
Rate the explicitness or aggressiveness of this comment on a scale from 0 to 1,
where 0 = completely neutral / non-aggressive, and 1 = extremely aggressive, offensive, or hostile.

Comment:
\"\"\"{comment}\"\"\"
Return only a number between 0 and 1.
"""
    messages = [
        {"role": "system", "content": "You are an assistant that scores comments for explicit/aggressive language."},
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
        score_text = response.json()['choices'][0]['message']['content'].strip()
        score = float(score_text)
        return min(max(score, 0), 1)
    except Exception as e:
        print(f"LLM error: {e}")
        return 0 


violations_df['explicit_score'] = violations_df['body'].apply(llm_explicitness)

# group by violation
output = []
for theme, group in violations_df.groupby('assigned_rule_cluster'):
    output.append({
        "theme": theme,
        "ids": group['comment_id'].tolist(),
        "reasoning": group['target_reason'].tolist(),
        "average_explicit_score": group['explicit_score'].mean()
    })

print(json.dumps(output, indent=2))

with open("violations_with_reasoning_and_llm_explicitness.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nSaved to violations_with_reasoning_and_llm_explicitness.json")
