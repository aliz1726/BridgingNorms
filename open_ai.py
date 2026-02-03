import os
import requests
import pandas as pd
from dotenv import load_dotenv

# load api key
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not found in environment")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "gpt-4o-mini" 

# load dataset
df_reduced = pd.read_csv(
    "data_training_selected_clusters_comments_and_rules.csv",
    usecols=['body', 'subreddit_id', 'label'],
    index_col=False
)

dataset_text = df_reduced.sample(120, random_state=5)

# Add the CSV row index as a new column
dataset_text = dataset_text.reset_index()  # 'index' column now contains original row numbers

dataset_text.rename(columns={"index": "comment_id"}, inplace=True)
print(dataset_text)



df_clusters_reduced = pd.read_csv("data_training_selected_clusters_comments_and_rules.csv",
                usecols=['body', 'subreddit_id', 'assigned_rule_cluster', 'label'])

dataset_cluster_text = df_clusters_reduced.sample(120, random_state=2).to_string() 
# full data is too much, select random subset

# Initialize conversation with dataset in system prompt
conversation = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant. Use the dataset provided to answer questions.\n"
            f"Dataset:\n{dataset_text}"
        )
    }
]

print("Chat with the LLM (type 'exit' to quit):")

# continue until 'exit'
while True:
    user_input = input("> ")
    if user_input.lower() == "exit":
        break

    conversation.append({"role": "user", "content": user_input})

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {"model": MODEL, "messages": conversation}

    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        reply = f"Error reading response: {e}"

    print("LLM:", reply)
    conversation.append({"role": "assistant", "content": reply})
    