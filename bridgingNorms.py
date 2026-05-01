import os
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import random
import sys
from datetime import datetime

# Load API configuration
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = input("Enter your OpenRouter API key: ").strip()
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not provided")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"


class NotEnoughSamplesError(Exception):
    pass

class CommunityNormAnalyzer:
    """
    Implementation of "Bridging Norms: What Do We Have In Common?"
    Analyzes and compares speech norms across online communities.
    """
    
    def __init__(self, api_key: str = None, model: str = None, api_url: str = None):
        """
        Initialize the analyzer.
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY)
            model: LLM model to use (defaults to MODEL)
            api_url: API endpoint URL (defaults to API_URL)
        """
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = model or MODEL
        self.api_url = api_url or API_URL
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def load_data(self, filepath: str, min_samp: int) -> pd.DataFrame:
        """
        Load moderated comments dataset.
        
        Expected columns:
        - body: the comment content
        - subreddit_id: identifier for the community
        - label: 0 if kept, 1 if removed (violation)
        """
        df = pd.read_csv(filepath)
        required_cols = ['body', 'subreddit_id', 'label']
        print(df['label'].value_counts())
        print(df['label'].dtype)
        print(df['label'].head(20))
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Dataset must contain columns: {required_cols}. Found: {df.columns.tolist()}")
        
        df = df.rename(columns={
            'body': 'comment_text',
            'subreddit_id': 'community_id',
            'label': 'violation'
        })
        
        df['violation'] = df['violation'] == 'violation'
        df['comment_id'] = range(1, len(df) + 1)

        # Debug: show what we're working with before filtering
        counts = df.groupby('community_id')['violation'].agg(
            violations=lambda x: x.sum(),
            non_violations=lambda x: (~x).sum()
        )
        print("Per-community counts (before filtering):")
        print(counts)
        print(f"\nRequiring min {min_samp} per class")

        valid = counts[(counts['violations'] >= min_samp) & 
                    (counts['non_violations'] >= min_samp)].index
        
        before = df['community_id'].nunique()
        df = df[df['community_id'].isin(valid)]
        after = df['community_id'].nunique()
        print(f"Filtered communities: {before} -> {after} (removed {before - after} with fewer than {min_samp} samples per class)")

        if after < 2:
            raise ValueError(
                f"Not enough valid communities after filtering (found {after}). "
                f"Try lowering n_samples (currently requiring {min_samp} per class)."
            )

        return df
    
    def sample_comments(
        self,
        df: pd.DataFrame,
        community_a: str,
        community_b: str,
        n_samples: int,
        # max_length: Optional[int] = None,
        balance_violations: bool = False
    ) -> List[Tuple[str, str, str, str]]:
        """
        Sample comments from two communities.
        
        Args:
            df: DataFrame with comments
            community_a: ID of first community
            community_b: ID of second community
            n_samples: Number of comments to sample per community
            max_length: Maximum character length for comments (None = no limit)
            balance_violations: Whether to balance violating/non-violating samples
            
        Returns:
            List of tuples (comment_id, comment_text, community_id, violation_status)
        """
        samples = []
        
        for community_id in [community_a, community_b]:
            community_df = df[df['community_id'] == community_id].copy()
            
            if len(community_df) == 0:
                print(f"Warning: No comments found for community '{community_id}'")
                continue
            
            if balance_violations:
                n_per_class = n_samples // 2
                
                violating_df = community_df[community_df['violation'] == True]
                non_violating_df = community_df[community_df['violation'] == False]
                
                # stop if not enough, or redo or lower?
                if len(violating_df) < n_per_class or len(non_violating_df) < n_per_class:
                    # raise ValueError(
                    #     f"Community '{community_id}' does not have enough samples.\n"
                    #     f"Required per class: {n_per_class}\n"
                    #     f"Violations available: {len(violating_df)}\n"
                    #     f"Non-violations available: {len(non_violating_df)}"
                    # )
                    return None
                violating = violating_df.sample(n_per_class, random_state=42)
                non_violating = non_violating_df.sample(n_per_class, random_state=42)
                sampled = pd.concat([violating, non_violating])
            else:
                sampled = community_df.sample(
                    min(n_samples, len(community_df)),
                    random_state=42
                )
            
            for idx, row in sampled.iterrows():
                violation_status = "violation" if row['violation'] else "non_violation"
                samples.append((
                    row["comment_id"],
                    row['comment_text'],
                    community_id,
                    violation_status
                ))
        
        random.shuffle(samples)
        return samples
    
    def create_task1_prompt(
        self,
        comments: List[Tuple[str, str, str, str]],
        num_norms: Optional[int] = 5
    ) -> str:
        """
        Create prompt for Task 1: Identifying shared norms and differences.
        
        Args:
            comments: List of comment tuples
            num_norms: Number of norms to extract (None = unsupervised)
            
        Returns:
            Formatted prompt string
        """
        comment_list = "\n".join([
            f"({cid}, \"{text}\", {comm}, {status})"
            for cid, text, comm, status in comments
        ])
        
        num_norms_text = f"{num_norms}" if num_norms else "unsupervised number of"
        
        prompt = f"""Below there is a list of {len(comments)} tuples with format (comment id, comment text, community id, violation or non violation). These tuples represent comments from an online discussion platform from two different community ids.

In these online communities, a speech norm governs someone's speech and is defined as what is expected and allowed in someone's speech. These norms could refer to encouraged (prescriptive) behaviors or discouraged (restrictive) behaviors.

Given the above context, to answer the following question, please make sure to cite as many comments as possible by ID in brackets.

Question: By analyzing all comments, what are the {num_norms_text} main speech norms that govern these communities? Note that the violating comments are removed as they go against their policies, and non violating comments are kept as they obey them.

For each single shared norm, please provide:
1. A brief description of the norm.
2. The classification of prescriptive or restrictive for the norm.
3. An empirically-grounded definition of how this norm is *differently understood within each different community along with referencing in brackets all the comment IDs that exemplify this shared norm*. For example: [comment id], [next comment id]

If the norm is prescriptive, please write it in the form of: "Please [action/behavior]"
If the norm is restrictive, please write it in the form of: "No [action/behavior]"

Here are the comments:
{comment_list}

""" + """

Please output the information in the following format, as a json file, with one element per extracted norm.

output = [

{   
    "Norm 1": "[Norm Name, parsed as either restrictive or prescriptive]",
    "Community 0": "[Definition as particularly understood in Community 0]"
    "Community 0 Citations": "Relevant Comments for Community 0 definition: [list of all comments ids from Community 0 that meet and define this speech norm, without returning the comment text]"
    "Community 1": "[Definition as particularly understood in Community 1]"
    "Community 1 Citations": "Relevant Comments for Community 1 definition: [list of all comments ids from Community 1 that meet and define this speech norm, without returning the comment text]"

},

{   
    "Norm [new norm number]": "[Norm Name, parsed as either restrictive or prescriptive]",
    "Community 0": "[Definition as particularly understood in Community 0]"
    "Community 0 Citations": "Relevant Comments for Community 0 definition: [list of all comments ids from Community 0 that meet and define this speech norm, without returning the comment text]"
    "Community 1": "[Definition as particularly understood in Community 1]"
    "Community 1 Citations": "Relevant Comments for Community 1 definition: [list of all comments ids from Community 1 that meet and define this speech norm, without returning the comment text]"

}
]
"""
        return prompt
    
    def create_task2_prompt(
        self,
        comments: List[Tuple[str, str, str, str]],
        norm_a: str,
        norm_b: str,
        joint: bool = True
    ) -> str:
        """
        Create prompt for Task 2: Analyzing interpretations of a single norm.
        
        Args:
            comments: List of comment tuples (should be violations only)
            norm: The specific norm to analyze
            joint: If True, compare both communities; if False, analyze independently
            
        Returns:
            Formatted prompt string
        """
        if joint:
            community_a = comments[0][2]

            def prefix(cid, comm):
                return f"A{cid}" if comm == community_a else f"B{cid}"

            comment_list = "\n".join([
                f"({prefix(cid, comm)}, \"{text}\", {comm}, {status})"
                for cid, text, comm, status in comments
            ])
            
            prompt = f"""Below there is a list of {len(comments)} tuples with format (comment id, comment text, community id, violation status). These tuples represent comments from an online discussion platform from two different community ids.

            {norm_a + norm_b}

Here are the comments:
{comment_list}

These comments include both removed (violations) and kept (non-violations) comments.
How does each community define disrespect? Make sure to cite a good mix of both violation and non-violation comments in your answer.
Focus on what makes each community's definition different from the other:

Please output the information a dictionary format with the following key-value pairs:
comm_a_desc: [Definition of the norm as particularly understood in Community A]
comm_a_ids: [list of all comments ids from Community 0 that represent this speech norm in brackets (ie. [A23, A41]), without returning the comment text]
comm_b_desc: [Definition of the norm as particularly understood in Community B]
comm_b_ids: [list of all comments ids from Community 1 that represent this speech norm in brackets (ie. [B42, B630]), without returning the comment text]

Make sure to cite only comments from community A–those ids that begin with the letter A in the list of comments for community A. And similarly for B–those ids that begin with the letter B in the list of comments for community B.
"""
        else:
            community_a = comments[0][2]

            def prefix(cid, comm):
                return f"A{cid}" if comm == community_a else f"B{cid}"

            comment_list = "\n".join([
                f"({prefix(cid, comm)}, \"{text}\", {comm}, {status})"
                for cid, text, comm, status in comments
            ])
            
            prompt = f"""Below there is a list of {len(comments)} tuples with format (comment id, comment text, community id, violation status). These tuples represent comments from an online discussion platform from two different community ids.

Here are the comments:
{comment_list}

How does this community define disrespect? Provide:
1. A detailed definition of what constitutes disrespect in this community
2. A list of comment IDs that exemplify this definition in brackets

Please format your response as:
definition: [Your definition]
comment_ids: [List of relevant comment IDs in brackets]
"""
        
        return prompt
    
    def call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Call the LLM via OpenRouter with the given prompt.
        
        Args:
            prompt: The prompt to send
            temperature: Sampling temperature
            
        Returns:
            LLM response text
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature
        }
        
        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def parse_task1_response(self, response: str) -> Dict:
        """
        Parse LLM response for Task 1 with improved robustness.
        
        Returns:
            Dictionary with norms and their definitions
        """
        import re
        
        norms = {}
        current_norm = None
        current_norm_key = None
        
        lines = response.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            norm_match = re.match(r'[\*\s]*Norm\s*[\[\(]?\s*(\d+)\s*[\]\)]?\s*:?\s*(.+)', line_stripped, re.IGNORECASE)
            if norm_match:
                norm_number = norm_match.group(1)
                norm_name = norm_match.group(2).strip().rstrip('*:')
                current_norm_key = f"Norm {norm_number}: {norm_name}"
                current_norm = current_norm_key
                norms[current_norm] = {
                    'definition': '',
                    'community_0': {'definition': '', 'comments': []},
                    'community_1': {'definition': '', 'comments': []}
                }
                continue
            
            if not current_norm:
                continue

            if 'definition of norm' in line_stripped.lower() or 'definition:' in line_stripped.lower():
                parts = re.split(r':\s*', line_stripped, maxsplit=1)
                if len(parts) == 2:
                    norms[current_norm]['definition'] = parts[1].strip()
                elif i + 1 < len(lines):
                    norms[current_norm]['definition'] = lines[i + 1].strip()
            
            community_match = re.match(r'[\*\s]*-?\s*Community\s+(\d+)\s*:?\s*(.+)', line_stripped, re.IGNORECASE)
            if community_match:
                comm_num = community_match.group(1)
                comm_def = community_match.group(2).strip()
                norms[current_norm][f'community_{comm_num}']['definition'] = comm_def
                continue
            
            relevant_match = re.search(r'relevant\s+comments?\s+for\s+community\s+(\d+)', line_stripped, re.IGNORECASE)
            if relevant_match:
                comm_num = relevant_match.group(1)
                comment_ids = self._extract_comment_ids(line_stripped)
                
                if not comment_ids and i + 1 < len(lines):
                    comment_ids = self._extract_comment_ids(lines[i + 1])
                
                norms[current_norm][f'community_{comm_num}']['comments'] = comment_ids
        
        return norms

    def _extract_comment_ids(self, text: str) -> List[int]:
        """Extract numeric comment IDs"""

        matches = re.findall(r'\d+', text)
        print("TEXT:", text)
        seen = set()
        unique = []

        for m in matches:
            num = int(m)
            if num not in seen:
                seen.add(num)
                unique.append(num)

        return unique

    def parse_task2_response(self, response: str) -> Dict:
        result = {
            'comm_a_desc': '',
            'comm_a_ids': [],
            'comm_b_desc': '',
            'comm_b_ids': []
        }

        cleaned = re.sub(r'```(?:python|json)?\s*', '', response)
        cleaned = re.sub(r'```', '', cleaned).strip()
        cleaned = re.sub(r'"+([AB]\d+)"+', r'"\1"', cleaned)
        cleaned = re.sub(r'\b([AB]\d+)\b', r'"\1"', cleaned)
        cleaned = re.sub(r'"+([AB]\d+)"+', r'"\1"', cleaned)

        try:
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                data = json.loads(match.group())

                desc_a = data.get('comm_a_desc', '')
                desc_b = data.get('comm_b_desc', '')
                result['comm_a_desc'] = desc_a[0] if isinstance(desc_a, list) else desc_a
                result['comm_b_desc'] = desc_b[0] if isinstance(desc_b, list) else desc_b

                result['comm_a_ids'] = [
                    int(re.sub(r'[A-Za-z]', '', id_str))
                    for id_str in data.get('comm_a_ids', [])
                    if re.search(r'\d+', str(id_str))
                ]
                result['comm_b_ids'] = [
                    int(re.sub(r'[A-Za-z]', '', id_str))
                    for id_str in data.get('comm_b_ids', [])
                    if re.search(r'\d+', str(id_str))
                ]

                min_len = min(len(result['comm_a_ids']), len(result['comm_b_ids']))
                result['comm_a_ids'] = result['comm_a_ids'][:min_len]
                result['comm_b_ids'] = result['comm_b_ids'][:min_len]

                return result

        except (json.JSONDecodeError, TypeError) as e:
            print("JSON parse failed:", e)
            print("Cleaned text was:", cleaned)

        return result

    def calculate_metrics(
        self,
        parsed_response: Dict,
        input_comments: List[Tuple[str, str, str, str]],
        task: int
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            parsed_response: Parsed LLM output
            input_comments: Original input comments
            task: Task number (1 or 2)
            
        Returns:
            Dictionary of metric values
        """
        metrics = {}
        metrics["task"] = task
        
        if task == 1:
            all_cited = []
            for norm_data in parsed_response.values():
                all_cited.extend(norm_data['community_0']['comments'])
                all_cited.extend(norm_data['community_1']['comments'])
        else: 
            all_cited = parsed_response['comm_a_ids'] + parsed_response['comm_b_ids']

        metrics['unique_comments'] = len(set(all_cited))
        
        total_input = len(input_comments)
        metrics['coverage'] = len(set(all_cited)) / total_input if total_input > 0 else 0
        
        if len(all_cited) > 0:
            metrics['redundancy'] = 1 - (len(set(all_cited)) / len(all_cited))
        else:
            metrics['redundancy'] = 0
        

        input_id_to_status = {c[0]: c[3] for c in input_comments}
        valid_cited = [cid for cid in all_cited if cid in input_id_to_status]

        if valid_cited:
            n_violations = sum(1 for cid in valid_cited if input_id_to_status[cid] == "violation")
            metrics['violating_fraction'] = n_violations / len(valid_cited)
        else:
            metrics['violating_fraction'] = float('nan')
        
        metrics['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return metrics
    
    def calculate_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Calculate cosine similarity between two texts using embeddings.
        """
        embeddings = self.embedding_model.encode([text1, text2])
        
        vec1 = embeddings[0]
        vec2 = embeddings[1]
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        similarity = dot_product / (norm1 * norm2)
        
        return float(similarity)

    def run_task(
        self,
        df: pd.DataFrame,
        community_a: str,
        community_b: str,
        task_name: str,
        n_samples: int,
        num_norms: int,
        # max_length: Optional[int] = None,
        joint: bool = True,
        verbose: bool = True
    ) -> Dict:
        reason = input("Why are you running this test? ")
        run_log = {
            "task": {
                "name": task_name,
                "community_a": community_a,
                "community_b": community_b,
                "n_samples": n_samples
                # "max_length": max_length
            },
            "warnings": [],
            "sampling": {},
            "prompt": None,
            "llm_response": {},
            "parsed_output": None,
            "metrics": None,
            "reason": reason
        }
        comments = None
        norm_a = None
        norm_b = None
        if task_name == "task1":
            comments = self.sample_comments(
                df, community_a, community_b,
                n_samples=n_samples
                # max_length=max_length
            )

        elif task_name == "task2":
            comments = self.sample_comments(
                df, community_a, community_b,
                n_samples=n_samples,
                # max_length=max_length
                balance_violations=True
            )
            norm_a = " | ".join(df[df['community_id'] == community_a]['assigned_rule_cluster'].dropna().unique())
            norm_b = " | ".join(df[df['community_id'] == community_b]['assigned_rule_cluster'].dropna().unique())
                

        else:
            raise ValueError("task_name must be 'task1' or 'task2'")
        if comments is None:
            raise NotEnoughSamplesError(f"Not enough samples for {community_a} or {community_b}")

        run_log["sampling"]["total_sampled"] = len(comments)

        if len(comments) == 0:
            run_log["warnings"].append("No comments sampled")
            return {}

        if task_name == "task1":
            prompt = self.create_task1_prompt(comments, num_norms=num_norms)
        elif task_name == "task2":
            norm_a = " | ".join(df[df['community_id'] == community_a]['assigned_rule_cluster'].dropna().unique())
            norm_b = " | ".join(df[df['community_id'] == community_b]['assigned_rule_cluster'].dropna().unique())
            prompt = self.create_task2_prompt(comments, norm_a=norm_a, norm_b=norm_b, joint=joint)

        run_log["prompt"] = prompt

        if verbose:
            print("\n" + "="*60)
            print("PROMPT PREVIEW:")
            print(prompt[:500])
            print("="*60 + "\n")


        response = self.call_llm(prompt)
        run_log["llm_response"]["raw_response"] = response

        if verbose:
            print("\n" + "="*60)
            print("RAW LLM RESPONSE:")
            print(response)
            print("="*60 + "\n")

        if task_name == "task1":
            parsed = self.parse_task1_response(response)

        else:
            parsed = self.parse_task2_response(response)
            print("PARSED OUTPUT:", parsed)  # add this

        run_log["parsed_output"] = parsed

        if not parsed:
            run_log["warnings"].append("Parsing returned empty results")

        task_num = 1 if task_name == "task1" else 2
        metrics = self.calculate_metrics(parsed, comments, task=task_num)

        metrics["inputs"] = {
            "task_name": task_name,
            "community_a": community_a,
            "community_b": community_b,
            "n_samples": n_samples,
            "num_norms": num_norms,
            # "max_length": max_length,
            "joint": joint,
            "num_input_comments": len(comments),
        }
        if task_name == "task2":
            metrics["inputs"]["definition_similarity"] = self.calculate_similarity(
            parsed["comm_a_desc"], parsed["comm_b_desc"]
            )


        run_log["metrics"] = metrics

        with open(f"{task_name}_run_log.json", "w") as f:
            json.dump(run_log, f, indent=2, ensure_ascii=False)

        return {
            "norms": parsed,
            "metrics": metrics,
            "raw_response": response,
            "input_comments": comments,
            "prompt": prompt
        }


if __name__ == "__main__":
    analyzer = CommunityNormAnalyzer()
    if len(sys.argv) < 4:
        print("Error: not enough inputs")
        sys.exit(1)
    n_samples = int(sys.argv[2])
    df = analyzer.load_data("data_training_selected_clusters_comments_and_rules.csv", min_samp=n_samples//2)

    print("Available communities:")
    print(df['community_id'].value_counts())

    task_name = sys.argv[1]
    num_norms = int(sys.argv[3])
    all_communities = df['community_id'].unique().tolist()

    while True:
        community_a, community_b = random.sample(all_communities, 2)
        try:
            task_results = analyzer.run_task(
                df,
                community_a=community_a,
                community_b=community_b,
                task_name="task2",
                n_samples=n_samples,
                num_norms=2
                # max_length=280
            )
            break
        except NotEnoughSamplesError as e:
            print(f"{e}, retrying with new communities...")

    output = {
        "community_a": community_a,
        "community_b": community_b,
        "prompt": task_results["prompt"],
        "raw_response": task_results["raw_response"],
        "input_comments": [
            {"comment_id": cid, "text": text, "community": comm, "status": status}
            for cid, text, comm, status in task_results["input_comments"]
        ]
    }
    with open(f"{task_name}_results.json", "w") as f:
        json.dump(output, f, indent=2)
