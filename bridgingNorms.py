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
import math
import random

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = input("Enter your OpenRouter API key: ").strip()
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not provided")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash"


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
        - norm: reason violation/non-violation
        - subreddit_id: identifier for the community
        - label: 0 if kept, 1 if removed (violation)
        """
        df = pd.read_csv(filepath)
        required_cols = ['body', 'target_reason', 'subreddit_id', 'label']
        print(df['label'].value_counts())
        print(df['label'].dtype)
        print(df['label'].head(20))
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Dataset must contain columns: {required_cols}. Found: {df.columns.tolist()}")
        
        df = df.rename(columns={
            'body': 'comment_text',
            'target_reason': 'norm',
            'subreddit_id': 'community_id',
            'label': 'violation'
        })
        
        df['violation'] = df['violation'].isin([1, "1", True, "violation"])
        df['comment_id'] = range(1, len(df) + 1)

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
                
                if len(violating_df) < n_per_class or len(non_violating_df) < n_per_class:
                    return []
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
    
    def infer_norms_prompt(self, comments: List[Tuple[str, str, str, str]], community_id: str) -> str:
        community_comments = [(cid, text, comm, status) for cid, text, comm, status in comments if comm == community_id]
        
        comment_list = "\n".join([
            f"({cid}, \"{text}\", {status})"
            for cid, text, comm, status in community_comments
        ])
        
        prompt = f"""Below are {len(community_comments)} comments from an online community, each labeled as either a violation (removed) or non_violation (kept).

    Based on the pattern of what gets removed vs. kept, infer the main speech norms of this community. Return only a comma-separated list of single words or short phrases (e.g. respect, civility, no hate speech). No sentences, no explanations, no JSON.

    Comments:
    {comment_list}
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
        community_a = comments[0][2]

        def prefix(cid, comm):
            return f"A{cid}" if comm == community_a else f"B{cid}"

        comment_list = "\n".join([
            f"({prefix(cid, comm)}, \"{text}\", {comm}, {status})"
            for cid, text, comm, status in comments
        ])
        if random.random() < 0.5:
            norm_one, norm_two = norm_a, norm_b
        else:
            norm_one, norm_two = norm_b, norm_a
        
        norm_str = str(norm_one) + ", " + str(norm_two)


        prompt = f"""Below there is a list of {len(comments)} tuples with format (comment id, comment text, community id, violation status). These tuples represent comments from an online discussion platform from two different community ids.

Here is the norm that these communities have used to moderate content: {norm_str}

Here are the comments:
{comment_list}.
These comments include both removed (violations) and kept (non-violations) comments.
Using these comments, cite an even mix of both violation and non-violation comments, showing how each community defines their norms.

From the moderations, focus on what makes each community's definition of their norm different from the other:

Please output the information exactly in a dictionary format with the following key-value pairs:

comm_a_desc: [Definition of the norm as particularly understood in Community A]
comm_a_ids: [list of all comments ids from Community 0 that represent this speech norm in brackets (ie. [A23, A41]), without returning the comment text]
comm_b_desc: [Definition of the norm as particularly understood in Community B]
comm_b_ids: [list of all comments ids from Community 1 that represent this speech norm in brackets (ie. [B42, B630]), without returning the comment text]

Make sure to cite only comments from community A–those ids that begin with the letter A in the list of comments for community A. And similarly for B–those ids that begin with the letter B in the list of comments for community B.
Important: do not reference any comment IDs inside comm_a_desc or comm_b_desc. IDs belong only in comm_a_ids and comm_b_ids. Do not include any additional text in the dictionary format.
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

        import re

        cleaned = (
            re.sub(r'```(?:python|json)?\s*|```', '', response)
            .translate(str.maketrans({
                '\u202f': ' ',
                '\u2011': '-',
                '\u2013': '-',
                '\u2014': '-',
                '\u201c': '"',
                '\u201d': '"',
                '\u201e': '"',
                '\u00ab': '"',
                '\u00bb': '"',
                '\u2018': "'",
                '\u2019': "'",
                '\u201a': "'",
            }))
        )

        cleaned = re.sub(r'"+([AB]\d+)"+|\b([AB]\d+)\b', r'"\1\2"', cleaned)
        cleaned = re.sub(r'^\*+|\*+$', '', cleaned).strip()
        try:
            match = list(re.finditer(r'\{.*\}', cleaned, re.DOTALL))
            if match:
                data = json.loads(match[-1].group()) 

                result['comm_a_desc'] = data.get('comm_a_desc', '')
                result['comm_b_desc'] = data.get('comm_b_desc', '')
                def clean_id(val):
                    if isinstance(val, str):
                        nums = re.findall(r'\d+', val)
                        return int(nums[0]) if nums else None
                    return int(val) if isinstance(val, (int, float)) else None

                raw_a = self._flatten_ids(data.get('comm_a_ids', []))
                raw_b = self._flatten_ids(data.get('comm_b_ids', []))
                
                result['comm_a_ids'] = [clean_id(x) for x in raw_a if clean_id(x) is not None]
                result['comm_b_ids'] = [clean_id(x) for x in raw_b if clean_id(x) is not None]

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
        all_cited = []
        valid_cited = []
        
        if task == 1:
            for norm_data in parsed_response.values():
                all_cited.extend(norm_data['community_0']['comments'])
                all_cited.extend(norm_data['community_1']['comments'])
        else: 
            all_cited = parsed_response.get('comm_a_ids', []) + parsed_response.get('comm_b_ids', [])
        
        input_id_to_status = {int(c[0]): c[3] for c in input_comments}
        valid_cited = [int(cid) for cid in all_cited if int(cid) in input_id_to_status]
        unique_valid = set(valid_cited)
        metrics['unique_comments'] = len(unique_valid)
        if task == 2:
            a_cited = [int(cid) for cid in parsed_response.get('comm_a_ids', []) if int(cid) in input_id_to_status]
            b_cited = [int(cid) for cid in parsed_response.get('comm_b_ids', []) if int(cid) in input_id_to_status]
            metrics['unique_comments_a'] = len(set(a_cited))
            metrics['unique_comments_b'] = len(set(b_cited))
        total_input = len(input_comments)
        
        metrics['coverage'] = len(unique_valid) / total_input if total_input > 0 else 0
        
        if len(all_cited) > 0:
            metrics['redundancy'] = 1 - (len(set(all_cited)) / len(all_cited))
        else:
            metrics['redundancy'] = 0

        if valid_cited:
            n_violations = sum(1 for cid in valid_cited if input_id_to_status[cid] == "violation")
            metrics['violating_fraction'] = n_violations / len(valid_cited)
        else:
            metrics['violating_fraction'] = 0.0
        
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
        if isinstance(text1, list):
            text1 = " ".join(map(str, text1))
        if isinstance(text2, list):
            text2 = " ".join(map(str, text2))
        text1 = str(text1)
        text2 = str(text2)
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
            )

        elif task_name == "task2":
            comments = self.sample_comments(
                df, community_a, community_b,
                n_samples=n_samples,
                balance_violations=True
            )
            norm_a = df[df['community_id'] == community_a]['norm'].unique().tolist()
            norm_b = df[df['community_id'] == community_b]['norm'].unique().tolist()
                

        else:
            raise ValueError("task_name must be 'task1' or 'task2'")
        if comments is None:
            raise NotEnoughSamplesError(f"Not enough samples for {community_a} or {community_b}")

        run_log["sampling"]["total_sampled"] = len(comments)

        if len(comments) == 0:
            run_log["warnings"].append("No comments sampled")
            return {}

        prompt = self.create_task2_prompt(comments, norm_a=norm_a, norm_b=norm_b, joint=joint)

        run_log["prompt"] = prompt

        if verbose:
            print("\n" + "="*60)
            print("PROMPT PREVIEW:")
            print(prompt[:500])
            print("="*60 + "\n")

        max_retries = 3

        parsed = None
        metrics = None
        response = None

        for attempt in range(max_retries):
            prompt_retry = prompt 

            if attempt > 0:
                prompt_retry += "\n\nIMPORTANT: Return ONLY valid JSON. No prose."

            if verbose:
                print(f"\n Attempt {attempt + 1}/{max_retries}")
            response = self.call_llm(prompt_retry)

            if task_name == "task1":
                parsed = self.parse_task1_response(response)
            else:
                parsed = self.parse_task2_response(response)
                print("PARSED OUTPUT:", parsed)

            if parsed is None or parsed == {}:
                run_log["warnings"].append(f"Attempt {attempt+1}: parse failed")
                continue

            task_num = 1 if task_name == "task1" else 2
            metrics = self.calculate_metrics(parsed, comments, task=task_num)

            vf = metrics.get("violating_fraction", None)

            empty_output = (
                len(parsed.get("comm_a_ids", [])) == 0 and
                len(parsed.get("comm_b_ids", [])) == 0
            )

            if (
                vf is None
                or (isinstance(vf, float) and math.isnan(vf))
                or metrics.get("unique_comments", 0) == 0
                or empty_output
            ):
                run_log["warnings"].append(f"Attempt {attempt+1}: invalid metrics")
                continue
            break

        else:
            return {
                "error": "all retries failed",
                "log": run_log
            }
        
        metrics["inputs"] = {
            "task_name": task_name,
            "community_a": community_a,
            "community_b": community_b,
            "n_samples": n_samples,
            "joint": joint,
            "num_input_comments": len(comments),
        }
        def normalize_text(x):
            if isinstance(x, list):
                return " ".join(map(str, x))
            if x is None:
                return ""
            return str(x)
        if task_name == "task2":
            metrics["inputs"]["definition_similarity"] = self.calculate_similarity(
                normalize_text(parsed["comm_a_desc"]),
                normalize_text(parsed["comm_b_desc"])
            )


        run_log["llm_response"].setdefault("attempts", []).append({
            "attempt": attempt,
            "response": response,
            "parsed": parsed
        })
        run_log["parsed_output"] = parsed
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

    def _flatten_ids(self, ids_field) -> List:
        """Flatten nested lists of IDs if the LLM returns grouped arrays."""
        if not ids_field:
            return []
        if isinstance(ids_field[0], list):
            return [item for sublist in ids_field for item in sublist]
        return ids_field
if __name__ == "__main__":
    analyzer = CommunityNormAnalyzer()
    if len(sys.argv) < 4:
        print("Error: not enough inputs")
        sys.exit(1)
    n_samples = int(sys.argv[2])
    df = analyzer.load_data("data_training_selected_clusters_comments_and_rules.csv", min_samp=n_samples//2)

    print("Available communities:")
    print(df['community_id'].value_counts())

    all_communities = df['community_id'].unique().tolist()

    while True:
        community_a, community_b = random.sample(all_communities, 2)
        try:
            task_results = analyzer.run_task(
                df,
                community_a=community_a,
                community_b=community_b,
                task_name="task2",
                n_samples=n_samples
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
