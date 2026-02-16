import os
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import random

# Load API configuration
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = input("Enter your OpenRouter API key: ").strip()
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not provided")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o-mini"

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
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load moderated comments dataset.
        
        Expected columns:
        - body: the comment content
        - subreddit_id: identifier for the community
        - label: 0 if kept, 1 if removed (violation)
        """
        df = pd.read_csv(filepath)
        required_cols = ['body', 'subreddit_id', 'label']
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Dataset must contain columns: {required_cols}. Found: {df.columns.tolist()}")
        
        df = df.rename(columns={
            'body': 'comment_text',
            'subreddit_id': 'community_id',
            'label': 'violation'
        })
        
        df['violation'] = df['violation'].astype(bool)
        
        return df
    
    def sample_comments(
        self,
        df: pd.DataFrame,
        community_a: str,
        community_b: str,
        n_samples: int = 100,
        max_length: Optional[int] = None,
        balance_violations: bool = True
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
            
            if max_length:
                community_df = community_df[
                    community_df['comment_text'].str.len() <= max_length
                ]
            
            if balance_violations:
                n_per_class = n_samples // 2
                
                violating_df = community_df[community_df['violation'] == True]
                non_violating_df = community_df[community_df['violation'] == False]
                
                n_violating = min(n_per_class, len(violating_df))
                n_non_violating = min(n_per_class, len(non_violating_df))
                
                if n_violating == 0:
                    print(f"Warning: No violating comments for {community_id}, sampling all non-violating")
                    sampled = non_violating_df.sample(min(n_samples, len(non_violating_df)), random_state=42)
                elif n_non_violating == 0:
                    print(f"Warning: No non-violating comments for {community_id}, sampling all violating")
                    sampled = violating_df.sample(min(n_samples, len(violating_df)), random_state=42)
                else:
                    violating = violating_df.sample(n_violating, random_state=42)
                    non_violating = non_violating_df.sample(n_non_violating, random_state=42)
                    sampled = pd.concat([violating, non_violating])
            else:
                sampled = community_df.sample(
                    min(n_samples, len(community_df)),
                    random_state=42
                )
            
            for idx, row in sampled.iterrows():
                comment_id = f"{community_id[:1].upper()}{idx}"
                violation_status = "violation" if row['violation'] else "non_violation"
                samples.append((
                    comment_id,
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

Please output the information in the following format:
Norm [Norm Number]: [Norm Name, parsed as either restrictive or prescriptive]
- Definition of Norm: [General description of the norm]
- Community 0: [Definition as particularly understood in Community 0]
  – Relevant Comments for Community 0 definition: [list of all comments ids from Community 0 that meet and define this speech norm, without returning the comment text]
- Community 1: [Definition as particularly understood in Community 1]
  – Relevant Comments for Community 1 definition: [list of all comments ids from Community 1 that meet and define this speech norm, without returning the comment text]
"""
        return prompt
    
    def create_task2_prompt(
        self,
        comments: List[Tuple[str, str, str, str]],
        norm: str = "be respectful",
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
            violation_comments = [c for c in comments if c[3] == "violation"]
            
            comment_list = "\n".join([
                f"({cid}, \"{text}\", {comm})"
                for cid, text, comm, _ in violation_comments
            ])
            
            prompt = f"""Below there is a list of {len(violation_comments)} tuples with format (comment id, comment text, community id). These tuples represent comments from an online discussion platform from two different community ids.

Here are the comments:
{comment_list}

All these comments have been moderated and removed because they were considered disrespectful. How does each community define disrespect? Focus on what makes each community's definition different from the other:

Please output the information a dictionary format with the following key-value pairs:
comm_a_desc: [Definition of the norm as particularly understood in Community A]
comm_a_ids: [list of all comments ids from Community 0 that represent this speech norm in brackets (ie. [A23, A41]), without returning the comment text]
comm_b_desc: [Definition of the norm as particularly understood in Community B]
comm_b_ids: [list of all comments ids from Community 1 that represent this speech norm in brackets (ie. [B42, B630]), without returning the comment text]

Make sure to cite only comments from community A–those ids that begin with the letter A in the list of comments for community A. And similarly for B–those ids that begin with the letter B in the list of comments for community B.
"""
        else:
            community_id = comments[0][2]
            violation_comments = [
                c for c in comments 
                if c[3] == "violation" and c[2] == community_id
            ]
            
            comment_list = "\n".join([
                f"({cid}, \"{text}\")"
                for cid, text, _, _ in violation_comments
            ])
            
            prompt = f"""Below there is a list of {len(violation_comments)} comments that have been moderated and removed because they were considered disrespectful.

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

    def _extract_comment_ids(self, text: str) -> List[str]:
        """Extract comment IDs from text - improved version"""
        import re
        
        patterns = [
            r'\[([A-Z]\d+)\]',
            r'\b([A-Z]\d+)\b',  
            r'([A-Z]\d+)',       
        ]
        
        all_matches = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            all_matches.extend(matches)
            if matches: 
                break
        
        seen = set()
        unique_matches = []
        for match in all_matches:
            if match not in seen:
                seen.add(match)
                unique_matches.append(match)
        
        return unique_matches

    
    def parse_task2_response(self, response: str) -> Dict:
        """
        Parse LLM response for Task 2.
        
        Returns:
            Dictionary with community definitions
        """
        result = {
            'comm_a_desc': '',
            'comm_a_ids': [],
            'comm_b_desc': '',
            'comm_b_ids': []
        }
        
        for line in response.split('\n'):
            line = line.strip()
            
            if 'comm_a_desc:' in line:
                result['comm_a_desc'] = line.split(':', 1)[1].strip()
            elif 'comm_a_ids:' in line:
                result['comm_a_ids'] = self._extract_comment_ids(line)
            elif 'comm_b_desc:' in line:
                result['comm_b_desc'] = line.split(':', 1)[1].strip()
            elif 'comm_b_ids:' in line:
                result['comm_b_ids'] = self._extract_comment_ids(line)
        
        return result
    
    def calculate_metrics(
        self,
        parsed_response: Dict,
        input_comments: List[Tuple[str, str, str, str]],
        task: int = 1
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
        
        cited_violations = 0
        for comment_id, _, _, status in input_comments:
            if comment_id in all_cited and status == "violation":
                cited_violations += 1
        
        metrics['violating_fraction'] = (
            cited_violations / len(all_cited) if len(all_cited) > 0 else 0
        )
        
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
    
    def run_task1(
        self,
        df: pd.DataFrame,
        community_a: str,
        community_b: str,
        n_samples: int = 100,
        num_norms: int = 5,
        max_length: Optional[int] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Run complete Task 1 pipeline.
        
        Returns:
            Dictionary with results and metrics
        """
        print(f"Sampling comments from {community_a} and {community_b}...")
        comments = self.sample_comments(
            df, community_a, community_b,
            n_samples=n_samples,
            max_length=max_length
        )
        
        if len(comments) == 0:
            print("ERROR: No comments sampled!")
            return {
                'norms': {},
                'metrics': {},
                'raw_response': '',
                'input_comments': []
            }
        
        print(f"Sampled {len(comments)} comments total")
        
        print("Creating prompt...")
        prompt = self.create_task1_prompt(comments, num_norms=num_norms)
        
        if verbose:
            print(f"\n{'='*60}")
            print("PROMPT PREVIEW:")
            print(prompt)
            print(f"{'='*60}\n")
        
        print("Calling LLM...")
        response = self.call_llm(prompt)
        
        if verbose:
            print(f"\n{'='*60}")
            print("RAW LLM RESPONSE:")
            print(response)
            print(f"{'='*60}\n")
        
        print("Parsing response...")
        parsed = self.parse_task1_response(response)
        
        if not parsed:
            print("WARNING: Parsing returned empty results!")
            print("This might be a parsing issue. Check the raw response above.")
        else:
            print(f"Successfully parsed {len(parsed)} norms")
        
        print("Calculating metrics...")
        metrics = self.calculate_metrics(parsed, comments, task=1)
        
        return {
            'norms': parsed,
            'metrics': metrics,
            'raw_response': response,
            'input_comments': comments,
            'prompt': prompt 
        }
    def run_task2(
        self,
        df: pd.DataFrame,
        community_a: str,
        community_b: str,
        n_samples: int = 100,
        max_length: Optional[int] = None,
        joint: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Run complete Task 2 pipeline.
        
        Returns:
            Dictionary with results and metrics
        """
        print(f"Sampling comments from {community_a} and {community_b}...")
        comments = self.sample_comments(
            df, community_a, community_b,
            n_samples=n_samples,
            max_length=max_length,
            balance_violations=False  
        )
        
        violation_comments = [c for c in comments if c[3] == "violation"]
        
        if len(violation_comments) == 0:
            print("Warning: No violation comments found!")
            return {
                'definitions': {},
                'metrics': {},
                'raw_response': '',
                'input_comments': []
            }
        
        print(f"Found {len(violation_comments)} violation comments")
        
        print("Creating prompt...")
        prompt = self.create_task2_prompt(violation_comments, joint=joint)
        
        if verbose:
            print(f"\n{'='*60}")
            print("PROMPT PREVIEW (first 500 chars):")
            print(prompt[:500])
            print(f"{'='*60}\n")
        
        print("Calling LLM...")
        response = self.call_llm(prompt)
        
        if verbose:
            print(f"\n{'='*60}")
            print("RAW LLM RESPONSE:")
            print(response)
            print(f"{'='*60}\n")
        
        print("Parsing response...")
        parsed = self.parse_task2_response(response)
        
        print("Calculating metrics...")
        metrics = self.calculate_metrics(parsed, violation_comments, task=2)
        
        if joint and parsed['comm_a_desc'] and parsed['comm_b_desc']:
            metrics['definition_similarity'] = self.calculate_similarity(
                parsed['comm_a_desc'],
                parsed['comm_b_desc']
            )
        
        return {
            'definitions': parsed,
            'metrics': metrics,
            'raw_response': response,
            'input_comments': violation_comments,
            'prompt': prompt
        }
    def run_multiple_simulations(
        self,
        df: pd.DataFrame,
        task: int,
        n_simulations: int = 10,
        **kwargs
    ) -> List[Dict]:
        """
        Run multiple simulations with different community pairs.
        
        Args:
            df: DataFrame with comments
            task: Task number (1 or 2)
            n_simulations: Number of simulations to run
            **kwargs: Additional arguments for run_task1 or run_task2
            
        Returns:
            List of results from each simulation
        """
        communities = df['community_id'].unique()
        
        if len(communities) < 2:
            raise ValueError(f"Need at least 2 communities, found {len(communities)}")
        
        results = []
        
        for i in range(n_simulations):
            print(f"\n=== Simulation {i+1}/{n_simulations} ===")
            
            comm_a, comm_b = random.sample(list(communities), 2)
            
            try:
                if task == 1:
                    result = self.run_task1(df, comm_a, comm_b, **kwargs)
                elif task == 2:
                    result = self.run_task2(df, comm_a, comm_b, **kwargs)
                else:
                    raise ValueError(f"Invalid task number: {task}")
                
                result['community_a'] = comm_a
                result['community_b'] = comm_b
                results.append(result)
            except Exception as e:
                print(f"Error in simulation {i+1}: {e}")
                continue
        
        return results

if __name__ == "__main__":
    analyzer = CommunityNormAnalyzer()
    
    df = analyzer.load_data("data_training_selected_clusters_comments_and_rules.csv")
    
    print("Available communities:")
    print(df['community_id'].value_counts())

    top_communities = df['community_id'].value_counts().head(2).index.tolist()
    
    if len(top_communities) >= 2:
        community_a, community_b = top_communities[0], top_communities[1]
        
        print("\n" + "="*50)
        print("Running Task 1: Identifying Shared Norms")
        print("="*50)
        
        task1_results = analyzer.run_task1(
            df,
            community_a=community_a,
            community_b=community_b,
            n_samples=50,
            num_norms=5,
            max_length=280
        )
        
        print("\nTask 1 Results:")
        print(json.dumps(task1_results['norms'], indent=2))
        print("\nMetrics:")
        print(json.dumps(task1_results['metrics'], indent=2))
    else:
        print("Not enough communities in dataset")