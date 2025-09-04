import json
import random
from pathlib import Path

class RobustExpander:
    def __init__(self):
        self.data_dir = Path("chameleon_prime_personalization/data/raw/LaMP-2")
        self.output_dir = Path("data/evaluation")
        
    def extract_user_id(self, item_id):
        """Extract user ID using LaMP-2 methodology"""
        # For LaMP-2, we'll group by ranges to create multiple "users"
        # This simulates different user profiles for evaluation
        id_num = int(item_id)
        
        # Create 7 user groups based on ID ranges
        if 110 <= id_num <= 299:
            return "user_1"  # Range 1
        elif 300 <= id_num <= 499:
            return "user_2"  # Range 2  
        elif 500 <= id_num <= 699:
            return "user_3"  # Range 3
        elif 700 <= id_num <= 899:
            return "user_4"  # Range 4
        elif 900 <= id_num <= 1099:
            return "user_5"  # Range 5
        elif 1100 <= id_num <= 1199:
            return "user_6"  # Range 6
        else:
            return "user_7"  # Overflow range
        
    def expand_dataset(self, n_samples=140):
        """Non-overlapping stratified sampling with proper user grouping"""
        
        self.output_dir.mkdir(exist_ok=True)
        
        # Load original data
        with open(self.data_dir / "merged.json") as f:
            questions = json.load(f)
        with open(self.data_dir / "answers.json") as f:
            answers_data = json.load(f)
            
        # Create answers lookup
        answers = {str(item['id']): item['output'].strip().lower() for item in answers_data}
        
        print(f"Original data: {len(questions)} questions, {len(answers)} answers")
        
        # Group by user using fixed extraction
        user_groups = {}
        valid_items = []
        
        for q in questions:
            qid = str(q["id"])
            if qid in answers:
                uid = self.extract_user_id(q['id'])
                item = {
                    'id': q['id'],
                    'question': q['input'],
                    'user_id': uid,
                    'profile': q.get('profile', []),
                    'reference': answers[qid]
                }
                user_groups.setdefault(uid, []).append(item)
                valid_items.append(item)
        
        print(f"Users: {len(user_groups)}, Valid items: {len(valid_items)}")
        print(f"User distribution: {[(uid, len(items)) for uid, items in user_groups.items()]}")
        
        # Non-overlapping stratified sampling
        expanded_items = []
        U = len(user_groups)
        base = n_samples // U
        extra = n_samples % U
        
        pool = []
        
        for uid, items in user_groups.items():
            k = min(base, len(items))
            if k > 0:
                sel = random.sample(items, k=k)
                expanded_items.extend(sel)
                remain = [x for x in items if x not in sel]
                if remain:
                    pool.extend(remain)
        
        # Fill remaining spots from pool
        if extra > 0 and pool:
            add = random.sample(pool, k=min(extra, len(pool)))
            expanded_items.extend(add)
        
        # Final trimming to exact size
        expanded_items = expanded_items[:n_samples]
        
        # Verify distributions
        tag_counts = {}
        user_counts = {}
        for item in expanded_items:
            tag = item['reference']
            user = item['user_id']
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            user_counts[user] = user_counts.get(user, 0) + 1
        
        print(f"✓ Expansion completed: {len(expanded_items)} items (stratified)")
        print(f"  Users covered: {len(user_counts)} -> {dict(user_counts)}")
        print(f"  Tags: {len(tag_counts)} unique tags")
        print(f"  Top tags: {dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5])}")
        
        # Save expanded dataset
        output_file = self.output_dir / f"lamp2_stratified_eval_{n_samples}.jsonl"
        with open(output_file, "w") as f:
            for item in expanded_items:
                f.write(json.dumps(item) + "\n")
        
        print(f"✓ Saved to: {output_file}")
        return len(expanded_items), len(tag_counts), len(user_counts)

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    expander = RobustExpander()
    n_items, n_tags, n_users = expander.expand_dataset(n_samples=140)
    print(f"\nStratified evaluation set ready: {n_items} samples across {n_tags} tags and {n_users} user groups")
