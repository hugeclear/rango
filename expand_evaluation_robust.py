import json
import random
from pathlib import Path
from collections import Counter


def _extract_user_id(qid: str) -> str:
    """Extract user id by dropping the last character.

    Example: "111" -> "11"
    """
    return str(qid)[:-1] if qid else qid


class RobustExpander:
    def __init__(self):
        # Adapted to this repo: use LaMP-2 raw files
        self.data_dir = Path("data/raw/LaMP-2")
        self.output_dir = Path("data/lamp_expanded")

    def _load_raw(self):
        questions_path = self.data_dir / "questions.json"
        answers_path = self.data_dir / "answers.json"
        if not questions_path.exists() or not answers_path.exists():
            raise FileNotFoundError("Expected data/raw/LaMP-2/questions.json and answers.json")
        qs = json.load(open(questions_path, "r"))
        ans = json.load(open(answers_path, "r"))
        # answers.json is a dict with key 'golds' (list of {id, output})
        if not isinstance(ans, dict) or "golds" not in ans:
            raise ValueError("answers.json format unexpected: missing 'golds'")
        golds = {str(x["id"]): x["output"] for x in ans["golds"]}
        # Filter questions to those with golds
        qs = [q for q in qs if str(q.get("id")) in golds]
        return qs, golds

    def expand_dataset(self, n_samples: int = 100):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        qs, golds = self._load_raw()
        assert isinstance(qs, list)
        # Group by user bucket using id[:-1]
        user_groups = {}
        for q in qs:
            uid = _extract_user_id(q.get("id"))
            user_groups.setdefault(uid, []).append(q)

        print(f"元データ: {len(qs)}件, ユーザー数: {len(user_groups)}（期待≈70）")
        # Stratified, non-overlapping selection to reach n_samples
        U = len(user_groups)
        if U == 0:
            raise RuntimeError("No users found after grouping")
        base, extra = divmod(n_samples, U)
        expanded_q = []
        rng = random.Random(42)
        pool = []

        for uid, items in user_groups.items():
            # Ensure unique items per user
            k = min(base, len(items))
            if k > 0:
                sel = rng.sample(items, k=k)
                expanded_q.extend(sel)
                remain = [it for it in items if it not in sel]
                pool.extend(remain)

        if extra > 0 and pool:
            add = rng.sample(pool, k=min(extra, len(pool)))
            expanded_q.extend(add)

        # Trim/pad just in case
        expanded_q = expanded_q[:n_samples]

        # Build outputs aligned with questions
        expanded_y = [{"id": str(q["id"]), "output": golds[str(q["id"])]} for q in expanded_q]

        # Save
        json.dump(expanded_q, open(self.output_dir / "dev_questions_expanded.json", "w"), indent=2)
        json.dump(expanded_y, open(self.output_dir / "dev_outputs_expanded.json", "w"), indent=2)

        # Quick distribution check
        cnt = Counter(_extract_user_id(q.get("id")) for q in expanded_q)
        print("拡張後:", len(expanded_q), "件 / ユーザー数:", len(cnt))
        print("上位5ユーザー:", cnt.most_common(5))
        return len(expanded_q)


if __name__ == "__main__":
    RobustExpander().expand_dataset(n_samples=100)

