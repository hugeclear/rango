import argparse
import json
import os
from typing import List, Dict

import numpy as np
from sklearn.decomposition import PCA

try:
    import openai
except ImportError:  # pragma: no cover - openai is optional for testing
    openai = None


def select_representative_items(profile: List[str], k: int = 5) -> List[str]:
    """Return top-k representative items from the profile using PCA on OpenAI embeddings."""
    if openai is None:
        raise RuntimeError("openai package is not installed")
    if not profile:
        return []
    # Request embeddings for all profile items in a single API call
    try:  # pragma: no cover - network call
        if hasattr(openai, "Embeddings"):
            resp = openai.Embeddings.create(model="text-embedding-ada-002", input=profile)
        else:
            resp = openai.Embedding.create(model="text-embedding-ada-002", input=profile)
    except Exception as exc:  # pragma: no cover - network call
        print(f"Error obtaining embeddings: {exc}")
        return profile[:k]

    embeddings = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
    if len(profile) <= k:
        return profile
    pca = PCA(n_components=1)
    components = pca.fit_transform(embeddings)
    scores = np.abs(components[:, 0])
    top_idx = np.argsort(scores)[::-1][:k]
    return [profile[i] for i in top_idx]


def request_llm(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 50) -> str:
    """Send the prompt to the OpenAI API and return the response string."""
    if openai is None:
        raise RuntimeError("openai package is not installed")
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant and must reply in English."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message["content"].strip()
    except Exception as exc:  # pragma: no cover - network call
        print(f"Error during LLM call: {exc}")
        return ""


def main(args: argparse.Namespace):
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Dataset not found at {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        records: List[Dict] = json.load(f)

    # Configure OpenAI API key if available in environment
    if openai is not None and not openai.api_key:
        openai.api_key = os.getenv("OPENAI_API_KEY")

    personal_insights = []
    neutral_insights = []
    personal_pairs = []
    neutral_pairs = []

    for rec in records:
        rec_id = rec.get("id")
        input_text = rec.get("input", "")
        profile_items = rec.get("profile", [])

        selected = select_representative_items(profile_items, k=5)
        items_text = ", ".join(selected)

        personal_prompt = (
            "Here is the user's past movie-tagging history:\n"
            f"{items_text}\n"
            "Please provide in English one sentence describing this user's personal tagging tendencies."
        )
        neutral_prompt = (
            "Here is an example of a general user's tagging history:\n"
            f"{items_text}\n"
            "Please provide in English one sentence describing a neutral (no-bias) tagging tendency."
        )

        cP = request_llm(personal_prompt)
        cN = request_llm(neutral_prompt)

        personal_insights.append({"id": rec_id, "cP": cP})
        neutral_insights.append({"id": rec_id, "cN": cN})

        pair_p_prompt = (
            f"The user's tagging tendency is: \"{cP}\".\n"
            "Based on that, please answer in English, with a single word, the most suitable tag for the following movie description:\n"
            f"{input_text}"
        )
        pair_n_prompt = (
            f"The tagging tendency is: \"{cN}\".\n"
            "Based on that, please answer in English, with a single word, the most suitable general tag for the same movie description:\n"
            f"{input_text}"
        )

        pP = request_llm(pair_p_prompt)
        pN = request_llm(pair_n_prompt)

        personal_pairs.append({"id": rec_id, "pP": pP})
        neutral_pairs.append({"id": rec_id, "pN": pN})

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "personal_insights.json"), "w", encoding="utf-8") as f:
        json.dump(personal_insights, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "neutral_insights.json"), "w", encoding="utf-8") as f:
        json.dump(neutral_insights, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "personal_pairs.json"), "w", encoding="utf-8") as f:
        json.dump(personal_pairs, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "neutral_pairs.json"), "w", encoding="utf-8") as f:
        json.dump(neutral_pairs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate insights and tags via LLM")
    parser.add_argument("--input", required=True, help="Path to merged.json")
    parser.add_argument("--outdir", required=True, help="Directory to store outputs")
    args = parser.parse_args()
    main(args)
