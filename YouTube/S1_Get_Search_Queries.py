import os
import re
import json
import datetime as dt
from typing import Dict, List, Optional

import requests


MLVOCA_URL = "https://mlvoca.com/api/generate"
DEFAULT_MODEL = "deepseek-r1:1.5b"  # more capable than tinyllama for structured output
CATEGORY_ORDER: List[str] = ["generic", "specific", "niche"]


def read_all_keywords(topics_path: str) -> List[str]:
    """Read all non-empty lines from Underlying_Topics.txt as keywords."""
    keywords: List[str] = []
    with open(topics_path, "r", encoding="utf-8") as f:
        for line in f:
            kw = line.strip()
            if kw:
                keywords.append(kw)
    if not keywords:
        raise ValueError("No keywords found in topics file.")
    return keywords


def build_prompt_category(keyword: str, category: str) -> str:
    """Create a strict JSON-only instruction for generating one category with 10 queries."""
    return (
        "You are generating realistic YouTube search queries.\n"
        f"Keyword: {keyword}\n"
        f"Category: {category}\n"
        "Requirements:\n"
        "- Produce EXACTLY 10 queries users might type on YouTube search.\n"
        "- Keep queries natural (search-style phrases), short, and varied.\n"
        "- Prefer English queries.\n"
        "Output format: Return ONLY valid JSON with double quotes, no extra text.\n"
        "{\n"
        "  \"keyword\": \"<keyword>\",\n"
        "  \"category\": \"<category>\",\n"
        "  \"queries\": [\"...10 items...\"]\n"
        "}"
    )


def call_mlvoca(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Call the mlvoca API and return the raw string response."""
    body = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(MLVOCA_URL, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")


def _extract_json_block(text: str) -> Optional[str]:
    """Attempt to extract the first JSON object from a text blob."""
    # Remove code fences if present
    text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text.strip())
    # Find outermost JSON object by braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        return candidate
    return None


def parse_category_response(raw: str, keyword: str, category: str) -> List[str]:
    """Parse the model output for a single category and return a list of 10 queries or raise."""
    block = _extract_json_block(raw) or raw
    data = json.loads(block)
    if data.get("keyword") and str(data.get("keyword")).strip().lower() != keyword.lower():
        # tolerate; we don't hard fail on mismatched echo
        pass
    if str(data.get("category", "")).strip().lower() != category.lower():
        # tolerate as long as queries are present
        pass
    queries = data.get("queries")
    if not isinstance(queries, list):
        raise ValueError("Missing queries array")
    queries = [str(q).strip() for q in queries if isinstance(q, (str, int, float))]
    queries = [q for q in queries if q]
    queries = _dedupe_preserve_order(queries)
    if len(queries) != 10:
        raise ValueError("Expected exactly 10 unique queries")
    return queries


# Removed heuristic fallback per request; only accept valid categories and skip on failure.


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        if not isinstance(it, str):
            continue
        s = it.strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _find_item(agg: Dict, keyword: str) -> Optional[Dict]:
    for it in agg.get("items", []):
        if it.get("keyword") == keyword:
            return it
    return None


def ensure_item(agg: Dict, keyword: str) -> Dict:
    item = _find_item(agg, keyword)
    if item is None:
        item = {
            "keyword": keyword,
            "categories": {},
        }
        agg.setdefault("items", []).append(item)
    if "categories" not in item or not isinstance(item["categories"], dict):
        item["categories"] = {}
    return item


def category_done(item: Dict, category: str) -> bool:
    cats = item.get("categories", {})
    lst = cats.get(category)
    return isinstance(lst, list) and len(lst) == 10


def recompute_completed_keywords(agg: Dict) -> int:
    cnt = 0
    for it in agg.get("items", []):
        cats = it.get("categories", {})
        if all(isinstance(cats.get(cat), list) and len(cats.get(cat, [])) == 10 for cat in CATEGORY_ORDER):
            cnt += 1
    return cnt


def slugify(text: str) -> str:
    s = text.lower()
    s = re.sub(r"[&]", " and ", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def save_json_atomic(output_path: str, data: Dict) -> None:
    """Safely write JSON to disk using an atomic replace to avoid corruption."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)


def load_aggregate(output_path: str) -> Dict:
    """Load aggregate JSON if present, else initialize a new aggregate structure."""
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                # If file is corrupted, back it up and start fresh
                backup = output_path + ".backup"
                try:
                    os.replace(output_path, backup)
                except Exception:
                    pass
    # Fresh aggregate structure
    return {
        "meta": {
            "model": DEFAULT_MODEL,
            "total_keywords": 0,
            "completed_keywords": 0,
            "last_updated": dt.datetime.now(dt.UTC).isoformat(),
        },
        "items": [],
    }


def aggregate_has_keyword(agg: Dict, keyword: str) -> bool:
    return any(item.get("keyword") == keyword for item in agg.get("items", []))


def aggregate_append(agg: Dict, item: Dict) -> None:
    agg.setdefault("items", []).append(item)
    # Update meta
    all_keywords = {it.get("keyword") for it in agg["items"]}
    agg.setdefault("meta", {})
    agg["meta"]["completed_keywords"] = len(all_keywords)
    # total_keywords will be set in main based on topics file
    agg["meta"]["last_updated"] = dt.datetime.now(dt.UTC).isoformat()


def main():
    workspace_root = os.path.dirname(os.path.dirname(__file__))
    topics_path = os.path.join(workspace_root, "YouTube", "Datasets", "Underlying_Topics.txt")
    agg_path = os.path.join(workspace_root, "YouTube", "Datasets", "Search_Queries_All.json")

    keywords = read_all_keywords(topics_path)
    aggregate = load_aggregate(agg_path)
    aggregate.setdefault("meta", {})
    aggregate["meta"]["model"] = DEFAULT_MODEL
    aggregate["meta"]["total_keywords"] = len(keywords)

    for keyword in keywords:
        item = ensure_item(aggregate, keyword)
        for category in CATEGORY_ORDER:
            if category_done(item, category):
                print(f"Skip {keyword} - {category} already done")
                continue
            prompt = build_prompt_category(keyword, category)
            try:
                raw = call_mlvoca(prompt, model=DEFAULT_MODEL)
                queries = parse_category_response(raw, keyword, category)
            except Exception as e:
                print(f"Skip {keyword} - {category} failed: {e}")
                continue

            # Save the successful category immediately for resumability
            item["categories"][category] = queries
            aggregate["meta"]["last_updated"] = dt.datetime.now(dt.UTC).isoformat()
            aggregate["meta"]["completed_keywords"] = recompute_completed_keywords(aggregate)
            save_json_atomic(agg_path, aggregate)
            print(
                f"Saved {keyword} - {category} (completed keywords: {aggregate['meta']['completed_keywords']}/{aggregate['meta']['total_keywords']})"
            )

    print(f"All done. Aggregate at: {agg_path}")


if __name__ == "__main__":
    main()
