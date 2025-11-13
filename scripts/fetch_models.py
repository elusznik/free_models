#!/usr/bin/env python3
"""
Fetch zero-cost models from OpenRouter via the OpenAI Python SDK,
enhance them with artificialanalysis.ai overall scores (when available),
sort by that score descending (unscored last) and write a static `docs/index.html`.

The script reads:
- OPENROUTER_API_KEY (or api_key.txt) for OpenRouter
- ARTIFICIALANALYSIS_API_KEY (optional) for scoring

Optionally place a JSON mapping at scripts/aa_model_map.json with shape:
{
  "openrouter_model_id_or_name": "aa_model_id",
  "another-openrouter-id": "aa-model-123"
}

If no mapping exists, the script will attempt to use openrouter model id as the AA id as a guess.
"""
import html
import json
import logging
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

try:
    from openai import OpenAI
except Exception:
    log.error("Missing dependency 'openai'. Install with: pip install -r requirements.txt")
    raise

try:
    import requests
except Exception:
    log.error("Missing dependency 'requests'. Install with: pip install -r requirements.txt")
    raise

from concurrent.futures import ThreadPoolExecutor, as_completed

# Config for artificialanalysis.ai (update endpoint if AA publishes a different path)
AA_API_KEY_ENV = "ARTIFICIALANALYSIS_API_KEY"
AA_API_BASE = "https://api.artificialanalysis.ai"  # adjust if their docs show a different base
AA_SCORE_ENDPOINT = AA_API_BASE + "/v1/models/{aa_id}/scores?metrics=overall"

AA_MAP_PATH = Path("scripts/aa_model_map.json")

REQUEST_TIMEOUT = 10
RETRY_ATTEMPTS = 4
RETRY_BACKOFF_BASE = 1.5
MAX_WORKERS = 4


def safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return getattr(obj, "__dict__", {})


def load_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    p = Path("api_key.txt")
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    raise RuntimeError("No API key found. Set OPENROUTER_API_KEY or create api_key.txt")


def load_aa_map() -> dict[str, str]:
    """Load optional mapping file from openrouter id/name -> artificialanalysis.ai id."""
    if AA_MAP_PATH.exists():
        try:
            data = json.loads(AA_MAP_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # normalize keys to lower-case for case-insensitive matching
                return {k.lower(): v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
        except Exception:
            log.warning("Warning: failed to parse %s — ignoring mapping", AA_MAP_PATH)
    return {}


def find_aa_id_for_model(model: dict[str, Any], aa_map: dict[str, str]) -> str | None:
    """Try to determine the AA model id for a given OpenRouter model entry.

    Strategy:
    1. Check mapping file by openrouter id and name (case-insensitive).
    2. If mapping not found, guess using the original-case id/name (do not lower-case the returned guess).
    3. Return None if not found.
    """
    candidates: list[str] = []
    if model.get("id"):
        candidates.append(str(model.get("id")))
    name = model.get("name")
    if name:
        candidates.append(str(name))
    # also check raw.display_name if present
    raw = model.get("raw", {}) or {}
    display_name = raw.get("display_name")
    if display_name:
        candidates.append(str(display_name))

    for orig in candidates:
        key = orig.lower()
        if key in aa_map:
            return aa_map[key]

    # fallback guess: return the original-case first candidate (may or may not be correct)
    if candidates:
        return candidates[0]
    return None


def fetch_aa_overall(aa_id: str, api_key: str) -> float | None:
    """Fetch overall score from artificialanalysis.ai for aa_id.

    Returns numeric score or None.
    """
    if not aa_id or not api_key:
        return None
    url = AA_SCORE_ENDPOINT.format(aa_id=aa_id)
    headers = {"Authorization": f"Bearer {api_key}"}
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                # tolerant parsing for a few likely shapes:
                # { "overall": 95.3 }
                # { "scores": { "overall": 95.3 } }
                # { "metrics": [ { "name": "overall", "value": 95.3 }, ... ] }
                if isinstance(data, dict):
                    if "overall" in data:
                        try:
                            return float(data["overall"])
                        except Exception:
                            pass
                    scores = data.get("scores")
                    if isinstance(scores, dict) and "overall" in scores:
                        try:
                            return float(scores["overall"])
                        except Exception:
                            pass
                    metrics = data.get("metrics")
                    if isinstance(metrics, list):
                        for m in metrics:
                            try:
                                if isinstance(m, dict) and str(m.get("name", "")).lower() == "overall":
                                    return float(m.get("value"))
                            except Exception:
                                continue
                # if response is a simple number
                try:
                    return float(data)
                except Exception:
                    pass
                return None
            elif resp.status_code in (429, 502, 503, 504):
                sleep = (RETRY_BACKOFF_BASE ** attempt)
                time.sleep(sleep)
                continue
            else:
                # non-retryable error - log and return None
                log.warning("AA API returned status %s for id %s: %s", resp.status_code, aa_id, resp.text[:200])
                return None
        except requests.RequestException as e:
            sleep = (RETRY_BACKOFF_BASE ** attempt)
            log.warning("request error for AA id %s attempt %s: %s; sleeping %.1fs", aa_id, attempt, e, sleep)
            time.sleep(sleep)
    log.error("failed to fetch AA score for %s after retries", aa_id)
    return None


def render_index_html(data: dict[str, Any]) -> str:
    """Render a simple static HTML page for the models listing, including AA scores."""
    generated_at = data.get("generated_at")
    models = data.get("models", [])

    def esc(s: Any) -> str:
        if s is None:
            return ""
        return html.escape(str(s))

    rows: list[str] = []
    rank = 0
    for m in models:
        name = esc(m.get("name") or m.get("id") or "unknown")
        idv = esc(m.get("id") or "")
        desc = esc(m.get("description") or "")
        pricing = m.get("pricing") or {}
        prompt = esc(pricing.get("prompt", "-"))
        completion = esc(pricing.get("completion", "-"))
        flags = m.get("_flags", {}) or {}
        badges: list[str] = []
        if flags.get("zero_cost"):
            badges.append('<span class="badge zero">zero-cost</span>')
        if flags.get("free_tag"):
            badges.append('<span class="badge free">:free</span>')
        badge_html = " ".join(badges)

        score = m.get("aa_overall")
        score_display = "-"
        score_val: float | None = None
        if score is not None:
            try:
                score_val = float(score)
                if math.isnan(score_val):
                    score_val = None
            except Exception:
                score_val = None

        if score_val is not None:
            score_display = f"{score_val:.2f}"
            rank += 1

        if m.get("aa_id"):
            aa_info = f'<div class="meta">AA id: {esc(m.get("aa_id"))} — score: {esc(score_display)}</div>'
        else:
            aa_info = f'<div class="meta">AA id: (not mapped) — score: {esc(score_display)}</div>'

        rows.append(f"""
        <li class="model">
          <h3>{name}</h3>
          <div class="meta">{idv} {badge_html}</div>
          {('<p>'+desc+'</p>') if desc else ''}
          <div class="meta">pricing: prompt={prompt} completion={completion}</div>
          {aa_info}
        </li>
        """)

    models_html = "\n".join(rows) or '<li class="muted">No models found.</li>'

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>OpenRouter Zero-Cost Models</title>
    <link rel="stylesheet" href="styles.css" />
    <style>
      .badge{{display:inline-block;padding:2px 6px;border-radius:6px;font-size:0.8rem;margin-left:8px}}
      .badge.zero{{background:#fde68a;color:#42321b}}
      .badge.free{{background:#bae6fd;color:#052028}}
      .models-list{{list-style:none;padding:0}}
      .model{{margin:1rem 0;padding:0.8rem;border-bottom:1px solid #eee}}
      .meta{{color:#666;font-size:0.9rem;margin-top:0.4rem}}
    </style>
  </head>
  <body>
    <main>
      <h1>OpenRouter — Zero-Cost Models (sorted by artificialanalysis.ai overall score)</h1>
      <p class="lead">A live listing of zero-cost models surfaced from OpenRouter, enhanced with artificialanalysis.ai scores when available.</p>
      <div class="muted">Generated: {esc(generated_at)}</div>
      <ul class="models-list">{models_html}</ul>
      <footer><small>Data generated by <code>scripts/fetch_models.py</code>. Scores provided by artificialanalysis.ai when ARTIFICIALANALYSIS_API_KEY is set.</small></footer>
    </main>
  </body>
</html>"""


def main() -> None:
    try:
        api_key = load_api_key()
    except Exception as exc:
        log.error("%s", exc)
        sys.exit(1)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    try:
        response = client.models.list()
    except Exception as exc:
        log.error("Could not list models: %s", exc)
        sys.exit(1)

    models: list[dict[str, Any]] = []

    for item in getattr(response, "data", []) or []:
        data = to_dict(item)
        pricing = data.get("pricing", {}) or {}

        # Parse pricing if possible; allow missing/invalid values.
        prompt_cost = safe_float(pricing.get("prompt"), None)
        completion_cost = safe_float(pricing.get("completion"), None)

        # A free model is either explicitly zero-cost (both prompt & completion are present and 0)
        # or it contains the literal tag ":free" in the id/name/display_name.
        is_zero_cost = (prompt_cost is not None and completion_cost is not None and prompt_cost == 0 and completion_cost == 0)

        text_fields = []
        for k in ("id", "name", "display_name"):
            v = data.get(k)
            if v:
                text_fields.append(str(v).lower())
        has_free_tag = any(":free" in t for t in text_fields)

        if not (is_zero_cost or has_free_tag):
            continue

        models.append(
            {
                "id": data.get("id"),
                "name": data.get("name") or data.get("display_name"),
                "description": data.get("description"),
                "pricing": pricing,
                "raw": data,
                "_flags": {
                    "zero_cost": bool(is_zero_cost),
                    "free_tag": bool(has_free_tag),
                },
                # placeholders filled in after matching to AA
                "aa_id": None,
                "aa_overall": None,
            }
        )

    # Load AA mapping and API key
    aa_map = load_aa_map()
    aa_api_key = os.getenv(AA_API_KEY_ENV)
    if not aa_api_key:
        log.info("ARTIFICIALANALYSIS_API_KEY not set: generating index without AA scores")

    # Map models to AA ids (guess when possible)
    for m in models:
        aa_id = find_aa_id_for_model(m, aa_map)
        m["aa_id"] = aa_id

    # Fetch AA scores concurrently when we have an API key
    if aa_api_key:
        to_fetch = {m["aa_id"] for m in models if m.get("aa_id")}
        # Use mapping from aa_id -> future
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(fetch_aa_overall, aa_id, aa_api_key): aa_id for aa_id in to_fetch}
            aa_scores: dict[str, float | None] = {}
            for fut in as_completed(futures):
                aaid = futures[fut]
                try:
                    score = fut.result()
                    aa_scores[aaid] = score
                    log.info("Fetched AA score for %s: %s", aaid, score)
                except Exception as exc:
                    aa_scores[aaid] = None
                    log.warning("Error fetching AA score for %s: %s", aaid, exc)

        # assign scores back to models
        for m in models:
            aaid = m.get("aa_id")
            if aaid:
                m["aa_overall"] = aa_scores.get(aaid)

    def is_valid_score(m: dict[str, Any]) -> bool:
        v = m.get("aa_overall")
        if v is None:
            return False
        try:
            fv = float(v)
        except Exception:
            return False
        if isinstance(fv, float) and math.isnan(fv):
            return False
        return True

    # Sort models: numeric scores descending first, then those without scores alphabetically
    numeric = [m for m in models if is_valid_score(m)]
    non_numeric = [m for m in models if not is_valid_score(m)]
    try:
        numeric_sorted = sorted(numeric, key=lambda x: (-float(x["aa_overall"]), (x.get("name") or "").lower()))
    except Exception:
        # fallback to keeping original order
        numeric_sorted = numeric
    non_numeric_sorted = sorted(non_numeric, key=lambda x: (x.get("name") or "").lower())
    final_sorted = numeric_sorted + non_numeric_sorted

    out = {
        # Use a timezone-aware UTC timestamp. isoformat() produces +00:00; replace it with 'Z'.
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "count": len(final_sorted),
        "models": final_sorted,
    }

    index_path = Path("docs/index.html")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(render_index_html(out), encoding="utf-8")
    log.info("Wrote %s with %s models", index_path, len(final_sorted))


if __name__ == "__main__":
    main()
