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

# no longer using per-model concurrent fetches; single index fetch is used

# Config for artificialanalysis.ai (update endpoint if AA publishes a different path)
AA_API_KEY_ENV = "ARTIFICIALANALYSIS_API_KEY"
AA_API_BASE = "https://artificialanalysis.ai"
AA_MODELS_ENDPOINT = AA_API_BASE + "/api/v2/data/llms/models"

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


def fetch_all_aa_models(api_key: str) -> dict[str, dict]:
    """Fetch the AA LLM models index and return a case-insensitive lookup.

    Returns a dict mapping lowercased id/slug/name -> full AA model dict.
    """
    if not api_key:
        return {}
    url = AA_MODELS_ENDPOINT
    headers = {"x-api-key": api_key}
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            log.warning("AA models endpoint returned %s: %s", resp.status_code, resp.text[:200])
            return {}
        payload = resp.json()
        items = payload.get("data") or []
        index: dict[str, dict] = {}
        for m in items:
            # index by stable identifiers
            for key in (m.get("id"), m.get("slug"), m.get("name")):
                if key:
                    lk = str(key).lower()
                    if lk not in index:
                        index[lk] = m
        return index
    except requests.RequestException as e:
        log.warning("Failed to fetch AA models: %s", e)
        return {}


def extract_overall_from_evaluations(evals: dict) -> float | None:
    """Extract a sensible 'overall' score from an AA evaluations object.

    Prefers the `artificial_analysis_intelligence_index` field, then tries a
    small set of fallbacks and finally any numeric value found.
    """
    if not isinstance(evals, dict):
        return None
    # preferred keys (as observed in the docs)
    preferred = ("artificial_analysis_intelligence_index", "artificial_analysis_overall", "overall")
    for k in preferred:
        if k in evals:
            try:
                return float(evals[k])
            except Exception:
                pass
    # fallback: find any numeric-looking value
    for v in evals.values():
        try:
            fv = float(v)
            if not math.isnan(fv):
                return fv
        except Exception:
            continue
    return None


def find_aa_data_for_model(model: dict[str, Any], aa_map: dict[str, str], aa_index: dict[str, dict]) -> dict | None:
    """Find the AA model dict for an OpenRouter model using mapping or heuristics.

    Returns the AA model dict or None.
    """
    candidates: list[str] = []
    if model.get("id"):
        candidates.append(str(model.get("id")))
    name = model.get("name")
    if name:
        candidates.append(str(name))
    raw = model.get("raw", {}) or {}
    display_name = raw.get("display_name")
    if display_name:
        candidates.append(str(display_name))

    # Check mapping first (case-insensitive key lookup)
    for orig in candidates:
        key = orig.lower()
        if key in aa_map:
            mapped = aa_map[key]
            if not mapped:
                continue
            mk = str(mapped).lower()
            if mk in aa_index:
                return aa_index[mk]

    # Direct match against AA index keys (id/slug/name)
    for orig in candidates:
        lk = orig.lower()
        if lk in aa_index:
            return aa_index[lk]

    # Heuristic substring match against name/slug
    for orig in candidates:
        lo = orig.lower()
        for aa_model in aa_index.values():
            try:
                if lo == str(aa_model.get("name", "")).lower() or lo == str(aa_model.get("slug", "")).lower():
                    return aa_model
                if lo in str(aa_model.get("name", "")).lower() or lo in str(aa_model.get("slug", "")).lower():
                    return aa_model
            except Exception:
                continue
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

    # Fetch AA model index once and map entries to our models
    aa_index: dict[str, dict] = {}
    if aa_api_key:
        aa_index = fetch_all_aa_models(aa_api_key)
        if not aa_index:
            log.warning("Failed to fetch AA models index or index is empty")

    for m in models:
        # Prefer found AA model data when we have an index
        aa_data = find_aa_data_for_model(m, aa_map, aa_index) if aa_index else None
        if aa_data:
            aa_id = aa_data.get("id") or aa_data.get("slug")
            m["aa_id"] = aa_id
            m["aa_overall"] = extract_overall_from_evaluations(aa_data.get("evaluations") or {})
            log.info("Mapped %s -> AA %s score=%s", m.get("id"), aa_id, m.get("aa_overall"))
        else:
            # Fallback: preserve previous mapping/guess behaviour (no HTTP call)
            aa_id_guess = find_aa_id_for_model(m, aa_map)
            if aa_id_guess:
                m["aa_id"] = aa_id_guess

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
