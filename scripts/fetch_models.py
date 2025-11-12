#!/usr/bin/env python3
"""
Fetch zero-cost models from OpenRouter via the OpenAI Python SDK
and write a static `docs/index.html` for the GitHub Pages site.

The script reads the API key from the environment variable
`OPENROUTER_API_KEY`, falling back to a file named `api_key.txt`.
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    from openai import OpenAI
except Exception:
    print("Missing dependency 'openai'. Install with: pip install -r requirements.txt", file=sys.stderr)
    raise


def to_dict(obj: Any) -> Dict[str, Any]:
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


def main() -> None:
    try:
        api_key = load_api_key()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    try:
        response = client.models.list()
    except Exception as exc:
        print(f"Could not list models: {exc}", file=sys.stderr)
        sys.exit(1)

    models: List[Dict[str, Any]] = []

    for item in getattr(response, "data", []) or []:
        data = to_dict(item)
        pricing = data.get("pricing", {}) or {}

        # Parse pricing if possible; allow missing/invalid values.
        prompt_cost = None
        completion_cost = None
        try:
            prompt_cost = float(pricing.get("prompt", 0))
        except (TypeError, ValueError):
            prompt_cost = None
        try:
            completion_cost = float(pricing.get("completion", 0))
        except (TypeError, ValueError):
            completion_cost = None

        # A free model is either explicitly zero-cost (both prompt & completion are 0)
        # or it contains the literal tag ":free" in the id/name/display_name.
        is_zero_cost = (prompt_cost == 0 and completion_cost == 0)

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
            }
        )

    out = {
        # Use a timezone-aware UTC timestamp. isoformat() produces +00:00; replace
        # it with 'Z' to keep the compact Zulu form (e.g. 2025-11-12T00:00:00Z).
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "count": len(models),
        "models": models,
    }

    # Generate a fully static `docs/index.html` for GitHub Pages (no client JS required).
    index_path = Path("docs/index.html")
    index_path.write_text(render_index_html(out), encoding="utf-8")
    print(f"Wrote {index_path} with {len(models)} models")


def render_index_html(data: Dict[str, Any]) -> str:
    """Render a simple static HTML page for the models listing.

    The function keeps styling by referencing `styles.css` which is already in `docs/`.
    """
    generated_at = data.get("generated_at")
    models = data.get("models", [])

    def esc(s):
        if s is None:
            return ""
        return (str(s)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

    rows = []
    for m in models:
        name = esc(m.get("name") or m.get("id") or "unknown")
        idv = esc(m.get("id") or "")
        desc = esc(m.get("description") or "")
        pricing = m.get("pricing") or {}
        prompt = esc(pricing.get("prompt", "-"))
        completion = esc(pricing.get("completion", "-"))
        flags = m.get("_flags", {}) or {}
        badges = []
        if flags.get("zero_cost"):
            badges.append('<span class="badge zero">zero-cost</span>')
        if flags.get("free_tag"):
            badges.append('<span class="badge free">:free</span>')

        badge_html = " ".join(badges)

        rows.append(f"""
        <li class="model">
          <h3>{name}</h3>
          <div class="meta">{idv} {badge_html}</div>
          {('<p>'+desc+'</p>') if desc else ''}
          <div class="meta">pricing: prompt={prompt} completion={completion}</div>
        </li>
        """)

    models_html = "\n".join(rows) or "<li class=\"muted\">No models found.</li>"

    return f"""<!doctype html>
<html lang=\"en\"> 
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
    <title>OpenRouter Zero-Cost Models</title>
    <link rel=\"stylesheet\" href=\"styles.css\" />
    <style>
      .badge{{display:inline-block;padding:2px 6px;border-radius:6px;font-size:0.8rem;margin-left:8px}}
      .badge.zero{{background:#fde68a;color:#42321b}}
      .badge.free{{background:#bae6fd;color:#052028}}
    </style>
  </head>
  <body>
    <main>
      <h1>OpenRouter — Zero-Cost Models</h1>
      <p class=\"lead\">A live listing of zero-cost models surfaced from OpenRouter.</p>
      <div class=\"muted\">Generated: {esc(generated_at)}</div>
      <ul class=\"models-list\">{models_html}</ul>
      <footer><small>Data generated by <code>scripts/fetch_models.py</code>.</small></footer>
    </main>
  </body>
</html>"""


if __name__ == "__main__":
    main()
