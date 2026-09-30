from __future__ import annotations
from typing import Dict, Any, List


def build_alias_map(artifacts: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Build a mapping: alias_text_lower -> canonical_title_en
    from artifacts_kb.json 'aliases' field.
    """
    alias_map: Dict[str, str] = {}

    for a in artifacts:
        canonical = a.get("title_en", a.get("title", ""))
        aliases = a.get("aliases", {})
        if not canonical or not isinstance(aliases, dict):
            continue

        for _lang, names in aliases.items():
            if isinstance(names, list):
                for n in names:
                    if isinstance(n, str) and n.strip():
                        alias_map[n.strip().lower()] = canonical

    return alias_map


def normalize_query(q: str, alias_map: Dict[str, str]) -> str:
    """
    Replace any alias occurrence in the question with canonical artifact name.
    Example: 'ما هو حجر رشيد؟' -> 'ما هو Rosetta Stone؟'
    """
    q_norm = q
    q_lower = q.lower()

    for alias_lower, canonical in alias_map.items():
        if alias_lower in q_lower:
            # Simple replace on lowercase match:
            # We do a best-effort replacement (good enough for prototype).
            q_norm = q_norm.replace(alias_lower, canonical)
            q_norm = q_norm.replace(alias_lower.title(), canonical)

            # For Arabic, the .title() doesn't help, but the lowercase replace above covers it.
            # If the exact alias appears with different casing (Latin scripts), it's still covered.

    return q_norm