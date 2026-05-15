"""Shared parsing and leakage-detection utilities."""

from __future__ import annotations

import re

FORBIDDEN_PATTERNS = [
    r"\d+\s*%",
    r"conversion rate",
    r"satisfaction rate",
    r"CTR",
    r"CVR",
    r"click-through",
    r"sample size",
    r"n\s*=",
    r"historical data shows",
    r"\b0\.\d+\b",
    r"\b1\.0\b",
    r"ranked #?1 by conversion",
    r"purchase rate",
    r"return rate of",
    r"\d+\s*out of\s*\d+",
    r"rating of \d",
    r"scored? \d+(\.\d+)?",
    r"\d+ stars?",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_PATTERNS]


def detect_leakage(text: str) -> list[dict]:
    """Return list of leakage matches found in text."""
    matches = []
    for pattern, compiled in zip(FORBIDDEN_PATTERNS, _COMPILED):
        for m in compiled.finditer(text):
            matches.append({
                "pattern": pattern,
                "match": m.group(),
                "start": m.start(),
                "end": m.end(),
            })
    return matches


def has_leakage(text: str) -> bool:
    return len(detect_leakage(text)) > 0


def clean_json_text(text: str) -> str:
    """Strip markdown fences from JSON response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
        else:
            text = "\n".join(lines[1:])
    return text.strip()
