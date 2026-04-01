from __future__ import annotations

from collections import Counter
import unicodedata


def classify_char(char: str) -> str:
    if char.isspace():
        return "whitespace"
    if char.isdigit():
        return "digit"
    if char.isalpha():
        name = unicodedata.name(char, "")
        if "CJK" in name or "IDEOGRAPH" in name:
            return "cjk"
        if "HIRAGANA" in name:
            return "hiragana"
        if "KATAKANA" in name:
            return "katakana"
        if "HANGUL" in name:
            return "hangul"
        if "ARABIC" in name:
            return "arabic"
        if "CYRILLIC" in name:
            return "cyrillic"
        if "LATIN" in name:
            return "latin_upper" if char.isupper() else "latin_lower"
        return "alphabetic"

    category = unicodedata.category(char)
    if category.startswith("P"):
        return "punctuation"
    if category.startswith("S"):
        return "symbol"
    if category.startswith("M"):
        return "mark"
    return "other"


def analyze_text(text: str) -> dict:
    counter: Counter[str] = Counter()
    for ch in text:
        counter[classify_char(ch)] += 1

    total = len(text)
    alpha_count = sum(counter[k] for k in ["latin_upper", "latin_lower", "alphabetic", "cjk", "hiragana", "katakana", "hangul", "arabic", "cyrillic"])

    return {
        "characters": total,
        "alpha_ratio": round(alpha_count / total, 4) if total else 0.0,
        "types": dict(counter),
    }


def analyze_chunks(chunks: list[str]) -> list[dict]:
    return [analyze_text(chunk) for chunk in chunks]
