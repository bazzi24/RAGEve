from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

from rag.deepdoc.analyzer import classify_char


# ----------------------------------------------------------------------
# Chunk profile definitions
# ----------------------------------------------------------------------


class ChunkProfile(str, Enum):
    CLEAN_TEXT = "clean_text"
    OCR_NOISY = "ocr_noisy"
    TABLE_HEAVY = "table_heavy"
    CODE_MIXED = "code_mixed"
    GENERAL = "general"


@dataclass
class ChunkProfileConfig:
    profile: ChunkProfile
    chunk_size: int
    chunk_overlap: int
    max_tokens_per_chunk: int
    reason: str


PROFILE_PRESETS: dict[ChunkProfile, ChunkProfileConfig] = {
    ChunkProfile.CLEAN_TEXT: ChunkProfileConfig(
        profile=ChunkProfile.CLEAN_TEXT,
        chunk_size=1500,
        chunk_overlap=200,
        max_tokens_per_chunk=600,
        reason="High alpha ratio, consistent script, natural punctuation.",
    ),
    ChunkProfile.OCR_NOISY: ChunkProfileConfig(
        profile=ChunkProfile.OCR_NOISY,
        chunk_size=600,
        chunk_overlap=150,
        max_tokens_per_chunk=250,
        reason="High noise signals detected — using smaller chunks with higher overlap.",
    ),
    ChunkProfile.TABLE_HEAVY: ChunkProfileConfig(
        profile=ChunkProfile.TABLE_HEAVY,
        chunk_size=800,
        chunk_overlap=100,
        max_tokens_per_chunk=350,
        reason="Dense tabular structure detected — preserving row/section boundaries.",
    ),
    ChunkProfile.CODE_MIXED: ChunkProfileConfig(
        profile=ChunkProfile.CODE_MIXED,
        chunk_size=700,
        chunk_overlap=120,
        max_tokens_per_chunk=300,
        reason="Code or mixed content detected — delimiter-aware chunking.",
    ),
    ChunkProfile.GENERAL: ChunkProfileConfig(
        profile=ChunkProfile.GENERAL,
        chunk_size=1200,
        chunk_overlap=180,
        max_tokens_per_chunk=500,
        reason="Mixed content or standard document — balanced settings.",
    ),
}


# ----------------------------------------------------------------------
# Quality signals
# ----------------------------------------------------------------------


@dataclass
class QualitySignals:
    alpha_ratio: float
    ocr_noise_ratio: float  # garbled / symbol-heavy ratio
    broken_line_ratio: float  # lines split mid-sentence
    header_footer_ratio: float  # repeated header/footer
    table_density: float  # 0-1: fraction of text in table-like lines
    avg_sentence_length: float
    language_script_changes: int  # number of script switches (latin <-> cjk etc)
    repeated_word_ratio: float  # high repeat = likely OCR garbage
    code_delimiter_ratio: float  # braces, brackets, indentation
    issue_tags: list[str]


# ----------------------------------------------------------------------
# Signal extraction
# ----------------------------------------------------------------------


# Regex patterns for signal detection
BROKEN_LINE_RE = re.compile(r"\w{3,}$")  # ends mid-word
OCR_NOISE_RE = re.compile(r"[_]{5,}|\.{4,}|[~]{3,}|[\^]{3,}")
# Only match alphanumeric char repetition (OCR garbage), not punctuation
REPEATED_CHAR_RE = re.compile(r"([a-zA-Z0-9])\1{4,}")
# Matches markdown pipe tables and CSV-like lines with 2+ columns
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|.*\|")
CODE_DELIM_RE = re.compile(r"[\{\}\[\]\(\)]|    +|  {2}|^\s{0,4}(if|else|for|while|def|class|import|return)\b")
HEADER_FOOTER_RE = re.compile(
    r"^(page\s+\d+|chapter\s+\d|section\s+\d|©|\||\*{3,}|confidential|draft)",
    re.IGNORECASE,
)


def _script_family(ch: str) -> str:
    name = unicodedata.name(ch, "")
    if "CJK" in name or "IDEOGRAPH" in name or "HIRAGANA" in name or "KATAKANA" in name:
        return "cjk"
    if "LATIN" in name:
        return "latin"
    if "ARABIC" in name:
        return "arabic"
    if "CYRILLIC" in name:
        return "cyrillic"
    if "HANGUL" in name:
        return "hangul"
    return "other"


def _count_script_changes(text: str) -> int:
    prev_script = ""
    changes = 0
    for ch in text:
        if ch.isspace():
            continue
        current_script = _script_family(ch)
        if prev_script and current_script != prev_script:
            changes += 1
        prev_script = current_script
    return changes


def compute_quality_signals(text: str) -> QualitySignals:
    if not text:
        return QualitySignals(
            alpha_ratio=0.0,
            ocr_noise_ratio=0.0,
            broken_line_ratio=0.0,
            header_footer_ratio=0.0,
            table_density=0.0,
            avg_sentence_length=0.0,
            language_script_changes=0,
            repeated_word_ratio=0.0,
            code_delimiter_ratio=0.0,
            issue_tags=["empty"],
        )

    lines = text.split("\n")
    total_chars = len(text)
    total_lines = len(lines)

    # 1. Alpha ratio (from analyzer)
    alpha_chars = sum(1 for ch in text if classify_char(ch) != "whitespace" and classify_char(ch) != "other")
    alpha_ratio = alpha_chars / max(total_chars, 1)

    # 2. OCR noise ratio
    noise_matches = sum(1 for _ in OCR_NOISE_RE.finditer(text))
    ocr_noise_ratio = noise_matches / max(total_chars, 1)

    # 3. Broken line ratio (lines ending mid-word)
    broken_lines = sum(1 for line in lines if BROKEN_LINE_RE.search(line.rstrip()))
    broken_line_ratio = broken_lines / max(total_lines, 1)

    # 4. Header/footer ratio
    hf_lines = sum(1 for line in lines if HEADER_FOOTER_RE.match(line.strip()))
    header_footer_ratio = hf_lines / max(total_lines, 1)

    # 5. Table density
    table_lines = sum(1 for line in lines if TABLE_LINE_RE.search(line))
    table_density = table_lines / max(total_lines, 1)

    # 6. Average sentence length
    sentences = re.split(r"[.!?。！？]+", text)
    sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
    avg_sentence_length = sum(sentence_lengths) / max(len(sentence_lengths), 1)

    # 7. Script changes (latin <-> cjk etc)
    language_script_changes = _count_script_changes(text)

    # 8. Repeated char ratio (OCR garbage indicator)
    repeated_chars = sum(1 for _ in REPEATED_CHAR_RE.finditer(text))
    repeated_word_ratio = repeated_chars / max(total_chars, 1)

    # 9. Code delimiter ratio (curly braces, square brackets, 4-space indents)
    import re as _re

    code_delim_chars = (
        sum(1 for ch in text if ch in "{}[")
        + len(_re.findall(r"    +", text))  # 4+ space indentation
    )
    code_delimiter_ratio = code_delim_chars / max(total_chars, 1)

    # 10. Issue tags
    issue_tags: list[str] = []
    if ocr_noise_ratio > 0.01:
        issue_tags.append("ocr_noise")
    if broken_line_ratio > 0.25:
        issue_tags.append("broken_lines")
    if table_density > 0.05:
        issue_tags.append("table_heavy")
    if header_footer_ratio > 0.1:
        issue_tags.append("header_footer_noise")
    if repeated_word_ratio > 0.005:
        issue_tags.append("repeated_chars")
    if code_delimiter_ratio > 0.03:
        issue_tags.append("code_delimiters")
    if language_script_changes > 10:
        issue_tags.append("mixed_scripts")

    return QualitySignals(
        alpha_ratio=round(alpha_ratio, 4),
        ocr_noise_ratio=round(ocr_noise_ratio, 4),
        broken_line_ratio=round(broken_line_ratio, 4),
        header_footer_ratio=round(header_footer_ratio, 4),
        table_density=round(table_density, 4),
        avg_sentence_length=round(avg_sentence_length, 2),
        language_script_changes=language_script_changes,
        repeated_word_ratio=round(repeated_word_ratio, 4),
        code_delimiter_ratio=round(code_delimiter_ratio, 4),
        issue_tags=issue_tags,
    )


# ----------------------------------------------------------------------
# Quality score calculation
# ----------------------------------------------------------------------


def compute_quality_score(signals: QualitySignals) -> float:
    """
    Compute a 0-1 quality score from quality signals.
    Higher = cleaner, more reliable for retrieval.
    """
    score = 1.0

    # Penalise OCR noise heavily
    score -= signals.ocr_noise_ratio * 0.4

    # Penalise broken lines
    score -= signals.broken_line_ratio * 0.15

    # Penalise header/footer noise
    score -= signals.header_footer_ratio * 0.1

    # Penalise very short avg sentences (fragmented/OCR)
    if signals.avg_sentence_length < 5:
        score -= 0.1

    # Penalise high repeated char ratio
    score -= signals.repeated_word_ratio * 0.3

    return max(0.0, min(1.0, round(score, 4)))


# ----------------------------------------------------------------------
# Adaptive profile selector
# ----------------------------------------------------------------------


def select_chunk_profile(signals: QualitySignals, score: float) -> ChunkProfileConfig:
    """
    Select the most appropriate chunk profile based on quality signals.
    More specific profiles are checked before more general ones.
    """
    # Table heavy — specific structural profile
    if signals.table_density > 0.05:
        return PROFILE_PRESETS[ChunkProfile.TABLE_HEAVY]

    # Code / mixed — specific structural profile
    if signals.code_delimiter_ratio > 0.03:
        return PROFILE_PRESETS[ChunkProfile.CODE_MIXED]

    # OCR noisy — heavy noise only (separate repeated_char check below)
    if signals.ocr_noise_ratio > 0.005:
        return PROFILE_PRESETS[ChunkProfile.OCR_NOISY]

    # Repeated chars — standalone OCR garbage indicator
    if signals.repeated_word_ratio > 0.002:
        return PROFILE_PRESETS[ChunkProfile.OCR_NOISY]

    # Clean text
    if score >= 0.85 and not signals.issue_tags:
        return PROFILE_PRESETS[ChunkProfile.CLEAN_TEXT]

    # Default to general
    return PROFILE_PRESETS[ChunkProfile.GENERAL]


# ----------------------------------------------------------------------
# High-level scorer
# ----------------------------------------------------------------------


@dataclass
class QualityReport:
    quality_score: float
    profile: ChunkProfileConfig
    signals: QualitySignals


def score_and_select_profile(text: str) -> QualityReport:
    signals = compute_quality_signals(text)
    score = compute_quality_score(signals)
    profile = select_chunk_profile(signals, score)
    return QualityReport(quality_score=score, profile=profile, signals=signals)
