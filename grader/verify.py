from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from .models import Policy, VerificationResult
from .reveal_store import RevealStore


WORD_RE = re.compile(r"[A-Za-z0-9]+")
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
RANGE_RE = re.compile(r"(?i)\b(\d[\d,]*(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d[\d,]*(?:\.\d+)?)\b")
HOMOGLYPH_TABLE = str.maketrans(
    {
        "Α": "A",
        "А": "A",
        "Β": "B",
        "В": "B",
        "Ε": "E",
        "Е": "E",
        "Η": "H",
        "Н": "H",
        "Ι": "I",
        "І": "I",
        "Κ": "K",
        "К": "K",
        "Μ": "M",
        "М": "M",
        "Ν": "N",
        "О": "O",
        "Ο": "O",
        "Ρ": "P",
        "Р": "P",
        "С": "C",
        "Τ": "T",
        "Т": "T",
        "Χ": "X",
        "Х": "X",
        "Υ": "Y",
        "а": "a",
        "е": "e",
        "і": "i",
        "о": "o",
        "ο": "o",
        "р": "p",
        "с": "c",
        "х": "x",
        "у": "y",
    }
)
LEET_TABLE = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})
SENTINEL_RE = re.compile(r"(?i)\b(?:REVEAL[_-]?MAP|ORIGINAL[_-]?VALUE|DEBUG[_-]?RAW|RAW[_-]?PRIVATE|DO[_-]?NOT[_-]?PRINT)\b")
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
}
GENERIC_PIECES = {
    "likes",
    "work",
    "nervous",
    "budget",
    "launch",
    "billing",
    "fixed",
    "wants",
    "wait",
    "after",
    "private",
    "call",
    "unhappy",
    "invoice",
    "ready",
    "account",
    "issue",
    "resolved",
    "only",
    "agreed",
    "verbally",
    "mentioned",
    "competitor",
    "privately",
    "commission",
    "month",
    "daily",
    "deadline",
    "launches",
    "email",
    "phone",
    "url",
    "org",
    "person",
    "account",
    "author",
    "project",
    "client",
    "clients",
    "customer",
    "company",
    "contact",
    "notes",
    "note",
    "status",
    "home",
    "acme",
    "repo",
    "repos",
    "path",
    "plan",
    "brief",
    "debug",
    "raw",
    "sentinel",
    "reveal",
    "print",
    "test",
    "map",
    "original",
    "value",
    "values",
    "private",
    "billing",
    "clinic",
    "dental",
    "roofing",
    "insurance",
    "agency",
    "plumbing",
    "services",
    "solutions",
    "media",
    "marketing",
    "lawn",
    "model",
    "assistant",
    "source",
    "tool",
    "tools",
    "calls",
    "correctly",
    "huge",
    "context",
    "window",
    "small",
    "local",
    "followed",
    "rubric",
    "labeled",
    "winner",
    "blind",
    "eval",
    "other",
}
APPROVED_PLACEHOLDER_RE = re.compile(
    r"^(?:Person|Org|Email|Phone|Url|Path|Account|Model|Author|Repo|Client|Project)_\d+$|^\[REDACTED_[A-Z_]+\]$"
)


def verify_safe_text(
    safe_text: str,
    policy: Policy,
    *,
    artifact_id: str,
    base_dir: str | Path | None = None,
    canaries: list[str] | None = None,
    canary_field_types: list[str] | None = None,
) -> VerificationResult:
    result = VerificationResult(passed=True)
    canaries = canaries or []
    canary_field_types = canary_field_types or []
    normalized_safe = _normalize_match(safe_text)
    compact_safe = _compact_match(safe_text)

    if SENTINEL_RE.search(safe_text):
        result.reveal_map_leaks += 1
        result.sanitized_failure_ids.append("sentinel:unsafe_marker")

    for idx, canary in enumerate(canaries):
        failure_prefix = f"canary_{idx + 1}"
        field_type = canary_field_types[idx] if idx < len(canary_field_types) else ""
        normalized_canary = _normalize_match(canary)
        compact_canary = _compact_match(canary)
        if normalized_canary and normalized_canary in normalized_safe:
            result.hard_leaks += 1
            result.sanitized_failure_ids.append(f"{failure_prefix}:exact_or_substring")
        elif _normalize_ws(normalized_canary) and _normalize_ws(normalized_canary) in normalized_safe:
            result.hard_leaks += 1
            result.sanitized_failure_ids.append(f"{failure_prefix}:whitespace_normalized")
        elif _should_check_compact(field_type) and len(compact_canary) >= 4 and compact_canary in compact_safe:
            result.hard_leaks += 1
            result.sanitized_failure_ids.append(f"{failure_prefix}:punctuation_normalized")

        if _should_check_pieces(field_type):
            for piece in _meaningful_pieces(canary):
                normalized_piece = _normalize_match(piece)
                compact_piece = _compact_match(piece)
                if normalized_piece in normalized_safe or (len(compact_piece) >= 6 and compact_piece in compact_safe):
                    result.hard_leaks += 1
                    result.sanitized_failure_ids.append(f"{failure_prefix}:meaningful_piece")
                    break

        if _should_check_near(field_type) and _near_match_leak(canary, safe_text):
            result.near_leaks += 1
            result.sanitized_failure_ids.append(f"{failure_prefix}:near_match")

        if _numeric_leak(canary, safe_text):
            result.numeric_leaks += 1
            result.sanitized_failure_ids.append(f"{failure_prefix}:numeric")

    reveal_store = RevealStore(base_dir)
    entries = reveal_store.list_entries(artifact_id)
    for idx, entry in enumerate(entries):
        normalized_original = _normalize_match(entry.original_value)
        if len(_compact_match(entry.original_value)) >= 3 and normalized_original in normalized_safe:
            result.reveal_map_leaks += 1
            result.sanitized_failure_ids.append(f"reveal_entry_{idx + 1}:original_in_safe")
        if _placeholder_contains_original(entry.placeholder, entry.original_value):
            result.reveal_map_leaks += 1
            result.sanitized_failure_ids.append(f"reveal_entry_{idx + 1}:placeholder_hint")
        if not APPROVED_PLACEHOLDER_RE.match(entry.placeholder):
            result.warnings.append(f"placeholder_pattern_{idx + 1}")

    if _coreference_errors(entries):
        result.coreference_errors += 1
        result.sanitized_failure_ids.append("coreference:mapping_conflict")

    if (
        result.hard_leaks
        or result.near_leaks
        or result.numeric_leaks
        or result.reveal_map_leaks
        or result.coreference_errors
    ):
        result.passed = False
    return result


def _normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_match(value: str, *, leet: bool = False) -> str:
    normalized = unicodedata.normalize("NFKC", value).translate(HOMOGLYPH_TABLE).casefold()
    if leet:
        normalized = normalized.translate(LEET_TABLE)
    return _normalize_ws(normalized)


def _compact_match(value: str) -> str:
    return "".join(WORD_RE.findall(_normalize_match(value, leet=True)))


def _meaningful_pieces(value: str) -> list[str]:
    pieces = WORD_RE.findall(_normalize_match(value))
    return [piece for piece in pieces if len(piece) >= 4 and not piece.isdigit() and piece not in GENERIC_PIECES]


def _should_check_pieces(field_type: str) -> bool:
    return field_type in {"person_name", "business_name", "model_name", "email", ""}


def _should_check_near(field_type: str) -> bool:
    return field_type in {"person_name", "model_name", ""}


def _should_check_compact(field_type: str) -> bool:
    return field_type in {"person_name", "business_name", "model_name", "email", "url", "file_path", "account_id", ""}


def _near_match_leak(canary: str, safe_text: str) -> bool:
    canary_words = [word for word in _meaningful_pieces(canary) if 3 <= len(word) <= 20]
    safe_words = [
        word
        for word in WORD_RE.findall(_normalize_match(safe_text, leet=True))
        if 3 <= len(word) <= 20 and not word.isdigit() and word not in GENERIC_PIECES
    ]
    for canary_word in canary_words:
        for safe_word in safe_words:
            if safe_word == canary_word:
                continue
            if abs(len(safe_word) - len(canary_word)) > 1:
                continue
            if _levenshtein(canary_word, safe_word) == 1:
                return True
    return False


def _numeric_leak(canary: str, safe_text: str) -> bool:
    numbers = _numbers_in_text(canary)
    if not numbers:
        return False
    safe_without_placeholders = re.sub(r"\b[A-Z][A-Za-z]+_\d+\b", " ", safe_text)
    safe_numbers = _numbers_in_text(safe_without_placeholders)
    if numbers & safe_numbers:
        return True
    for low, high in _ranges_in_text(safe_without_placeholders):
        if any(low <= number <= high for number in numbers):
            return True
    return False


def _numbers_in_text(value: str) -> set[float]:
    numbers = {_normalize_number(num) for num in NUMBER_RE.findall(value)}
    numbers.update(_spelled_numbers(value))
    return numbers


def _ranges_in_text(value: str) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for match in RANGE_RE.finditer(value):
        first = _normalize_number(match.group(1))
        second = _normalize_number(match.group(2))
        ranges.append((min(first, second), max(first, second)))
    return ranges


def _normalize_number(value: str) -> float:
    return float(value.replace(",", ""))


def _spelled_numbers(value: str) -> set[float]:
    tokens = WORD_RE.findall(_normalize_match(value).replace("-", " "))
    found: set[float] = set()
    current = 0
    active = False
    for token in tokens + ["__flush__"]:
        if token not in NUMBER_WORDS:
            if active:
                found.add(float(current))
                current = 0
                active = False
            continue
        amount = NUMBER_WORDS[token]
        active = True
        if amount == 100:
            current = max(current, 1) * 100
        elif amount == 1000:
            current = max(current, 1) * 1000
        else:
            current += amount
    return found


def _placeholder_contains_original(placeholder: str, original: str) -> bool:
    lower_placeholder = placeholder.lower()
    for piece in _meaningful_pieces(original):
        if piece.lower() in lower_placeholder:
            return True
    return False


def _coreference_errors(entries: list) -> bool:
    seen_original: dict[tuple[str, str], str] = {}
    seen_placeholder: dict[tuple[str, str], str] = {}
    for entry in entries:
        original_key = (entry.field_type, entry.original_sha256)
        placeholder_key = (entry.field_type, entry.placeholder)
        if original_key in seen_original and seen_original[original_key] != entry.placeholder:
            return True
        if placeholder_key in seen_placeholder and seen_placeholder[placeholder_key] != entry.original_sha256:
            return True
        seen_original[original_key] = entry.placeholder
        seen_placeholder[placeholder_key] = entry.original_sha256
    return False


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if ca == cb else 1),
                )
            )
        previous = current
    return previous[-1]
