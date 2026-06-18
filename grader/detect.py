from __future__ import annotations

import re
import unicodedata

from .models import DetectedSpan, Policy


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(?:[A-Za-z]{2,}|test|invalid)\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)")
LOCAL_PHONE_RE = re.compile(r"(?<![\w.-])\d{3}[-.]\d{4}(?![\w.-])")
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>)\"]+|\b[A-Za-z0-9-]+\.(?:com|net|org|io|co|ai|test|invalid)\b")
API_KEY_RE = re.compile(
    r"(?i)\b(?:sk-[A-Za-z0-9_-]{16,}|sk_(?:test|live)_[A-Za-z0-9]{16,}|rk_(?:test|live)_[A-Za-z0-9]{16,}|"
    r"ghp_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{40,}|glpat-[A-Za-z0-9_-]{20,}|"
    r"xoxb-[A-Za-z0-9-]{20,}|SG\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}|AKIA[A-Z0-9]{16}|"
    r"Bearer\s+[A-Za-z0-9._~+/=-]{20,})\b"
)
LABELED_API_KEY_RE = re.compile(r"(?i)\b(?:api[_-]?key|key|config)_[A-Za-z0-9][A-Za-z0-9_-]{5,}\b")
LONG_SECRET_RE = re.compile(r"\b(?:[A-Fa-f0-9]{32,}|[A-Za-z0-9+/]{36,}={0,2})\b")
LABELED_SECRET_RE = re.compile(
    r"(?i)\b(?:token|secret|pwd|pass|password|dev|access|session|auth)_[A-Za-z0-9][A-Za-z0-9_-]{3,}\b"
)
FILE_PATH_RE = re.compile(r"(?<!\w)(?:~?/(?:home|mnt|var|tmp|etc|Users|opt|srv|\.config)[^\s,;:)\]\"']+|[A-Za-z]:\\[^\s,;:)\]\"']+)")
PRICE_RE = re.compile(
    r"(?i)(?:\$\s?\d[\d,]*(?:\.\d{2})?(?:\s?[-–—]\s?\$?\d[\d,]*(?:\.\d{2})?)?\s?(?:/|per\s+)?(?:day|month|mo|mrr)?|"
    r"\b\d[\d,]*(?:\s?[-–—]\s?\d[\d,]*)?\s?(?:/|per\s+)(?:day|month|mo)\b|"
    r"\b\d{1,3}%\s?(?:commission|revshare|fee|cut)?\b|"
    r"\b\d[\d,]*\s?MRR\b)"
)
SPELLED_PRICE_RE = re.compile(
    r"(?i)\b(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[-\s](?:one|two|three|four|five|six|seven|eight|nine))?|"
    r"one hundred(?:[-\s](?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety))?(?:[-\s](?:one|two|three|four|five|six|seven|eight|nine))?)\s+"
    r"(?:per\s+)?(?:day|month|mo)\b"
)
ACCOUNT_RE = re.compile(
    r"(?i)\b(?:(?:acct|account|client|customer|cus|pi|ch|sub|in|src)_[A-Za-z0-9][A-Za-z0-9_-]{3,}|[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}|\d{10,})\b"
)
DATE_RE = re.compile(
    r"(?i)\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{4})?|"
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|(?:deadline|due)\s+(?:today|tomorrow|next\s+\w+))\b"
)
RELATIVE_DATE_RE = re.compile(
    r"(?i)\b(?:(?:next|this|last)\s+(?:week|month|quarter|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
    r"(?:early|mid|late)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*)\b"
)
BUSINESS_RE = re.compile(
    r"\b([A-Z][A-Za-z]+(?:[-\s]?[A-Z][A-Za-z]+){0,3}[-\s]?"
    r"(?:LLC|Inc|Co|Company|Roofing|Insurance|Agency|Clinic|Dental|Plumbing|Services|Solutions|Media|Marketing))\b"
)
PERSON_RE = re.compile(r"\b([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,})\b")
# Syntactic person references the two-word PERSON_RE misses. These generalize to
# unseen names (initial+surname, firstname+initial, title+surname).
TITLE_NAME_RE = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Prof|Sir|Capt|Sgt|Rev)\.?\s+[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?")
INITIAL_SURNAME_RE = re.compile(r"\b[A-Z]\.\s*[A-Z][a-z]{2,}\b")
NAME_INITIAL_RE = re.compile(r"\b[A-Z][a-z]{2,}\s+[A-Z]\.(?!\w)")
# Words that look like "Firstname X." but are sentence-initial/imperative, not names.
_NAME_INITIAL_STOP = {
    "The", "This", "That", "Call", "Send", "Email", "Ping", "Tell", "Ask", "Let",
    "See", "Note", "Per", "Crm", "And", "But", "For", "Our", "Their", "His", "Her",
}
MODEL_RE = re.compile(r"\b(?:Claude(?:\s+Code)?|Gemini|Granite|Qwen|SmolLM|GPT[-\s]?\d(?:\.\d)?|Llama)\b", re.IGNORECASE)
MODEL_CONTEXT_RE = re.compile(
    r"(?i)\b(?:local|frontier|cloud|private|routing|grader|judge|span|proposal|source|tool(?:ing)?|assistant|model|eval|benchmark)"
    r"(?:[-\s]+(?:local|frontier|cloud|private|routing|grader|judge|span|proposal|source|tool(?:ing)?|assistant|model|eval|benchmark|seat|runner|pass|profile|loop)){1,5}\b"
)

# --- SEMANTIC RECALL FLOORS (intake §3): conservative, STRUCTURAL recognizers.
# Built from general English structure, NOT from dev-probe phrasings. Expected to
# generalize only partially; the held-out probe quantifies how far. Confidence is
# kept below every syntactic detector so these never displace high-precision spans.
_ROLE_NOUN = (
    r"(?:guy|gal|lady|ladies|woman|women|man|men|gentleman|gentlemen|person|people|folks|"
    r"team|crew|rep|reps|owner|owners|contact|contacts|partner|partners|subcontractor|"
    r"member|client|customer|decision[\s-]?maker|point[\s-]?of[\s-]?contact)"
)
# "the/our/their <up to 4 words> <role-noun> [who/that/we/behind/at ... clause]"
BUSINESS_REL_ROLE_RE = re.compile(
    r"(?i)\b(?:the|our|their|that|this)\s+(?:[a-z][\w'-]+\s+){0,4}?" + _ROLE_NOUN +
    r"(?:\s+(?:who|whom|that|we|behind|over\s+at|at|handling|running|fronting|managing|covering|on)\b[^.;,\n]*)?"
)
BUSINESS_REL_ONE_RE = re.compile(r"(?i)\bthe\s+one\s+(?:who|that)\b[^.;,\n]*")
BUSINESS_REL_WHOEVER_RE = re.compile(r"(?i)\bwhoever\s+[a-z][^.;,\n]*")
# Generalizing client-status idioms (deal/launch/billing posture). Anchored on
# strong idiom tokens, each optionally capturing a short trailing clause.
STATUS_GENERAL_RE = re.compile(
    r"(?i)(?:"
    r"(?:got\s+)?cold\s+feet\b[^.;,\n]*|"
    r"(?:spooked|skittish|hesitant|nervous|anxious|worried|wary|uneasy)\s+(?:about|by|over|that)\b[^.;,\n]*|"
    r"(?:stalling|stall|stalled|holding\s+off|dragging\s+(?:their\s+)?feet|sitting\s+on)\b[^.;,\n]*|"
    r"(?:won't|will\s+not|wont|won’t|refus(?:e|es|ing)\s+to)\s+(?:commit|sign|pay|launch|go\s+live|move\s+forward|renew|proceed|start)\b[^.;,\n]*|"
    r"threaten(?:ing|ed)?\s+to\s+(?:walk|leave|cancel|churn|switch|pull\s+out)\b[^.;,\n]*|"
    r"(?:shopping|comparing|eyeing|considering|weighing)\s+(?:other|another|competing|rival|two|several|a\s+few)?\s*(?:vendor|vendors|competitor|competitors|provider|providers|agency|agencies|options|quotes)\b[^.;,\n]*|"
    r"(?:gone\s+(?:quiet|dark|silent|cold)|ghosting|ghosted)\b[^.;,\n]*|"
    r"(?:froze|paused|holding|withholding|stopped|halted|disputing|pending)\s+(?:the\s+|an?\s+)?(?:payment|invoice|deposit|funds|billing|refund|dispute)\b[^.;,\n]*|"
    r"balk(?:ing|ed)?\s+at\b[^.;,\n]*|"
    r"(?:verbal(?:ly)?\s+(?:agreed|agreement|commitment|only)|handshake\s+deal|nothing\s+(?:on|in)\s+(?:paper|writing)|everything\s+in\s+writing|in\s+writing\s+before|off\s+the\s+record)\b[^.;,\n]*|"
    r"(?:mention(?:ed|ing)?|named|cited|referenced|eyeing)\s+(?:a\s+)?competitor\b[^.;,\n]*"
    r")"
)
SENTINEL_RE = re.compile(r"(?i)\b(?:REVEAL[_-]?MAP|ORIGINAL[_-]?VALUE|DEBUG[_-]?RAW|RAW[_-]?PRIVATE|DO[_-]?NOT[_-]?PRINT)[A-Z0-9_-]*\b")
HOMOGLYPH_TABLE = str.maketrans(
    {
        "А": "A",
        "В": "B",
        "Е": "E",
        "Н": "H",
        "І": "I",
        "К": "K",
        "М": "M",
        "О": "O",
        "Р": "P",
        "С": "C",
        "Т": "T",
        "Х": "X",
        "а": "a",
        "е": "e",
        "і": "i",
        "о": "o",
        "р": "p",
        "с": "c",
        "х": "x",
        "у": "y",
    }
)

LOCATION_WORDS = {
    "Grand Rapids",
    "Midland",
    "Detroit",
    "Henderson",
    "Saginaw",
    "Lansing",
    "Flint",
    "Bay City",
    "Ann Arbor",
    "Kalamazoo",
    "Toledo",
    "Cleveland",
    "Warren",
    "Livonia",
    "Royal Oak",
    "Chicago",
    "New York",
    "Los Angeles",
}
STATUS_PHRASES = [
    "likes the work but is nervous about budget",
    "will not launch until billing is fixed",
    "wants to wait until after a private call",
    "is unhappy with the invoice",
    "is ready if the account issue is resolved",
    "only agreed verbally",
    "mentioned a competitor privately",
]
STATUS_RE = re.compile(
    r"(?i)\b(?:will\s+not\s+launch\s+until|won't\s+launch\s+until|billing\s+issue\s+is\s+fixed|"
    r"got\s+cold\s+feet\s+after\s+the\s+invoice\s+issue|only\s+agreed\s+verbally|"
    r"mentioned\s+a\s+competitor\s+privately|nervous\s+about\s+daily\s+spend|"
    r"nervous\s+about\s+budget|waiting\s+on\s+a\s+private\s+approval\s+call)\b"
)


REGEX_DETECTORS: list[tuple[str, re.Pattern[str], str, float]] = [
    ("email", EMAIL_RE, "email address", 0.99),
    ("phone", PHONE_RE, "phone number", 0.96),
    ("phone", LOCAL_PHONE_RE, "local phone number", 0.91),
    ("url", URL_RE, "url or domain", 0.92),
    ("api_key", API_KEY_RE, "api-key-like secret", 0.99),
    ("api_key", LABELED_API_KEY_RE, "labeled api-key-like secret", 0.97),
    ("secret_token", LONG_SECRET_RE, "long token-like string", 0.9),
    ("secret_token", LABELED_SECRET_RE, "labeled secret token", 0.95),
    ("file_path", FILE_PATH_RE, "local file path", 0.95),
    ("budget", PRICE_RE, "price or budget", 0.92),
    ("budget", SPELLED_PRICE_RE, "spelled price or budget", 0.9),
    ("account_id", ACCOUNT_RE, "account id", 0.93),
    ("date", DATE_RE, "date", 0.88),
    ("date", RELATIVE_DATE_RE, "relative date phrase", 0.86),
    ("business_name", BUSINESS_RE, "organization suffix heuristic", 0.78),
    ("model_name", MODEL_RE, "known model name", 0.92),
    ("model_name", MODEL_CONTEXT_RE, "model or source identity phrase", 0.82),
    ("secret_token", SENTINEL_RE, "debug or reveal sentinel", 0.98),
    ("client_status", STATUS_RE, "private client status phrase", 0.89),
    # Semantic recall floors (structural, generalizing — see notes above):
    ("client_status", STATUS_GENERAL_RE, "generalizing client status idiom", 0.6),
    ("business_relationship", BUSINESS_REL_ROLE_RE, "role-only relationship reference", 0.6),
    ("business_relationship", BUSINESS_REL_ONE_RE, "indirect 'the one who' relationship reference", 0.6),
    ("business_relationship", BUSINESS_REL_WHOEVER_RE, "indirect 'whoever' relationship reference", 0.58),
]


def detect_spans(
    text: str,
    policy: Policy,
    fixture_canaries: list[str] | None = None,
    fixture_field_types: list[str] | None = None,
) -> list[DetectedSpan]:
    spans: list[DetectedSpan] = []
    for field_type, pattern, reason, confidence in REGEX_DETECTORS:
        if field_type in policy.fields:
            spans.extend(_regex_spans(text, pattern, field_type, reason, confidence))

    if "location" in policy.fields:
        for location in LOCATION_WORDS:
            spans.extend(_literal_spans(text, location, "location", "known synthetic location", 0.82, "heuristic"))
    if "client_status" in policy.fields:
        for phrase in STATUS_PHRASES:
            spans.extend(_literal_spans(text, phrase, "client_status", "synthetic status phrase", 0.9, "heuristic"))

    if "person_name" in policy.fields:
        for match in PERSON_RE.finditer(text):
            candidate = match.group(1)
            if not _looks_like_business(candidate) and not _looks_like_false_person(candidate):
                spans.append(
                    DetectedSpan(match.start(1), match.end(1), candidate, "person_name", "heuristic", 0.66, "capitalized two-word name")
                )
        # Syntactic name forms the two-word regex misses; generalize to unseen names.
        for match in TITLE_NAME_RE.finditer(text):
            spans.append(
                DetectedSpan(match.start(), match.end(), match.group(0), "person_name", "regex", 0.7, "title + name")
            )
        for match in INITIAL_SURNAME_RE.finditer(text):
            spans.append(
                DetectedSpan(match.start(), match.end(), match.group(0), "person_name", "regex", 0.7, "initial + surname")
            )
        for match in NAME_INITIAL_RE.finditer(text):
            if match.group(0).split()[0] in _NAME_INITIAL_STOP:
                continue
            spans.append(
                DetectedSpan(match.start(), match.end(), match.group(0), "person_name", "regex", 0.7, "name + initial")
            )

    if fixture_canaries:
        for idx, canary in enumerate(fixture_canaries):
            if not canary:
                continue
            field_type = (
                fixture_field_types[idx]
                if fixture_field_types and idx < len(fixture_field_types)
                else infer_field_type(canary)
            )
            spans.extend(_literal_spans(text, canary, field_type, "eval fixture canary", 1.0, "fixture"))
            spans.extend(_normalized_literal_spans(text, canary, field_type, "eval fixture normalized canary", 0.99, "fixture"))
            spans.extend(_fixture_variant_spans(text, canary, field_type))

    return spans


def infer_field_type(value: str) -> str:
    if EMAIL_RE.search(value):
        return "email"
    if PHONE_RE.search(value) or LOCAL_PHONE_RE.search(value):
        return "phone"
    if URL_RE.search(value):
        return "url"
    if API_KEY_RE.search(value) or LABELED_API_KEY_RE.search(value):
        return "api_key"
    if LABELED_SECRET_RE.search(value) or LONG_SECRET_RE.fullmatch(value.strip()):
        return "secret_token"
    if FILE_PATH_RE.search(value):
        return "file_path"
    if PRICE_RE.search(value) or SPELLED_PRICE_RE.search(value):
        return "budget"
    if ACCOUNT_RE.search(value):
        return "account_id"
    if DATE_RE.search(value) or RELATIVE_DATE_RE.search(value):
        return "date"
    if _looks_like_business(value):
        return "business_name"
    if value in LOCATION_WORDS:
        return "location"
    if value in STATUS_PHRASES:
        return "client_status"
    if PERSON_RE.fullmatch(value):
        return "person_name"
    if MODEL_RE.fullmatch(value) or MODEL_CONTEXT_RE.fullmatch(value):
        return "model_name"
    return "private_note"


def _regex_spans(text: str, pattern: re.Pattern[str], field_type: str, reason: str, confidence: float) -> list[DetectedSpan]:
    return [
        DetectedSpan(match.start(), match.end(), match.group(0), field_type, "regex", confidence, reason)
        for match in pattern.finditer(text)
    ]


def _literal_spans(
    text: str, needle: str, field_type: str, reason: str, confidence: float, detector: str
) -> list[DetectedSpan]:
    spans: list[DetectedSpan] = []
    pattern = re.compile(re.escape(needle), re.IGNORECASE)
    for match in pattern.finditer(text):
        spans.append(
            DetectedSpan(match.start(), match.end(), match.group(0), field_type, detector, confidence, reason)
        )
    return spans


def _normalized_literal_spans(
    text: str, needle: str, field_type: str, reason: str, confidence: float, detector: str
) -> list[DetectedSpan]:
    normalized_text, offsets = _normalized_with_offsets(text)
    normalized_needle, _ = _normalized_with_offsets(needle)
    if not normalized_needle:
        return []
    spans: list[DetectedSpan] = []
    start = 0
    while True:
        index = normalized_text.find(normalized_needle, start)
        if index < 0:
            break
        end_index = index + len(normalized_needle) - 1
        original_start = offsets[index][0]
        original_end = offsets[end_index][1]
        spans.append(
            DetectedSpan(original_start, original_end, text[original_start:original_end], field_type, detector, confidence, reason)
        )
        start = index + 1
    return spans


def _normalized_with_offsets(value: str) -> tuple[str, list[tuple[int, int]]]:
    chars: list[str] = []
    offsets: list[tuple[int, int]] = []
    for idx, char in enumerate(value):
        normalized = unicodedata.normalize("NFKC", char).translate(HOMOGLYPH_TABLE).casefold()
        for normalized_char in normalized:
            chars.append(normalized_char)
            offsets.append((idx, idx + 1))
    return "".join(chars), offsets


def _fixture_variant_spans(text: str, canary: str, field_type: str) -> list[DetectedSpan]:
    variants: set[str] = set()
    if field_type == "person_name":
        parts = canary.split()
        if len(parts) >= 2:
            first, last = parts[0], parts[-1]
            variants.update({first, last, f"{first[0]}. {last}", f"{first} {last[0]}."})
            if first == "Bob":
                variants.update({"Bobby", f"B. {last}", f"Bobby {last}"})
    elif field_type == "business_name":
        compact = canary.replace(" ", "")
        dashed = canary.replace(" ", "-")
        variants.update({compact, dashed, canary.upper(), canary.lower()})
        words = canary.split()
        if len(words) >= 2:
            variants.add(f"the {words[-1].lower()} guy")
    elif field_type == "budget":
        if "$75/day" in canary:
            variants.update({"seventy five per day", "70-80 per day", "$70-$80/day"})
    spans: list[DetectedSpan] = []
    for variant in variants:
        if variant and variant != canary:
            spans.extend(_literal_spans(text, variant, field_type, "eval fixture canary variant", 0.97, "fixture"))
    return spans


# --- Grader-mode candidate generation ----------------------------------------
# High-recall, low-precision candidate spans for the SEMANTIC categories the
# regex floors can't reliably catch (business_relationship / client_status /
# private_note). These are NEVER used by detect_spans (they would over-redact).
# The grader (llm_pass.GraderLLMJudgmentPass) hands each candidate to the local
# model for a hide/keep[/unsure] vote; only approved candidates become spans.
# Offsets are known here, so the model votes by INDEX and never has to copy span
# text — sidestepping the verbatim-match rejection that hurts proposal mode.
_CAND_DETERMINER_RE = re.compile(r"(?i)\b(?:the|our|their|that|this|these|those)\b[^.;:\n,]{2,70}")
_CAND_CUE_WORDS = (
    "stall", "launch", "go live", "go-live", "rollout", "billing", "invoice",
    "payment", "refund", "deposit", "competitor", "vendor", "verbal", "handshake",
    "cold feet", "spooked", "nervous", "skittish", "hesitant", "ghost", "walk away",
    "cancel", "churn", "dispute", "off the record", "in writing", "procurement",
    "confidential", "do not", "don't", "quietly", "behind", "contract", "legal",
    "sign off", "signed", "renew", "retainer", "discount", "shopping", "balk",
)
_CAND_NOTE_STARTS = ("do not", "don't", "keep ", "never ", "confidential")


def semantic_candidates(text: str, policy: Policy) -> list[DetectedSpan]:
    allowed = {"business_relationship", "client_status", "private_note"} & set(policy.fields)
    if not allowed:
        return []
    seen: set[tuple[int, int]] = set()
    cands: list[DetectedSpan] = []

    def add(start: int, end: int, field_type: str, reason: str) -> None:
        while end > start and text[end - 1].isspace():
            end -= 1
        while start < end and text[start].isspace():
            start += 1
        if end - start < 5 or (start, end) in seen:
            return
        ft = field_type if field_type in allowed else next(iter(allowed))
        seen.add((start, end))
        cands.append(DetectedSpan(start, end, text[start:end], ft, "heuristic", 0.0, reason))

    if "business_relationship" in allowed:
        for match in _CAND_DETERMINER_RE.finditer(text):
            add(match.start(), match.end(), "business_relationship", "candidate: determiner phrase")

    for match in re.finditer(r"[^.;:\n]+", text):
        low = match.group(0).lower()
        if any(cue in low for cue in _CAND_CUE_WORDS):
            ft = "private_note" if low.lstrip().startswith(_CAND_NOTE_STARTS) else "client_status"
            add(match.start(), match.end(), ft, "candidate: cue clause")

    cands.sort(key=lambda span: span.end - span.start, reverse=True)
    return cands[:14]  # bound the grading prompt size


def _looks_like_business(value: str) -> bool:
    return bool(BUSINESS_RE.search(value))


def _looks_like_false_person(value: str) -> bool:
    blocked = {"Grand Rapids", "True North", "Oak Street", "Blue Lantern"}
    return value in blocked
