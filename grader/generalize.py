from __future__ import annotations

import re
from datetime import date


NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
NUMBER_WORD_VALUES = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "hundred": 100,
}


def generalize(field_type: str, value: str, style: str | None = None) -> str:
    if field_type in {"exact_price", "budget"}:
        return _budget(value)
    if field_type == "location":
        return "a regional location"
    if field_type == "date":
        return _date(value)
    if field_type == "client_status":
        return "a private client status"
    if field_type == "business_relationship":
        return "a private business relationship"
    if field_type == "private_note":
        return "a private contextual note"
    return f"a private {field_type.replace('_', ' ')}"


def _budget(value: str) -> str:
    lower = value.lower()
    nums = [float(num.replace(",", "")) for num in NUMBER_RE.findall(value)]
    if not nums:
        spelled = _spelled_amount(lower)
        if spelled is not None:
            nums = [float(spelled)]
    if not nums:
        return "a business budget amount"
    amount = nums[0]
    if "%" in lower or "commission" in lower:
        return "a private commission arrangement"
    if "month" in lower or "/mo" in lower or "mrr" in lower:
        if amount < 500:
            return "a small monthly budget"
        if amount <= 1500:
            return "a modest monthly budget"
        return "a significant monthly budget"
    if amount < 50:
        return "a small daily budget"
    if amount <= 150:
        return "a modest daily budget"
    if amount <= 500:
        return "a significant daily budget"
    return "a business budget amount"


def _spelled_amount(value: str) -> int | None:
    tokens = re.findall(r"[a-z]+", value.replace("-", " "))
    current = 0
    active = False
    for token in tokens:
        if token not in NUMBER_WORD_VALUES:
            if active:
                break
            continue
        active = True
        amount = NUMBER_WORD_VALUES[token]
        if amount == 100:
            current = max(current, 1) * 100
        else:
            current += amount
    return current if active else None


def _date(value: str) -> str:
    if "deadline" in value.lower():
        return "a deadline"
    year_match = re.search(r"\b(20\d{2})\b", value)
    if year_match:
        year = int(year_match.group(1))
        return "a future date" if year > date.today().year else "a recent date"
    return "a recent date"
