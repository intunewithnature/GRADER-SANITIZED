from __future__ import annotations

import json
import random
from pathlib import Path


PEOPLE = [
    "Bob Calder",
    "Mary Jones",
    "Alex Rivera",
    "Nina Patel",
    "Greg Thompson",
    "Lisa Monroe",
    "Kevin Hartwell",
    "Sara Kim",
]
BUSINESSES = [
    "Smith Roofing",
    "Riverbend Dental",
    "True North Insurance",
    "Midland Lawn Services",
    "Oak Street Clinic",
    "Blue Lantern Media",
]
BUDGETS = ["$25/day", "$75/day", "$150/day", "$500/month", "$1,200/month", "20% commission"]
LOCATIONS = ["Grand Rapids", "Midland", "Detroit", "Henderson", "Saginaw"]
STATUSES = [
    "likes the work but is nervous about budget",
    "will not launch until billing is fixed",
    "wants to wait until after a private call",
    "is unhappy with the invoice",
    "is ready if the account issue is resolved",
    "only agreed verbally",
    "mentioned a competitor privately",
]
TEMPLATES = [
    "{person} at {business} {status}. Budget is {budget} in {location}.",
    "## CRM note\n- Contact: {person}\n- Company: {business}\n- Market: {location}\n- Terms: {budget}\n- Status: {status}",
    "From: {email}\nSubject: launch\nCall {phone}. {business} {status}; proposed spend {budget}.",
    '{{"contact":"{person}","org":"{business}","city":"{location}","budget":"{budget}","note":"{status}"}}',
    "LOG client={business} owner={person} path=/home/acme/{slug}/brief.md budget={budget} status=\"{status}\"",
    "// TODO: Follow up with {person} from {business}; {status}; keep {budget} private.",
    "{person} ({email}) said {business} in {location} {status}. Account acct_{acct} has {budget}.",
    "weird casing note: {person} / {business} / {location} / {budget}. client {status}.",
]


def make_corpus(n: int, out: str | Path, seed: int = 42) -> None:
    rng = random.Random(seed)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for idx in range(n):
            case = make_case(idx + 1, rng)
            fh.write(json.dumps(case, sort_keys=True) + "\n")


def make_case(index: int, rng: random.Random) -> dict:
    person = rng.choice(PEOPLE)
    business = rng.choice(BUSINESSES)
    budget = rng.choice(BUDGETS)
    location = rng.choice(LOCATIONS)
    status = rng.choice(STATUSES)
    email = f"{person.lower().replace(' ', '.')}@example.test"
    phone = f"555-{rng.randint(200, 999)}-{rng.randint(1000, 9999)}"
    acct = "".join(rng.choice("abcdef0123456789") for _ in range(12))
    slug = business.lower().replace(" ", "-")
    raw = rng.choice(TEMPLATES).format(
        person=person,
        business=business,
        budget=budget,
        location=location,
        status=status,
        email=email,
        phone=phone,
        acct=acct,
        slug=slug,
    )
    canaries = [person, business, budget, location, status]
    field_types = ["person_name", "business_name", "budget", "location", "client_status"]
    if email in raw:
        canaries.append(email)
        field_types.append("email")
    if phone in raw:
        canaries.append(phone)
        field_types.append("phone")
    if f"acct_{acct}" in raw:
        canaries.append(f"acct_{acct}")
        field_types.append("account_id")
    if f"/home/acme/{slug}/brief.md" in raw:
        canaries.append(f"/home/acme/{slug}/brief.md")
        field_types.append("file_path")
    return {
        "case_id": f"case_{index:03d}",
        "raw": raw,
        "canaries": canaries,
        "expected_field_types": field_types,
        "expected_semantics": ["client context preserved at a safe level"],
        "notes": "synthetic",
    }
