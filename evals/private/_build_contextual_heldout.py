"""Build evals/private/contextual_heldout_100.jsonl — a FROZEN held-out probe.

GRADER measurement-integrity discipline (intake §2):
  - Same sensitivity CATEGORIES as the contextual dev probe
    (client_status, business_relationship, business_name, person_name,
     model_name, file_path, location, budget) ...
  - ... but DIFFERENT surface phrasings: paraphrases / synonyms / novel
    constructions that are NOT present in contextual_llm_probe_100,
    no_fixture_probe_100, or the spec's STATUS_PHRASES / STATUS_RE lists.

The point of the held-out set: a detector that *generalizes* (semantic
understanding) catches both dev and held-out; a detector that *memorized*
the dev phrasings catches dev but fails here. The dev-vs-held-out gap is the
overfit. Deterministic (seeded) so the frozen file is reproducible.

All entities are SYNTHETIC (fake orgs / people / domains), per the rules.
Build once, freeze, do NOT inspect per-case failures during detector dev.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 20260617
OUT = Path(__file__.replace("_build_contextual_heldout.py", "contextual_heldout_100.jsonl"))

# --- synthetic, fake fillers (none overlap the dev-probe entities) -----------
TRADES = ["dental", "orthodontics", "hvac", "landscaping", "paving", "electrical",
          "veterinary", "chiropractic", "catering", "auto-detailing", "pool-service",
          "window-tint", "tree-care", "septic", "drywall"]
ORG_BASES = ["Kestrel", "Ridgeline", "Vantage", "Northgate", "Brightwater", "Cardinal",
             "Sablefield", "Lumen", "Ironwood", "Marigold", "Pinnacle", "Thornbury",
             "Halcyon", "Westvale", "Quill", "Ember", "Drayton", "Foxglen"]
ORG_SUFFIX = ["Dental", "Plumbing Co", "Marketing LLC", "Roofing", "Clinic", "Agency",
              "Insurance", "Solutions", "Services", "Media", "Dental Group Inc"]
FIRST = ["Marcus", "Priya", "Tomas", "Adaeze", "Lena", "Rashid", "Cormac", "Yuki",
         "Desmond", "Ines", "Bartek", "Noor", "Sven", "Camila", "Theo", "Ottilie"]
LAST = ["Vance", "Okafor", "Brennan", "Delgado", "Holloway", "Nakamura", "Esposito",
        "Farrow", "Castellanos", "Whitfield", "Abara", "Mholt", "Renthrea", "Sandoval"]
# cities deliberately NOT in detect.LOCATION_WORDS (tests location generalization)
CITIES = ["Kalamazoo", "Toledo", "Ann Arbor", "Flint", "Lansing", "Battle Creek",
          "Akron", "Dayton", "Muskegon", "Elkhart", "Sandusky", "Bloomington"]

# business_relationship: role-only references, structures disjoint from dev probe
BREL_TEMPLATES = [
    "our main contact over at the {trade} outfit",
    "the lady who runs their {trade} front desk",
    "whoever signed the renewal last quarter",
    "their decision-maker on the retainer",
    "the franchise owner we onboarded in spring",
    "the {trade} subcontractor we keep rehiring",
    "the gentleman fronting the expansion deal",
    "the person who actually controls their budget",
    "the silent partner behind the account",
    "the {trade} crew lead we coordinate with",
    "the rep who escalates everything to legal",
    "the family member who co-signs their invoices",
]
# client_status: deal/launch/billing posture, disjoint idioms from STATUS_RE
BSTATUS_TEMPLATES = [
    "they keep stalling the rollout until legal clears it",
    "still spooked by the monthly number",
    "threatening to walk unless we discount",
    "has gone quiet on us since the demo",
    "won't commit without a refund guarantee",
    "is quietly shopping two other vendors",
    "balking hard at the retainer size",
    "insists on everything in writing before go-live",
    "froze payment pending an internal dispute",
    "turned lukewarm right after the pricing call",
    "wants a handshake deal, nothing on paper",
    "leaning toward a cheaper competitor they named off the record",
]
# model_name as contextual description, disjoint from dev probe descriptions
MODEL_TEMPLATES = [
    "the engine we benchmarked second",
    "that long-memory system from the bake-off",
    "the bot that nailed the function-calling test",
    "the lightweight one we run on-box",
    "whichever model topped our internal rubric",
    "the reasoning model the client preferred blind",
    "the smaller system that beat the big one on cost",
]
# private_note: free-form sensitive aside
PNOTE_TEMPLATES = [
    "do not forward this thread to their procurement team",
    "keep the margin breakdown off any shared doc",
    "the owner's divorce is slowing every approval",
    "their CFO is under audit, tread carefully",
    "we lowballed the first bid on purpose",
]


def org(rng):
    return f"{rng.choice(ORG_BASES)} {rng.choice(ORG_SUFFIX)}"


def person(rng):
    style = rng.random()
    f, l = rng.choice(FIRST), rng.choice(LAST)
    if style < 0.45:
        return f"{f} {l}"
    if style < 0.65:
        return f"{f[0]}. {l}"          # initial + surname
    if style < 0.82:
        return f"{f} {l[0]}."          # first + initial
    if style < 0.92:
        return f"Dr. {l}"
    return f


def path(rng):
    base = rng.choice(["/home/acme/clients", "/srv/vault", "/home/acme/accounts",
                       "/data/crm/notes", "~/work/clients"])
    slug = f"{rng.choice(ORG_BASES).lower()}-{rng.choice(TRADES)}"
    fname = rng.choice(["strategy.md", "q3-plan.txt", "pricing.md", "renewal.txt", "notes.md"])
    return f"{base}/{slug}/{fname}"


def budget(rng):
    style = rng.random()
    if style < 0.35:
        return f"${rng.choice([90,110,140,175,210,260])} a day"
    if style < 0.6:
        return f"around {rng.choice([2,3,4,6,8])}k a month"
    if style < 0.8:
        return f"{rng.choice(['twelve hundred','nine hundred','two thousand'])} monthly"
    return f"a {rng.choice([12,15,18,20,25])}% cut"


def build(rng, i):
    """Compose one case from 1-3 disjoint-phrased sensitive spans."""
    canaries: list[str] = []
    fts: list[str] = []
    parts: list[str] = []

    def add(text, ft):
        canaries.append(text)
        fts.append(ft)

    # every contextual case has a relationship OR status anchor (the hard part)
    anchor = rng.random()
    if anchor < 0.5:
        brel = rng.choice(BREL_TEMPLATES).format(trade=rng.choice(TRADES))
        add(brel, "business_relationship")
        parts.append(f"CRM: {brel} is")
        stat = rng.choice(BSTATUS_TEMPLATES)
        add(stat, "client_status")
        parts.append(f" {stat}")
    else:
        stat = rng.choice(BSTATUS_TEMPLATES)
        add(stat, "client_status")
        parts.append(f"Note: the account {stat}")

    # optionally name the org (syntactic - has a suffix)
    if rng.random() < 0.65:
        o = org(rng)
        add(o, "business_name")
        parts.append(f"; this is for {o}")

    # optionally a person, model description, path, location, budget, note
    extras = []
    if rng.random() < 0.45:
        p = person(rng); add(p, "person_name"); extras.append(f"loop in {p}")
    if rng.random() < 0.30:
        m = rng.choice(MODEL_TEMPLATES); add(m, "model_name")
        extras.append(f"they liked {m}")
    if rng.random() < 0.28:
        pa = path(rng); add(pa, "file_path"); extras.append(f"see {pa}")
    if rng.random() < 0.30:
        c = rng.choice(CITIES); add(c, "location"); extras.append(f"based in {c}")
    if rng.random() < 0.35:
        b = budget(rng); add(b, "budget"); extras.append(f"they floated {b}")
    if rng.random() < 0.18:
        n = rng.choice(PNOTE_TEMPLATES); add(n, "private_note"); extras.append(n)

    if extras:
        parts.append(". " + "; ".join(extras))
    raw = "".join(parts).strip()
    if not raw.endswith("."):
        raw += "."

    # sanity: every canary must occur verbatim in raw
    for can in canaries:
        assert can in raw, f"canary not in raw: {can!r} :: {raw!r}"

    return {
        "case_id": f"heldout_{i:03d}",
        "raw": raw,
        "canaries": canaries,
        "expected_field_types": fts,
        "expected_semantics": ["contextual identity/relationship is sensitive"],
        "notes": "held-out: disjoint phrasings, same categories as dev probe",
    }


def main():
    rng = random.Random(SEED)
    rows = [build(rng, i + 1) for i in range(100)]
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    # report category distribution only (no phrasing dump)
    from collections import Counter
    c = Counter(ft for r in rows for ft in r["expected_field_types"])
    print(f"wrote {len(rows)} cases -> {OUT}")
    for ft, n in c.most_common():
        print(f"  {ft:24s} {n}")


if __name__ == "__main__":
    main()
