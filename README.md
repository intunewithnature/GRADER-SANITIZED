# Grader

**A local context-blinding compiler** — it strips private client context out of working
notes so the *safe* version can be handed to a cloud LLM, with a canary-based **verifier**
as the final authority on whether anything leaked, plus an optional **LLM "grader"** pass
whose value is measured honestly on a **frozen held-out set**.

The interesting part isn't the regexes — it's the **measurement discipline**: every
"improvement" is checked against held-out data it was never tuned on, so the numbers
reflect real generalization, not memorized test answers.

---

## The problem

You want to use a cloud model on private notes ("the silent partner behind Ridgeline
Dental froze payment pending a dispute") without leaking who the client is. Pattern-based
redaction handles the easy stuff (emails, paths, dollar amounts) but misses *contextual*
identity — role-only references, deal/billing status, indirect descriptions. A small
**local** model could catch those... if it actually helps. This project measures whether
it does, and makes it safe to use when it does.

## Design — three layers, one authority

1. **Detectors** (`grader/detect.py`) — fast, high-precision regex/heuristics for syntactic
   PII (emails, phones, URLs, file paths, account IDs, dates, budgets, model names, name
   initials) plus conservative structural floors for semantic categories.
2. **Verifier** (`grader/verify.py`) — the **final authority**. It plants canaries (the real
   sensitive strings) in eval cases and checks the safe text for any survivor (exact,
   whitespace/punctuation-normalized, homoglyph/leet-folded, meaningful-piece, near-match,
   numeric, and reveal-map leaks). It is never weakened to make a pass rate look better.
3. **Local LLM (optional)** — proposes spans only; the verifier still decides. Runs on
   [Ollama](https://ollama.com) with `granite3.3:2b`, CPU-only, bound to localhost.

## Measurement integrity (the core idea)

Tuning a detector to a dev probe's *exact phrasings* inflates that probe's score without
improving real recall. So capability is measured on a **frozen held-out set**
(`evals/private/contextual_heldout_100.jsonl`) — same sensitivity *categories*, deliberately
*disjoint* surface phrasings — that the detectors were never tuned against.

| detector-only, no answer key | passed |
|---|---|
| dev probe (tuned against) | **63 / 100** |
| **held-out** (never tuned against) | **23 / 100** |

That **63 → 23** gap is the overfit, made visible. Held-out is the real number. (Fixture-on
safety, where canaries are provided, stays at 100% with zero leaks across 850 adversarial
cases — that path is the safety gate, not the capability measure.)

## Two results

### 1. A displacement bug — found by measurement, fixed for free
Overlap resolution used to keep the single highest-confidence span and drop the rest. A
confident *narrow* LLM span could evict a wider detector span that was covering a canary —
silently **un-redacting** it. The fix: **merge** overlapping redactions (union of ranges)
so an added span can only ever *increase* coverage. On held-out this took the LLM pass from
net **+1** (5 saved / 4 *caused*) to net **+9** (10 saved / ~0 caused).

### 2. Flip the model from author to grader
A small model is weak at *generating* spans (open-ended; wrong targets; needs verbatim
copies) but good at *classifying*. So the detector over-generates candidate phrases and the
model only votes **hide / keep** on each — one batched call, voting by index (no
verbatim-match problem, no displacement). A `triage` mode adds an **unsure** vote routed to
*hide* (fail-safe).

## Results — first-party held-out, no answer key

| pass | passed | prevented | caused | latency |
|---|---|---|---|---|
| detector-only | 23 / 100 | — | — | — |
| LLM propose (pre-fix) | 24 / 100 | 5 | **4** | 6.1 s |
| LLM propose (post-fix) | 32 / 100 | 10 | ~0\* | 6.4 s |
| **grade — binary** | 33 / 100 | 11 | ~0\* | 4.6 s |
| **grade — triage** | **34 / 100** | 12 | ~0\* | 4.6 s |

An off-the-shelf 2B model went from "23 and occasionally *unsafe*" to "34 and safe" — with
no fine-tuning, just a bug fix and an architecture change. The grader is also ~30% faster
than free-form proposal and can't displace detector coverage by design.

### Independent confirmation (third-party held-out)

The set above is first-party. A second held-out set authored by a *different* model
(`evals/private/independent_150.jsonl`, 150 cases, 61 `business_relationship`, **1/172**
phrasing overlap) was scored once — neither the detectors nor the author shaped it:

| fixture-off | detector-only | grade triage | caused |
|---|---|---|---|
| first-party held-out (**4.1** spans/case) | 23 % | 34 % | ~0 |
| **independent** (**1.8** spans/case) | **26 %** | **82 %** | **0** |

- **The overfit gap replicates on data we didn't write:** detector-only is 26% here vs 63%
  on the dev probe.
- **The grader's lift and safety replicate:** it prevents 84 failures and causes **0**
  (hit-rate 55%).
- **Absolute pass rates track case difficulty, by design.** A case passes only if *every*
  canary is caught; the first-party set is a deliberately dense stress test at **4.1**
  sensitive spans/case vs **1.8** here — that, not a behavior change, is why 82% vs 34%.
  The honest unit is the *per-set lift* and the *caused count*, not a single headline %.

## Honest verdict
Worth enabling as a **verifier-gated, offline/batch** pass; not as a default real-time pass
at ~4.6 s/case on CPU. Fine-tuning is deliberately **deferred but now justified and
targeted**: the held-out gap is real and concentrated on semantic categories, so a small
adapter has a clear target and a built-in generalization check — rather than being adopted
on a hunch.

\* The lone "caused" failure on the first-party set is a **verifier false-positive, not a
leak**: the grader generalizes a clause to *"a private business relationship"*, and the
verifier flags the name *"Ines D."* because **"ines" is a substring of "bus·ines·s"**. The
name is genuinely redacted; the gate is just conservative on short names — the correct bias
for a safety gate, so it was left as-is. (See `REPORT.md`.)

## Run it

```bash
# 1. local model (no sudo, localhost only)
ollama serve & ollama pull granite3.3:2b

# 2. tests + a deterministic eval
python3 -m unittest discover -q
python3 -m grader eval --suite evals/private/contextual_heldout_100.jsonl \
  --policy policies/client_strategy.json --out-dir evals/results --fixture-transform off

# 3. the grader (triage), measured on held-out
python3 -m grader eval --suite evals/private/contextual_heldout_100.jsonl \
  --policy policies/client_strategy.json --out-dir evals/results --fixture-transform off \
  --llm ollama --llm-model granite3.3:2b --llm-strategy grade --grader-verdicts triage
python3 scripts/summarize.py evals/results/*.aggregate.json
```

`--fixture-transform on` gives the model the answer key (safety-gate / regression use);
`off` is the real capability measure.

## Layout
- `grader/` — `detect` (detectors + grader candidates), `verify` (authority), `transform`
  (blinding + union-merge), `llm_pass` (propose + grader), `eval_runner`, `generalize`.
- `policies/` — field operations (redact / pseudonymize / generalize).
- `evals/private/` — **synthetic** eval suites (fake orgs/people/domains): the dev probe,
  the frozen first-party held-out + its generator, and the independent third-party set.
- `REPORT.md` — full methodology and results.

## Notes
- All data is **synthetic**. No real client data, secrets, or PII.
- The local model proposes; the verifier decides. No cloud calls; the model binds to
  `127.0.0.1` only.

## License
MIT — see [LICENSE](LICENSE).
