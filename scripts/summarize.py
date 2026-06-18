"""Print a compact one-line summary of one or more eval aggregate JSON files."""
import json
import sys

KEYS = ["cases_passed", "cases_total", "hard_leaks", "near_leaks", "numeric_leaks",
        "reveal_map_leaks", "coreference_errors", "json_or_process_failures"]
LLM = ["cases_where_llm_prevented_failure", "cases_where_llm_caused_failure",
       "cases_where_llm_changed_output", "llm_json_failures", "llm_timeouts",
       "llm_transport_failures", "llm_spans_accepted", "p50_llm_latency_ms",
       "p95_llm_latency_ms"]

for path in sys.argv[1:]:
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        print(f"{path}: MISSING")
        continue
    g = lambda k: d.get(k, 0)
    line = (f"{g('cases_passed')}/{g('cases_total')}  "
            f"hard={g('hard_leaks')} near={g('near_leaks')} num={g('numeric_leaks')} "
            f"reveal={g('reveal_map_leaks')} coref={g('coreference_errors')} "
            f"procfail={g('json_or_process_failures')}")
    if d.get("llm_mode", "none") != "none":
        line += (f"  | LLM prevented={g('cases_where_llm_prevented_failure')} "
                 f"caused={g('cases_where_llm_caused_failure')} "
                 f"changed={g('cases_where_llm_changed_output')} "
                 f"json_fail={g('llm_json_failures')} timeout={g('llm_timeouts')} "
                 f"accepted={g('llm_spans_accepted')} "
                 f"p50={g('p50_llm_latency_ms')}ms p95={g('p95_llm_latency_ms')}ms")
    print(f"{path.split('/')[-1]:64s} {line}")
