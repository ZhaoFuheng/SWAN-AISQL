#!/usr/bin/env python3
"""Small-scale verification of the prefix-cache client policy design, at protocol level.

Workload: prefixes with mixed fan-out (FANOUTS, e.g. three used once + three used six times),
mimicking a join where some reps keep partners and some are pruned to a single pair. Policies:

  plain        no breakpoints (baseline)
  always_write breakpoint on every call (naive: pays the 1.25x write even for singletons)
  adaptive     1st arrival of a prefix goes plain; 2nd arrival writes; rest read  (proposed default)
  oracle       breakpoint only for prefixes with fanout>=2, first call writes     (= hinted operators)

Cost accounting (input side only, GPT-5.6: 1x uncached / 1.25x write / 0.1x read): usage reports
cached_tokens on reads; writes are inferred from the policy trace (first breakpoint call on a cold
prefix) and their prefix size taken from sibling reads' cached_tokens. Output nondeterminism does
not touch any of these numbers.

Also PARK-NECESSITY: fire K same-prefix breakpoint requests concurrently with no parking -> expect
~0 cached reads (all miss the in-flight write); then K more after the write completed -> all read.
"""
import json
import os
import threading
import time
import urllib.request

UP = os.environ.get("UPSTREAM", "http://127.0.0.1:4000")
MODEL = os.environ.get("AI_MODEL", "gpt-5.6-luna")
FANOUTS = [int(x) for x in os.environ.get("FANOUTS", "1,1,1,6,6,6").split(",")]
P_IN = 0.25  # $/1M input; write 1.25x, read 0.1x

SYS = "You judge whether a candidate accessory is compatible with a product. Answer yes or no."
BODYTXT = ("a weather-sealed aluminium component with reinforced housing, precision bearings, "
           "quick-release levers and a matte anodized finish designed for demanding daily use. ") * 45


def call(prefix, suffix, breakpoint_):
    if breakpoint_:
        content = [{"type": "text", "text": prefix, "prompt_cache_breakpoint": {"mode": "explicit"}},
                   {"type": "text", "text": suffix}]
        body = {"model": MODEL, "temperature": 0, "messages": [
            {"role": "system", "content": SYS}, {"role": "user", "content": content}],
            "prompt_cache_options": {"mode": "explicit"}}
    else:
        body = {"model": MODEL, "temperature": 0, "messages": [
            {"role": "system", "content": SYS}, {"role": "user", "content": prefix + suffix}]}
    req = urllib.request.Request(UP + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    u = json.load(urllib.request.urlopen(req, timeout=120)).get("usage", {})
    return u.get("prompt_tokens", 0), (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)


def prefixes(policy):
    # unique per (policy, run) so no cross-policy cache contamination
    salt = f"[{policy} {time.time():.0f}] "
    return [salt + f"Product {i}: " + BODYTXT for i in range(len(FANOUTS))]


def run_policy(policy):
    pfx = prefixes(policy)
    tot_p = tot_cached = write_pfx_tok = calls = 0
    for i, fan in enumerate(FANOUTS):
        seen = 0
        read_cached = []
        for j in range(fan):
            if policy == "plain":
                bp = False
            elif policy == "always_write":
                bp = True
            elif policy == "adaptive":
                bp = seen >= 1          # 1st plain; 2nd is the write; rest read
            else:                        # oracle: hinted fanout>=2 -> first call writes
                bp = fan >= 2
            p, cached = call(pfx[i], f" Candidate: accessory variant {j}", bp)
            calls += 1
            tot_p += p
            tot_cached += cached
            if cached:
                read_cached.append(cached)
            seen += 1
        # infer written prefix tokens (for the 0.25x write premium) from sibling reads
        wrote = (policy == "always_write") or (policy == "adaptive" and fan >= 2) \
            or (policy == "oracle" and fan >= 2)
        if wrote:
            write_pfx_tok += read_cached[0] if read_cached else int(len(pfx[i]) / 4)
    cost = ((tot_p - tot_cached) / 1e6) * P_IN + (tot_cached / 1e6) * P_IN * 0.1 \
        + (write_pfx_tok / 1e6) * P_IN * 0.25
    return calls, tot_p, tot_cached, cost


def park_necessity():
    pfx = prefixes("park")[0]
    K = 4
    res = [None] * K

    def worker(k, tag):
        res[k] = call(pfx, f" Candidate: concurrent item {tag}-{k}", True)

    ts = [threading.Thread(target=worker, args=(k, "burst")) for k in range(K)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    burst_cached = sum(c for _, c in res)
    burst_tot = sum(p for p, _ in res)
    for k in range(K):  # write has completed now; sequential siblings should all read
        worker(k, "after")
    after_cached = sum(c for _, c in res)
    after_tot = sum(p for p, _ in res)
    print(f"park-necessity: {K} concurrent same-prefix (no parking): cached {burst_cached}/{burst_tot} "
          f"({100.0 * burst_cached / burst_tot:.0f}%)")
    print(f"                {K} after write completed (parked-equiv):  cached {after_cached}/{after_tot} "
          f"({100.0 * after_cached / after_tot:.0f}%)")


def main():
    print(f"fanouts={FANOUTS} ({sum(FANOUTS)} calls/policy), model={MODEL}")
    print(f"{'policy':<13} {'calls':>5} {'prompt_tok':>11} {'cached_tok':>11} {'input$':>9}")
    base = None
    for policy in ["plain", "always_write", "adaptive", "oracle"]:
        calls, p, cached, cost = run_policy(policy)
        base = base or cost
        print(f"{policy:<13} {calls:>5} {p:>11} {cached:>11} {cost:>9.5f}  ({cost / base:.2f}x plain)")
    park_necessity()


main()
