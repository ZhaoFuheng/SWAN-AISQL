#!/usr/bin/env python3
"""Explicit vs implicit provider prompt caching on the sem_filter(a, b) pattern.

Isolates the INPUT-side caching effect (cached_tokens + input cost) from LLM output
nondeterminism by summing cached_tokens across all pairs -- caching is a prompt-prefix property,
independent of the yes/no judgment the model returns. Each request mimics exactly what
sem_filter(a, b) emits: system prompt + a-record (a long, cacheable left side) + b-item.

Modes (same L*R pairs, real API via litellm :4000):
  nocache   -- a unique nonce prepended to every request -> nothing caches (baseline)
  implicit  -- default caching; the (system + a_i) prefix repeats across b-items
  explicit  -- prompt_cache_options mode=explicit + a breakpoint ending the a_i block

Dispatch order is the lever: 'contiguous' groups a_i's b-items together (implicit's best case);
'interleaved' round-robins over b (a_i's prefix is revisited only every R requests -- the shape
adaptive/diversity dispatch produces). Reports cached_tokens and input-token cost per mode/order.
"""
import json
import os
import sys
import time
import urllib.request

UP = os.environ.get("UPSTREAM", "http://127.0.0.1:4000")
MODEL = os.environ.get("AI_MODEL", "gpt-5.6-luna")
L = int(os.environ.get("L", "4"))
R = int(os.environ.get("R", "6"))
# GPT-5.6 input pricing (per 1M): uncached 0.25, cache write 1.25x, cache read 0.1x.
P_IN, P_WRITE, P_READ = 0.25, 0.25 * 1.25, 0.25 * 0.1

SYS = "You judge whether a candidate accessory is compatible with a product. Answer yes or no."
# each left record ~1.6k tokens so system+left clears the 1,024 cacheable minimum
LEFT = [f"Product {i}: " + ("a weather-sealed aluminium component with reinforced housing, precision "
        "bearings, quick-release levers and a matte anodized finish designed for demanding daily "
        "commuting use. ") * 60 for i in range(L)]
RIGHT = [f"accessory variant {j}" for j in range(R)]


def call(a_text, b_text, mode, nonce=""):
    if mode == "explicit":
        content = [{"type": "text", "text": nonce + a_text, "prompt_cache_breakpoint": {"mode": "explicit"}},
                   {"type": "text", "text": " Candidate: " + b_text}]
        body = {"model": MODEL, "temperature": 0, "messages": [
            {"role": "system", "content": SYS}, {"role": "user", "content": content}],
            "prompt_cache_options": {"mode": "explicit"}}
    else:  # implicit / nocache both use a plain string; nocache prepends a unique nonce
        body = {"model": MODEL, "temperature": 0, "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": nonce + a_text + " Candidate: " + b_text}]}
    req = urllib.request.Request(UP + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    u = json.load(urllib.request.urlopen(req, timeout=120)).get("usage", {})
    p = u.get("prompt_tokens", 0)
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return p, cached


def run(mode, order):
    pairs = [(i, j) for i in range(L) for j in range(R)]
    if order == "interleaved":
        pairs = [(i, j) for j in range(R) for i in range(L)]
    salt = f"[cell {order}/{mode} {time.time():.0f}] "  # unique per cell: no cross-phase cache reuse
    tot_p = tot_cached = 0
    for n, (i, j) in enumerate(pairs):
        nonce = salt + (f"[req {n}] " if mode == "nocache" else "")
        p, cached = call(LEFT[i], RIGHT[j], mode, nonce)
        tot_p += p
        tot_cached += cached
    uncached = tot_p - tot_cached
    # cost model: reads at 0.1x; the first touch of each prefix is an (implicit) write ~1.25x, but
    # cached_tokens already excludes writes, so uncached tokens are billed at 1x (writes fold in as
    # ordinary input on their first occurrence). Report the read discount as the measured effect.
    cost = (uncached / 1e6) * P_IN + (tot_cached / 1e6) * P_READ
    return tot_p, tot_cached, cost


def main():
    print(f"L={L} R={R} ({L*R} pairs), model={MODEL}")
    print(f"{'order':<12} {'mode':<9} {'prompt_tok':>11} {'cached_tok':>11} {'cached%':>8} {'input$':>9}")
    for order in ["contiguous", "interleaved"]:
        base = None
        for mode in ["nocache", "implicit", "explicit"]:
            p, cached, cost = run(mode, order)
            pct = 100.0 * cached / p if p else 0
            note = ""
            if base is None:
                base = cost
            else:
                note = f"  ({cost / base:.2f}x nocache$)"
            print(f"{order:<12} {mode:<9} {p:>11} {cached:>11} {pct:>7.1f}% {cost:>9.5f}{note}")


main()
