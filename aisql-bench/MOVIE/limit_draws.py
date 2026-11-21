#!/usr/bin/env python3
"""MOVIE quality of the systems that answer an unordered LIMIT with the first rows to arrive.

Four MOVIE queries end in a LIMIT with no ORDER BY (q1, q2: 5 rows; q5, q6: 10 review pairs). BlendSQL, PLOP and
Palimpzest return the first rows whose answers arrive, so which rows they return, and their score on those
queries, changes from one replay to the next even though every answer comes from the same recorded store. One
replay is therefore one draw. The README's MOVIE quality for these three systems is the mean over replays: the
full pass (<system>_movie_results.json) plus the replays of their order-dependent queries recorded next to it
(<system>_movie_results_draw<k>.json). Calls, cost and latency are the full pass's.

  python3 limit_draws.py            the table from the result files in this folder
  python3 limit_draws.py run 10     first replay those queries ten more times (tags _draw1 .. _draw10, a tag
                                    that has a results file is kept), then the table

`run` needs the stack replaying with the recorded latency (serve/start_stack.sh) and nothing else using the
proxy. BlendSQL runs under BLENDSQL_PYTHON (a python with blendsql; default: this interpreter).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# system -> (label, runner, the queries replayed)
SYSTEMS = {
    "blendsql": ("BlendSQL", "blendsql_movie.py", ["q1", "q2", "q5", "q6"]),
    "palimpzest": ("Palimpzest", "palimpzest_movie.py", ["q5", "q6"]),      # its q1 and q2 score 1.0 in every pass
    "plop": ("PLOP", "plop_movie.py", ["q1", "q2", "q5", "q6"]),
}


def per_query(path):
    return {q: r.get("quality") for q, r in json.load(open(path))["per_query"].items()}


def draws(system):
    out, k = [], 1
    while os.path.exists(os.path.join(HERE, f"{system}_movie_results_draw{k}.json")):
        out.append(per_query(os.path.join(HERE, f"{system}_movie_results_draw{k}.json")))
        k += 1
    return out


def run(n):
    for k in range(1, n + 1):
        for system, (_, runner, queries) in SYSTEMS.items():
            if os.path.exists(os.path.join(HERE, f"{system}_movie_results_draw{k}.json")):
                continue
            python = os.environ.get("BLENDSQL_PYTHON", sys.executable) if system == "blendsql" else sys.executable
            env = dict(os.environ, SWAN_TAG=f"_draw{k}")
            print(f"replay {k}: {system} {' '.join(queries)}", flush=True)
            subprocess.run([python, runner, *queries], cwd=HERE, env=env, check=True, stdout=subprocess.DEVNULL)


def report():
    print("| | full pass | replays | range | mean | queries that moved |")
    print("|---|---|---|---|---|---|")
    for system, (label, _, _) in SYSTEMS.items():
        full = per_query(os.path.join(HERE, f"{system}_movie_results.json"))
        macros, moved = [sum(full.values()) / len(full)], set()
        for d in draws(system):
            one = {q: d.get(q, v) for q, v in full.items()}      # a query that was not replayed keeps the full pass's score
            macros.append(sum(one.values()) / len(one))
            moved |= {q for q in d if d[q] != full[q]}
        print(f"| {label} | {macros[0]:.3f} | {len(macros)} | {min(macros):.3f}–{max(macros):.3f} | "
              f"**{sum(macros) / len(macros):.3f}** | {', '.join(sorted(moved, key=lambda q: int(q[1:]))) or '-'} |")
    print("\nreplays = the full pass plus the recorded replays of the order-dependent queries; mean = the README's quality.")


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"]:
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    report()
