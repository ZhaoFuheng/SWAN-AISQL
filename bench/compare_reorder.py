#!/usr/bin/env python3
"""Per-query OFF vs ON for the govreport reorder bench, with the recorded baseline ON beside a new run's ON.
  python3 compare_reorder.py <baseline.json> <new.json>"""
import json, sys
def load(path):                      # bench.py writes {"docs","concurrency","results"}; older snapshots are a bare list
    d = json.load(open(path)); return d["results"] if isinstance(d, dict) else d
base = {r["name"]: r for r in load(sys.argv[1])}
new  = {r["name"]: r for r in load(sys.argv[2])} if len(sys.argv) > 2 else {}
print(f"{'query':16}{'OFF calls':>10}{'base ON':>9}{'saving':>8} |{'new ON':>8}{'saving':>8}{'match':>7} |{'base ON lat':>12}{'new ON lat':>11}")
tot = dict(off=0, bon=0, non=0, bl=0.0, nl=0.0, n=0)
for name, r in base.items():
    off, bon = r["off"]["chat_calls"], r["on"]["chat_calls"]
    nr = new.get(name); non = nr["on"]["chat_calls"] if nr else None
    match = "" if not nr else ("ok" if nr["on"]["checksum"] == nr["off"]["checksum"] else "DIFF")
    print(f"{name:16}{off:>10}{bon:>9}{(off-bon)/off*100:>7.1f}% |{str(non) if non is not None else '-':>8}"
          f"{((off-non)/off*100 if non is not None else 0):>7.1f}%{match:>7} |{r['on']['latency_s']:>11.0f}s{(nr['on']['latency_s'] if nr else 0):>10.0f}s")
    tot["off"] += off; tot["bon"] += bon; tot["non"] += non or 0; tot["bl"] += r["on"]["latency_s"]; tot["nl"] += (nr["on"]["latency_s"] if nr else 0); tot["n"] += 1 if nr else 0
print("-"*100)
print(f"{'TOTAL':16}{tot['off']:>10}{tot['bon']:>9}{(tot['off']-tot['bon'])/tot['off']*100:>7.1f}% |{tot['non']:>8}"
      f"{((tot['off']-tot['non'])/tot['off']*100 if tot['n'] else 0):>7.1f}%{'':>7} |{tot['bl']:>11.0f}s{tot['nl']:>10.0f}s")
