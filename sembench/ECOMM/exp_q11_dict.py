#!/usr/bin/env python3
"""A/B for ECOMM q11: pass the FULL productDescriptors dictionary (as LOTUS does) instead of only
description.value. Single variable: only the `pre` CTE's data source changes, so all 7 prompt sites
(4 filters + 3 same-brand joins) keep their exact wording. Reuses swan_ecomm's run_one (execution +
scoring) and writes to exp_q11_dict_results.json -- never to the certified swan_ecomm_results.json.
  run (from sembench/ECOMM):  python3 exp_q11_dict.py [--check]
"""
import json, os, sys
from datetime import datetime, timezone
import swan_ecomm as S

OLD = "coalesce(s.description,'') AS description, s.price, i.filepath\n     FROM styles_details s JOIN images i ON i.id = s.id)"
NEW = ("coalesce(p.productDescriptors::VARCHAR,'') AS description, s.price, i.filepath\n"
       "     FROM styles_details s JOIN images i ON i.id = s.id\n"
       "     JOIN read_parquet('data/sf_500/styles_details.parquet') p ON p.id = s.id)")
assert S.SQL["q11"].count(OLD) == 1, "q11 pre-CTE text changed; update the patch"
S.SQL["q11"] = S.SQL["q11"].replace(OLD, NEW)

if "--check" in sys.argv:          # no LLM calls: validate the patched CTE only
    cte = S.SQL["q11"].split("footwear AS")[0].rstrip().rstrip(",").replace("WITH pre AS (", "", 1)[:-1]
    print(cte)
    sys.exit(0)

stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
rec, raw, line = S.run_one("q11", stamp)
rec["variant"] = "full productDescriptors dict"
out = os.path.join(S.HERE, "exp_q11_dict_results.json")
json.dump({"q11_dict": rec, "raw": raw}, open(out, "w"), indent=1)
print(line); print("logged ->", out)
