# agent_bench Q1-Q30: SWAN vs PLOP-DP vs LOTUS (gpt-5.6-luna) -- 2026-09-26

Quality = deterministic scorer (eval_agentbench.py): LIMIT-free PLOP execution as ground truth; LIMIT queries scored sound+complete. Calls = true request counts through the recording proxy (SWAN replays PLOP's samples via the shared-verdict alias; LOTUS Q1-Q18 replayed from 2026-09-21, Q19-Q30 recorded fresh 2026-09-26). LOTUS translations are ours (sembench/README.md, provenance table). SWAN run: results/swan_gate2.

**Read the LOTUS quality column with care.** SWAN and PLOP consume the *same* recorded verdict for every
shared prompt (the proxy's shared-verdict alias), so they agree with the ground truth by construction
wherever their plans agree. LOTUS uses its own prompt templates, so its verdicts are independent samples of
the same model: on queries whose answer hinges on a handful of judgments (Q13/Q16/Q17/Q30 have an EMPTY
ground truth; LOTUS admitted 3 / 261 / 3,026 / 4 rows) any disagreement scores 0 even when LOTUS's answer
is a perfectly reasonable reading. Its macro therefore measures verdict agreement with PLOP's samples as
much as correctness; the calls / latency / cost columns are the like-for-like comparison.

| q | SWAN | PLOP | LOTUS | SWAN calls | PLOP calls | LOTUS calls | SWAN lat (s) | PLOP lat (s) | LOTUS lat (s) | SWAN $ | PLOP $ | LOTUS $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | 1.000 | 1.000 | 1.000 | 5 | 94 | 200 | 4.0 | 11.3 | 76.5 | 0.0004 | 0.0131 | 0.0179 |
| Q2 | 1.000 | 1.000 | 0.000 | 3 | 100 | 100 | 2.0 | 9.9 | 6.9 | 0.0003 | 0.0107 | 0.0064 |
| Q3 | 1.000 | 1.000 | 1.000 | 4 | 79 | 79 | 14.6 | 14.7 | 7.7 | 0.0001 | 0.0073 | 0.0120 |
| Q4 | 1.000 | 1.000 | 0.064 | 141 | 141 | 141 | 12.1 | 13.0 | 56.4 | 0.0206 | 0.0285 | 0.0251 |
| Q5 | 1.000 | 1.000 | 1.000 | 1 | 1 | 1 | 1.4 | 1.6 | 1.6 | 0.0001 | 0.0001 | 0.0001 |
| Q6 | 1.000 | 1.000 | 1.000 | 9 | 9 | 200 | 2.4 | 2.2 | 44.5 | 0.0021 | 0.0026 | 0.0449 |
| Q7 | 1.000 | 1.000 | 0.000 | 19 | 19 | 210 | 4.1 | 3.7 | 46.6 | 0.0023 | 0.0030 | 0.0462 |
| Q8 | 1.000 | 1.000 | 1.000 | 1 | 1 | 200 | 1.2 | 1.3 | 38.8 | 0.0003 | 0.0004 | 0.0455 |
| Q9 | 1.000 | 1.000 | 1.000 | 1 | 1 | 200 | 1.4 | 1.6 | 52.7 | 0.0002 | 0.0003 | 0.0462 |
| Q10 | 1.000 | 1.000 | 1.000 | 22 | 23 | 26 | 5.0 | 6.9 | 1.8 | 0.0008 | 0.0010 | 0.0012 |
| Q11 | 1.000 | 1.000 | 1.000 | 13 | 13 | 79 | 2.4 | 4.3 | 3.6 | 0.0004 | 0.0005 | 0.0033 |
| Q12 | 1.000 | 1.000 | 1.000 | 9 | 9 | 26 | 1.9 | 2.4 | 1.6 | 0.0003 | 0.0004 | 0.0012 |
| Q13 | 1.000 | 1.000 | 0.000 | 2 | 3 | 73 | 1.5 | 1.7 | 3.9 | 0.0001 | 0.0001 | 0.0034 |
| Q14 | 1.000 | 1.000 | 1.000 | 2 | 2 | 61 | 1.5 | 1.6 | 2.7 | 0.0001 | 0.0001 | 0.0026 |
| Q15 | 1.000 | 1.000 | 0.613 | 103 | 103 | 1,100 | 9.5 | 18.9 | 216.2 | 0.0027 | 0.0037 | 0.1274 |
| Q16 | 1.000 | 1.000 | 0.000 | 3 | 172 | 1,200 | 7.5 | 28.0 | 149.9 | 0.0001 | 0.0055 | 0.0724 |
| Q17 | 1.000 | 1.000 | 0.000 | 156 | 2,178 | 3,953 | 36.8 | 148.9 | 435.6 | 0.0088 | 0.1019 | 0.4646 |
| Q18 | 1.000 | 1.000 | 0.671 | 69 | 69 | 78 | 8.8 | 20.7 | 15.1 | 0.0018 | 0.0025 | 0.0076 |
| Q19 | 1.000 | 1.000 | 0.360 | 9,537 | 9,537 | 13,340 | 580.7 | 589.9 | 1355.5 | 0.2083 | 0.2922 | 1.4207 |
| Q20 | 1.000 | 1.000 | 1.000 | 3 | 3 | 100 | 1.4 | 1.5 | 12.8 | 0.0003 | 0.0003 | 0.0064 |
| Q21 | 1.000 | 1.000 | 1.000 | 100 | 100 | 100 | 7.1 | 6.5 | 5.3 | 0.0047 | 0.0061 | 0.0060 |
| Q22 | 1.000 | 1.000 | 1.000 | 10 | 10 | 100 | 2.5 | 3.8 | 11.8 | 0.0008 | 0.0011 | 0.0061 |
| Q23 | 1.000 | 1.000 | 0.814 | 100 | 100 | 100 | 8.6 | 7.2 | 5.5 | 0.0052 | 0.0069 | 0.0073 |
| Q24 | 1.000 | 1.000 | 0.998 | 62 | 62 | 62 | 4.9 | 5.2 | 12.9 | 0.0032 | 0.0043 | 0.0045 |
| Q25 | 1.000 | 1.000 | 0.613 | 178 | 143 | 1,164 | 15.0 | 24.0 | 257.5 | 0.0043 | 0.0049 | 0.1342 |
| Q26 | 1.000 | 1.000 | 0.094 | 391 | 389 | 1,429 | 35.1 | 45.4 | 276.7 | 0.0119 | 0.0160 | 0.1639 |
| Q27 | 1.000 | 1.000 | 0.288 | 172 | 172 | 1,095 | 25.4 | 47.2 | 118.7 | 0.0066 | 0.0095 | 0.1284 |
| Q28 | 1.000 | 1.000 | 0.833 | 42 | 42 | 273 | 6.0 | 21.7 | 60.0 | 0.0011 | 0.0015 | 0.0260 |
| Q29 | 1.000 | 1.000 | 0.250 | 7 | 20 | 36 | 1.7 | 17.5 | 2.5 | 0.0002 | 0.0003 | 0.0027 |
| Q30 | 1.000 | 1.000 | 0.000 | 7 | 7 | 12 | 4.0 | 17.1 | 15.3 | 0.0004 | 0.0006 | 0.0013 |
| **macro / Σ** | **1.000** | **1.000** | **0.620** | 11,172 | 13,602 | 25,738 | 811 | 1080 | 3297 | 0.288 | 0.525 | 2.835 |
