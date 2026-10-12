# Round 7 partial output — superseded, retained for provenance

These artifacts come from the Round 7 run, which was both **truncated** (2,000-word
cap, since retired) and **incomplete** (15 of 59 filing rows scored, every one of them
large-cap, because the run stalled on Groq's 200k-token/day free tier).

They are not results. `summary.txt` here still names `llama-3.1-8b-instant` in its
limitations block, reports "Mid-cap (n=0)", and carries the retired half-median
`extraction_short` rule.

`intermediate_results.json` is tagged `__cache_version__ = "v4"`. The current expected
key is `v5-c1200-penaigptoss20b`, so these 7 cached large-cap firms would be ignored by
a current run regardless; they are archived to remove the ambiguity entirely.

The one number from this run that was reported to the advisor is the length-vs-
disagreement correlation (numeric x complete section length, rho = +0.524, p = 0.045,
n = 15, one tier), and it was reported explicitly as continuity rather than evidence.

Nothing in this directory may be cited in the paper.
