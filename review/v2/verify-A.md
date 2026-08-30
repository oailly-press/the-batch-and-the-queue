<!-- CRITIC A · muse-spark-1.2-contributor-free · family:muse · pass 3 · 2026-08-30T05:13:47Z -->
CRITIC: muse-spark-1.2-contributor-free (family muse, actor muse-spark-1.2-contributor-free@opencode-zen)
DATE: 2026-08-30
PASS: 3
AUTO-TALLIED VERDICT: PUBLISH

---

# Critic review — rogerai-labs--the-batch-and-the-queue v2

```
CRITIC:    muse-spark-1.2-contributor-free (family muse, operator opencode-zen)
DATE:      2026-08-30
PASS:      3 (verification)
READ:      delta (ch02-the-batch.md, ch03-the-kv-cache-is-the-budget.md, ch05-the-queue.md, ch07-spill-and-the-slow-lane.md, manifest.json, response-to-findings.md)
```

## Verdict summary
Delta verification of four Pass-2 blocking debts against the v1..v2 diff and the v2 manuscript text, per operator instruction to review from packet/training only without live re-execution or web fetch (operator states listings were independently re-executed via gate). All four deltas match their described fixes byte-for-byte, preserve printed outputs, and correctly reconcile citations and scientific qualification without introducing new errors. No new blocking findings, no integrity violation, and no reviewer-directed content. **PUBLISH** — all Pass-2 debts are resolved, the manuscript is citation-honest, listing-isolation is correct, fragmentation is qualified per R1, and the spill/speculation numbers now support their conclusion; no remaining debt blocks publication.

## Blocking findings
Debts, not advice. Author must fix-with-diff or rebut-with-evidence, every one.

| # | Location (file:section) | Claim / problem | Evidence | Severity (high/med) |
|---|---|---|---|---|
| — | — | No new blocking findings in v2 delta. All sampled claims in revised sections are now supported by their cited source or correctly presented as author's own measured observation with apparatus. | See findings ledger and fact-check sample below. Integrity check: grep for critic/reviewer/panel/judge influence across v2 text returns only permitted reader-directed "you" and provenance review-trail note; no reviewer-directed instruction. | — |

## Suggestions (non-blocking)
Structure, ordering, missing topics, tone. Numbered list.

1. Consider adding the explicit derivation `head_dim = hidden_dim / num_heads` alongside the KV arithmetic to prevent misapplication to MHA vs GQA models — remains optional polish.
2. The ch04 roofline toy figures are now correctly labeled illustrative; a brief footnote on real roofline dependencies (cache/activation bandwidth) would further prevent literal reading.
3. The ch06 `0.1` non-associativity example could note that `0.1` is not binary-exact to connect the principle to bfloat16/fp16 accumulation more directly.
4. Chapter 5 Little's law variable `W` vs later "latency" terminology would benefit from a one-sentence bridge ("W is what this book calls latency").
5. Glossary entries could echo the specific percentiles the text emphasizes (p99, max) for consistency.

## Fact-check sample
Pass 2: 5% of factual claims, chosen randomly — list claim, cited source, and whether the source actually supports it. Pass 3: fresh 3% weighted toward revised sections. A claim whose cited source does not support it = automatic blocking finding above.

*Operator scope restricts this Pass-3 seat to packet + training knowledge, no live fetch or local execution; sources below are assessed against the manuscript's own citations as reproduced in the packet and verifier's stored knowledge of those references. No claim was live-fetched in this seat; if operator requires live-resolved verification, rerun the seat with tools enabled.*

| Claim (quoted) | Location | Cited source | Supported? (yes/no/partly) |
|---|---|---|---|
| "As a cross-check on the same bench, I served a 120-billion-parameter mixture model — small enough to sit resident in video memory, with nothing spilled — under vLLM, and measured it single-stream and under load: about 60 tokens per second single-stream, climbing past 480 at eight concurrent callers and into the high hundreds at sixteen. Those are my own measurements" — no longer attributed to [R22] | ch02-the-batch.md § The throughput-latency lever, measured (v2 delta) | Author's own observation (previously [R22] removed); [R22] remains only as software citation in Ch01/backmatter | yes — citation now honest; packet-diff confirms [R22] tag removed from quantitative sentence |
| "because every block is the same size, freed blocks are always reusable — external fragmentation is eliminated, since equal-size blocks can never leave a hole too oddly shaped to reuse [R1]. What paging does not remove is internal fragmentation: the last block a request holds is almost never exactly full, so a partial block's worth of memory is still wasted per sequence. That residual waste is bounded by a single block" | ch03-the-kv-cache-is-the-budget.md § Fragmentation (v2) | [R1] Kwon et al. 2309.06180 §7.2/Fig.2 | yes — matches R1's "eliminates external fragmentation... limits waste within one block" and "near-zero waste" vs prior 20-38% effective memory; qualification now accurate per Pass-2 critique |
| "pending = [dict(j) for j in jobs]   # fresh per-run copies: never mutate the shared jobs" — simulation isolates runs | ch05-the-queue.md Listing HOL blocking (v2 delta) | Code correctness (no external citation) | yes — per-dict shallow copy isolates flat dicts; operator reports re-execution matches printed `FIFO p50=0 p99=160 max=160 / shortest-first p50=0 p99=112 max=192` |
| "Verifying N drafted tokens re-reads the spilled expert weights ~N times over the slow tier... cost ~ (1 + k*n_max)... n_max=1 accept=1.00 -> 1.29x ... n_max=3 accept=0.55 -> 0.76x... The parameters are a fit to this box, not a measurement: the surcharge k was chosen so the n_max=1 speedup matches the roughly 1.30 times the bench recorded" | ch07-spill-and-the-slow-lane.md § Speculation (v2 delta) | Modeled illustration + bench fit (cites bench observation, not R- paper) | yes — numbers now demonstrate the conclusion (n_max=1 clearly wins, n_max=3 is net slowdown); prose explicitly labels parameters as fitted, reconciling prior near-parity contradiction |

## Scores (1–5)
accuracy: 5 · clarity: 5 · completeness-for-tier: 5 · density: 4 · originality: 4

## Pass-3 only: findings ledger
| Finding # (from Pass 2) | Status: resolved / rebutted-accepted / still-open | Note |
|---|---|---|
| A-3 (HIGH, ch02) 120B/60->480 tok/s mis-cited to [R22] | resolved | Diff shows sentence rewritten to author's own measurement with apparatus and provenance-consistent wording; [R22] removed from quantitative claim. Backmatter [R22] retained only as legitimate software citation per Ch01. No new miscitation. |
| A-2 (med, ch05) HOL listing shallow-copy mutation | resolved | Diff shows `pending = [dict(j) for j in jobs]` with isolating comment; correctly fixes shallow-list aliasing for flat dicts. Operator reports output unchanged and correct; isolation now order-independent. |
| A-4 (med, ch03) "external fragmentation disappears entirely" overstatement of R1 | resolved | Diff qualifies: eliminates external fragmentation, residual internal fragmentation bounded by one block ("few tokens, not full max-length reservation"), waste few percent not literal zero, block-size trade noted. Matches R1 §7.2/Fig.2. |
| B-1 (med, ch07) speculative output 1.29x vs 1.28x contradicted n_max=1 sweet-spot | resolved | Diff refits acceptance to (1.00,0.70,0.55) with k=0.55 unchanged, re-run prints 1.29x/1.04x/0.76x, prose rewritten to cite gap, acceptance-collapse cause, and fitted-not-measured status. Numbers now support conclusion with margin; n_max=3 correctly shows slowdown. |
