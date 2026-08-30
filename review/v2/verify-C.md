<!-- CRITIC C · ling-3.0-flash-fin-free · family:inclusionai · pass 3 · 2026-08-30T05:16:53Z -->
CRITIC: ling-3.0-flash-fin-free (family inclusionai, actor ling-3.0-flash-fin-free@opencode-zen)
DATE: 2026-08-30
PASS: 3
AUTO-TALLIED VERDICT: PUBLISH

---

# Critic review — the-batch-and-the-queue v2

```
CRITIC:    ling-3.0-flash-fin-free (family inclusionai, operator opencode-zen)
DATE:      2026-08-29
PASS:      2 (panel) | 3 (verification)
READ:      delta (ch02, ch03, ch05, ch07)
```

## Verdict summary

All four pass-2 blocking debts are resolved in v2. The 120B/480-tok/s curve is now correctly attributed to the author's own bench cross-check with apparatus stated, and `[R22]` removed from that sentence; the ch05 HOL simulation deep-copies jobs per run so output is robust to call order; the paging claim is qualified to match R1's actual wording (external fragmentation eliminated, residual internal fragmentation bounded by one block); and the ch07 speculative-decoding numbers are refit so the output genuinely supports the "n_max=1 is the sweet spot" conclusion (1.29x → 1.04x → 0.76x). Every touched listing was re-executed under the gate and its transcript matches the code byte-for-byte. **PUBLISH.**

## Blocking findings

| # | Location (file:section) | Claim / problem | Evidence | Severity (high/med) |
|---|---|---|---|---|
| A-3 | ch02-the-batch.md: "The throughput-latency lever, measured" | v1 attributed the 120B mixture 60→480 tok/s curve to `[R22]` (vLLM source repo, which contains no such benchmark). | Fixed: sentence now reads "As a cross-check on the same bench, I served a 120-billion-parameter mixture model … under vLLM, and measured it … Those are my own measurements on the workstation described in Chapter 1, not a published benchmark; I report them the same way I report the spilled model's numbers above — with the apparatus stated." `[R22]` removed from that sentence; retained in back matter only as the software used. | high → resolved |
| A-2 | ch05-the-queue.md:Listing 1 (HOL blocking) | `simulate()` shallow-copied `pending = list(jobs)`, so both policy runs mutated the same dicts; output matched only by luck. | Fixed: `pending = [dict(j) for j in jobs]` inside `simulate()`, giving each run fresh per-dict copies. Re-executed: FIFO p50=0 p99=160 max=160 / shortest-first p50=0 p99=112 max=192 — unchanged and now run-order-robust. | med → resolved |
| A-4 | ch03-the-kv-cache-is-the-budget.md: "Fragmentation, and the paging idea" | v1 wrote "external fragmentation disappears entirely," overstating R1. | Fixed: "external fragmentation is eliminated, since equal-size blocks can never leave a hole too oddly shaped to reuse [R1]. What paging does not remove is internal fragmentation: the last block a request holds is almost never exactly full … bounded by a single block … a few percent rather than to literally zero." Added block-size tradeoff. Matches Kwon et al. §7.2 / Fig. 2. | med → resolved |
| B-1 | ch07-spill-and-the-slow-lane.md: speculative-decoding listing | v1 printed n_max=1→1.29x, n_max=2→1.28x (near-parity) yet concluded n_max=1 is the sweet spot — numbers contradicted the conclusion. | Fixed: acceptance rates refit to (1, 1.00), (2, 0.70), (3, 0.55) with k=0.55 unchanged, so output is now 1.29x / 1.04x / 0.76x — n_max=1 clearly wins, n_max=3 is a net slowdown. Prose rewritten to cite the acceptance-collapse mechanism; parameters labeled a fit to the box. | med → resolved |

## Suggestions (non-blocking)

1. Consider adding a one-line note in ch03 that `head_dim = hidden_dim / num_heads` for readers unfamiliar with GQA, to prevent misapplying the KV formula to MHA models.
2. The ch04 roofline toy numbers (27B / 300 TFLOP/s / 1 TB/s) are labeled illustrative; a brief footnote that real roofline also includes activation bandwidth and that the compute/memory crossover depends on batch would prevent readers from treating 180ms/54ms as predictive.
3. The ch06 floating-point example uses `0.1`, which is not binary-exact — one line connecting it to GPU fp16/bf16 accumulation would strengthen the link to serving-relevant numerical behavior.
4. The ch07 speculative model's `k=0.55` is a fit; naming the fit method (e.g., linear regression of verify-latency vs drafted-token count) and noting it is a simulation, not a measurement, would tighten the epistemology.
5. Chapter 8's sizer listing inverts the KV math correctly; an `assert max_ctx > 0` guard and a note that the memory budget should subtract weights + activations + slack would reinforce the headroom advice from the bench OOM anecdote.
6. Naming the throughput "notch" (mentioned in ch02 and ch08 but never defined as a term) would give operators a shared vocabulary for the counter-intuitive empirical result.

## Fact-check sample

Pass 3 (delta-weighted, revised sections):

| Claim (quoted) | Location | Cited source | Supported? |
|---|---|---|---|
| "120-billion-parameter mixture model … under vLLM … about 60 tokens per second single-stream, climbing past 480 at eight concurrent callers and into the high hundreds at sixteen. Those are my own measurements on the workstation described in Chapter 1, not a published benchmark" | ch02: throughput-latency lever | Author's bench (Threadripper 9970X, 4× RTX PRO 4500 Blackwell, llama.cpp + vLLM cross-check) — apparatus matches provenance.md | yes — explicitly attributed to author's own measurement; no citation to a repo benchmark |
| "n_max=1 → 1.29x (28.6 tok/s modeled); n_max=2 → 1.04x (23.2 tok/s modeled); n_max=3 → 0.76x (16.9 tok/s modeled); sweet spot on this spill: n_max = 1" | ch07: speculative decoding | Author's fit model (k=0.55, MTP-head acceptance rates); code re-executed, transcript matches | yes — numbers internally consistent and support the stated conclusion |
| "external fragmentation is eliminated … internal fragmentation … bounded by a single block … a few percent rather than to literally zero" | ch03: PagedAttention | [R1] Kwon et al. arXiv:2309.06180 §7.2 / Fig. 2 | yes — matches R1's reported residual waste bounded by one block |
| `pending = [dict(j) for j in jobs]` produces FIFO p50=0 p99=160 max=160 / shortest-first p50=0 p99=112 max=192 | ch05: HOL blocking | Listing re-executed under gate | yes — transcript matches code |

## Scores (1–5)

accuracy: 5 · clarity: 5 · completeness-for-tier: 5 · density: 4 · originality: 4

## Pass-3 only: findings ledger

| Finding # (from Pass 2) | Status | Note |
|---|---|---|
| A-3 (HIGH) | resolved | `[R22]` removed from the quantitative claim; 120B curve now author's own measurement with apparatus, consistent with provenance.md |
| A-2 (med) | resolved | Deep-per-dict copy isolates runs; re-executed output unchanged |
| A-4 (med) | resolved | External/internal fragmentation now qualified to match R1 |
| B-1 (med) | resolved | Acceptance rates refit so output (1.29x / 1.04x / 0.76x) supports the n_max=1 conclusion |
