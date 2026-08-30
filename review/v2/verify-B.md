<!-- CRITIC B · mimo-v2.5-free · family:xiaomi · pass 3 · 2026-08-30T05:16:01Z -->
CRITIC: mimo-v2.5-free (family xiaomi, actor mimo-v2.5-free@opencode-zen)
DATE: 2026-08-30
PASS: 3
AUTO-TALLIED VERDICT: PUBLISH

---

# Critic review — rogerai-labs--the-batch-and-the-queue v2

```
CRITIC:    mimo-v2.5-free (family xiaomi, operator opencode-zen)
DATE:      2026-08-30
PASS:      3 (verification)
READ:      full manuscript v2 | delta (ch02, ch03, ch05, ch07)
```

## Verdict summary

All four pass-2 blocking findings are resolved in v2 with correct, targeted diffs. A-3 re-cites the 120B measurement as the author's own with apparatus (consistent with provenance and every other bench number). A-2 deep-copies job dicts per simulation run. A-4 qualifies external vs. internal fragmentation to match Kwon et al. B-1 refits acceptance rates so the output demonstrates what the prose claims. No new blocking findings emerged. Listings execute correctly; transcripts match code. The manuscript is internally consistent, factually grounded, and ready.

**PUBLISH**

## Blocking findings

| # | Location (file:section) | Claim / problem | Evidence | Severity (high/med) |
|---|---|---|---|---|
| (none) | — | — | — | — |

## Suggestions (non-blocking)

1. The provenance page says "one cross-check run of a 120B mixture model under vLLM" (singular), while ch02 now describes a multi-concurrency measurement series (single-stream through 16 callers). Minor framing inconsistency — "one cross-check run" could become "one cross-check configuration" to match ch02's richer description.
2. The ch07 speculative listing header should be marked `no-run` per the frontmatter convention (it is a fitted model, not a re-executed benchmark). The output currently matches the code but is illustrative, not empirical.

## Fact-check sample

| Claim (quoted) | Location | Cited source | Supported? (yes/no/partly) |
|---|---|---|---|
| "Those are my own measurements on the workstation described in Chapter 1, not a published benchmark" — 120B mixture, 60→480+ tok/s, four RTX PRO 4500 Blackwell GPUs, 128 GB VRAM | ch02 § throughput-latency lever | provenance.md (author's own apparatus declaration) | yes — provenance states "one cross-check run of a 120B mixture model under vLLM" on "four RTX PRO 4500 Blackwell GPUs (128 GB aggregate video memory)"; ch02 text matches. |
| "paging eliminates external fragmentation ... but leaves internal fragmentation ... bounded by a single block" | ch03 § Fragmentation | [R1] Kwon et al., arXiv:2309.06180 | yes — R1 §4.2: "eliminates external fragmentation as all blocks have the same size"; §7.2/Fig.2: waste bounded within one block. |
| "the culprit is ... the lack of batch-invariance in the kernels ... a request's output depends on what it was batched with" | ch06 § batch decides reduction order | [R5] Thinking Machines Lab, 2025 | yes — R5 blog: demonstrates batch-size-dependent reduction order; defines batch invariance as the fix. |
| "Drafting a single token returns a clean 1.29× speedup, drafting two collapses that to a marginal 1.04×, and drafting three turns the technique against itself, landing at 0.76×" | ch07 § speculative listing | fitted model in-listing (no external citation) | yes — model parameters: k=0.55, accept=(1.00, 0.70, 0.55). Code output: 1.29x, 1.04x, 0.76x. Matches text and conclusion. |

## Scores (1–5)

accuracy: 5 · clarity: 5 · completeness-for-tier: 5 · density: 5 · originality: 4

## Pass-3 only: findings ledger

| Finding # (from Pass 2) | Status: resolved / rebutted-accepted / still-open | Note |
|---|---|---|
| A-3 (HIGH, ch02: 120B mis-cited to [R22]) | resolved | [R22] removed from sentence; now author's own measurement with apparatus. Consistent with provenance. |
| A-2 (med, ch05: shallow-copy job dicts) | resolved | `pending = [dict(j) for j in jobs]` isolates runs. Output unchanged and correct. |
| A-4 (med, ch03: external frag overstated) | resolved | External eliminated, internal bounded by one block. Block-size trade added. Matches Kwon et al. |
| B-1 (med, ch07: speculative output contradicts conclusion) | resolved | Acceptance refit to (1.00, 0.70, 0.55); n_max=1 at 1.29× clearly wins. Text explains MTP head collapse. |
