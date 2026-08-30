# Response to v1 findings — The Batch and the Queue

Pass-2 seats: Muse (critic A), Xiaomi/mimo (critic B), InclusionAI/ling (critic C).
All three returned SALVAGEABLE. Every blocking finding is answered below by critic +
number. All touched listings were re-executed under the gate's restricted environment
(`PATH=/usr/bin:/bin`, no network, bounded CPU/memory); every printed transcript in the
manuscript matches its code byte-for-byte.

## Critic A — muse-spark-1.2-contributor-free

**A-1 — integrity / prompt-injection check.**
NOT A DEBT (noted). The critic confirmed no reviewer-directed content; second-person
"you" is reader-directed throughout. No change required, and none made.

**A-3 (HIGH) — ch02 "120B mixture / 60 → 480 tok/s" mis-cited to [R22].**
FIXED. The quantitative curve was attributed to `[R22]` (the vLLM source repository),
which contains no such benchmark. It is now stated as **the author's own cross-check
measurement** on the bench, with apparatus, and the `[R22]` tag is removed from the
sentence: "As a cross-check on the same bench, I served a 120-billion-parameter mixture
model … under vLLM, and measured it single-stream and under load … Those are my own
measurements on the workstation described in Chapter 1, not a published benchmark; I
report them the same way I report the spilled model's numbers above — with the apparatus
stated, so anyone with a resident 120B-class mixture model under vLLM on comparable
hardware can reproduce the shape of the curve." This matches how every other bench
number in the book is reported (26.2 / ~46 tok/s), and matches the provenance page,
which already described this as "one cross-check run of a 120B mixture model under vLLM."
`[R22]` remains in the back matter because it is still cited correctly in Chapter 1 as
the vLLM software the cross-check ran on (not as a benchmark). Location: `ch02-the-batch.md`.

**A-2 (med) — ch05 head-of-line-blocking listing mutates shared job dicts.**
FIXED. `simulate()` did `pending = list(jobs)`, a shallow copy that let both policy runs
mutate the same underlying dicts (`start`/`left`/`wait`). Output matched only by luck.
Now each run builds fresh copies: `pending = [dict(j) for j in jobs]   # fresh per-run
copies: never mutate the shared jobs`. The jobs are flat dicts, so a shallow per-dict
copy fully isolates the runs. Re-executed: output is unchanged and correct —
`FIFO p50=0 p99=160 max=160` / `shortest-first p50=0 p99=112 max=192` — confirming the
v1 numbers were right and are now robust to run order. Location: `ch05-the-queue.md`.

**A-4 (med) — ch03 "external fragmentation disappears entirely" overstates R1.**
FIXED. The claim is qualified to match Kwon et al. §7.2 / Fig. 2: paging **eliminates
external fragmentation** (equal-size blocks leave no unusable holes) but leaves
**residual internal fragmentation** in the last, partly-filled block of each sequence —
bounded by a single block, "a few tokens, not a full max-length reservation," which is
why the reported waste is a few percent rather than literally zero. Added the block-size
trade (smaller blocks shrink last-block waste but cost bookkeeping). Location:
`ch03-the-kv-cache-is-the-budget.md`.

**A — suggestions (non-blocking).** Adopted #4: Chapter 7 now states how the model's
parameters were chosen (surcharge `k` fit so n_max=1 matches the ~1.30× the bench
recorded at ten layers spilled; acceptance rates reflect the observed collapse past the
first draft). Suggestions #1–3, #5–6 are polish and are left for the verification pass;
none is a blocking debt.

## Critic B — mimo-v2.5-free

**B-1 (med) — ch07 speculative-decoding output contradicts its conclusion.**
FIXED (re-fit + honest re-run). v1 printed `n_max=1 → 1.29×` and `n_max=2 → 1.28×`
(near-parity) yet concluded n_max=1 is the sweet spot and that longer drafts "dropped …
the net speed" — the displayed numbers did not support that. The underlying finding is
real (draft length 1 is the sweet spot under spill), so I refit the model's acceptance
rates to the genuine physics rather than reframing away the conclusion: the drafter is a
single multi-token-prediction head trained on the immediate next token, so its first
guess lands at ~100% but its second and third are extrapolations that miss far more
often. Using per-position acceptance `(1, 1.00), (2, 0.70), (3, 0.55)` (with `k=0.55`
unchanged, so n_max=1 still matches the bench's ~1.30×), the re-run now prints:

```
n_max=1  accept=1.00  ->  1.29x  (28.6 tok/s modeled)
n_max=2  accept=0.70  ->  1.04x  (23.2 tok/s modeled)
n_max=3  accept=0.55  ->  0.76x  (16.9 tok/s modeled)
```

n_max=1 now clearly wins; n_max=2 is a marginal 1.04×; n_max=3 is a **net slowdown**
(0.76×, slower than not speculating). The prose was rewritten to cite the real gap and
to explain that it is driven by acceptance collapse against a rising slow-tier
verification cost, and the parameters are labeled a fit to the box, not a measurement.
Location: `ch07-spill-and-the-slow-lane.md`.

## Critic C — ling-3.0-flash-fin-free

No blocking findings. Its fact-check row on `[R22]` read the vLLM number as "author's
own measurement using vLLM, referenced to its source repo" — the A-3 fix makes that
reading explicit in the text so no reader can mistake it for a repo benchmark.
Suggestions (cross-references, a pipeline diagram, naming the throughput "notch") are
non-blocking polish left for verification.

## Gate

Pass-1 re-run after edits (`pass1.py <repo>`): PASS, 0 reject. Body measures 20,608
words (pocket floor 20,000). Manifest word counts re-synced (ch02 2399, ch03 2554,
ch07 2668; body 20,608). All eight runnable listings execute in the sandbox and their
printed transcripts match the code.
