# The Batch and the Queue — outline

Serving many requests to one model at once

## Contents
- Chapter 1: One Model, Many Callers — Reframe serving as a system that multiplexes one model across concurrent requests, with the request as a first-class object, not a loop around inference.
- Chapter 2: The Batch — Explain continuous/dynamic batching: how tokens from different requests share one forward pass, and the throughput-vs-latency lever.
- Chapter 3: The KV Cache Is the Budget — Show that memory, not compute, sets how many requests fit; prefix sharing, eviction, paging.
- Chapter 4: Prefill and Decode — Separate the two regimes and their opposite bottlenecks (compute-bound vs bandwidth-bound) and why mixing them is hard.
- Chapter 5: The Queue — Admission control, scheduling, fairness, head-of-line blocking, tail latency; what a queue hides and reveals.
- Chapter 6: Determinism Under Load — Explain batch-packing nondeterminism at temp 0 and its consequences for evaluation.
- Chapter 7: Spill and the Slow Lane — What happens when model or KV won't fit; CPU/DDR offload economics; why one slow request stalls a batch.
- Chapter 8: A Serving Checklist — A reusable protocol for sizing, load-testing, and reading a server's real behavior under concurrency.

Every claim must resolve to a real, cited source. Boundaries in plain text.
