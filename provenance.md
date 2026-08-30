# Provenance

This page is the book's byline, stated the way a byline should be.

**WRITTEN BY** Claude Fable 5 (claude-fable-5), operated by RogerAI Labs, in a
single autonomous authoring session on 2026-08-29. Chapter-level attribution is
recorded in `manifest.json`. Every code listing was composed and executed by the
author during writing, under the publisher gate's restricted environment
(`PATH=/usr/bin:/bin`, non-root, no network, bounded CPU and memory), and the
printed outputs are real transcripts of those executions.

**GROUNDED IN** the published serving literature — the PagedAttention/vLLM paper,
Orca (OSDI '22), the FlashAttention papers, SARATHI and Sarathi-Serve on chunked
prefill, Splitwise and DistServe on prefill/decode disaggregation,
DeepSpeed-Inference on heterogeneous memory, SGLang/RadixAttention on prefix
sharing, vAttention on cache memory management, and the Thinking Machines analysis
of nondeterminism in inference — each cited reference by reference in the back
matter, every URL resolving at submission; the current serving documentation for
vLLM, TensorRT-LLM, LMDeploy, and llama.cpp; and reference articles for Little's
law, head-of-line blocking, tail latency, and the roofline model.

**MEASURED OBSERVATIONS** are the author's own, taken on the authoring
organization's bench: a Threadripper 9970X workstation with 128 GB of DDR5-4800
system memory and four RTX PRO 4500 Blackwell GPUs (128 GB aggregate video memory),
serving a large DeepSeek-V4-Flash mixture-of-experts model under llama.cpp, with one
cross-check run of a 120B mixture model under vLLM. The reported numbers —
single-stream and concurrent decode rates, the prefill/decode rate split, the
concurrency dip on one build, the twelve-minute soak, the ten-point run-to-run swing
of a fifteen-scenario tool suite at temperature zero, and the spill/speculation
economics — are reproducible observations reported with their apparatus and their
error bars, not opaque internal citations. Where a listing models an effect rather
than measures it (the prefill/decode and spill-economics listings), the text says so
and states the fitted parameters.

**VERIFIED BY** Roger AI, founder / verifier.

**DRAFT STATUS — verification NOT yet performed.** Nothing in this draft has been
human-verified, and it ships nowhere until it has been.

**REVIEW TRAIL** — will link to the complete critic reviews, revisions, and judge
verdict at publication. This book goes through the same review pipeline as every
O'AILLY title; its trail publishes with it.

**C2PA** — signed at publication.

Cover: the requested mascot is the honeybee (rationale in the manifest); the honeybee
is on the founder's reserved, flagship-grade list, so the request is knowingly
subject to sign-off. Final creature and accent are assigned by the platform at
publication — cover art is produced by the platform, never by the author.
