# Back Matter

## Glossary

- Admission control: the policy governing which waiting requests a server lets into service and which it refuses or defers under load.
- Aggregate throughput: total tokens produced per second across all concurrent callers, the number that sets cost per token.
- Arithmetic intensity: the amount of computation performed per byte read from memory; low intensity is memory-bound, high intensity is compute-bound.
- Backpressure: a queue signaling upstream to slow or stop sending work when it is falling behind, instead of accepting an unbounded backlog.
- Batch: the set of requests advanced together in one forward pass so they share a single read of the model's weights.
- Batch-invariance: the property of a kernel that gives bit-identical results regardless of batch size or shape, the basis for reproducible temperature-zero output.
- Compute-bound: limited by how fast the hardware can do arithmetic rather than how fast it can read memory; the prefill regime.
- Continuous batching: admitting and releasing requests on every token step rather than in fixed groups, keeping the batch full as requests finish; also called dynamic or in-flight batching.
- Chunked prefill: splitting a long prompt into bounded pieces processed across several forward passes to avoid stalling ongoing decodes.
- Decode: the phase that generates output tokens one at a time; memory-bandwidth-bound.
- Disaggregation: running prefill and decode on separate hardware pools so neither interferes with the other.
- Eviction: reclaiming an in-flight request's cache under memory pressure, later recovered by recomputation or by swapping.
- Goodput: the throughput that actually met the latency objective, as distinct from raw tokens produced.
- Grouped-query attention (GQA): an attention design in which many query heads share a smaller set of key-value heads, shrinking the KV cache.
- Head-of-line blocking: a slow item at the front of a line delaying faster items behind it that could otherwise be served.
- KV cache: the stored per-request keys and values for every token seen so far, which make decode linear instead of quadratic and which dominate the memory budget.
- Little's law: the average number of items in a system equals arrival rate times average time in system (L = lambda x W).
- Memory-bandwidth-bound: limited by how fast the hardware can read memory rather than compute; the decode regime.
- Paging (PagedAttention): storing the KV cache in fixed-size blocks scattered across memory rather than contiguous reservations, eliminating fragmentation.
- Prefill: the phase that processes a prompt and builds its KV cache; compute-bound.
- Prefix sharing: reusing the cached keys and values of a shared prompt prefix across the requests that share it.
- Preemption: pausing an admitted request to free resources, resuming it later via recomputation or swapping.
- Roofline model: a framing of performance as bounded by either compute or memory bandwidth, depending on arithmetic intensity.
- Selective batching: batching the weight-shared parts of a forward pass while handling per-request attention separately.
- Speculative decoding: guessing several tokens with a cheap process and verifying them in one pass of the full model.
- Spill: pushing weights or cache that do not fit in fast memory down to a slower tier (system memory or disk).
- Static batching: processing a fixed group of requests together and releasing them together, wasting slots as members finish early.
- Tail latency: the high-percentile latency (p99, max) that dominates user experience, as opposed to the average or median.
- Throughput-latency trade: the tension whereby larger batches raise aggregate throughput while worsening each request's latency.
- Time per output token: the pace of generation after the first token; a decode-regime measurement.
- Time to first token: the delay from submitting a prompt to the first output token; a prefill-plus-queue measurement.
- Token budget: a per-step cap on total tokens processed, used to keep large prefills from stalling decodes.

## References

- [R1] Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM), 2023 — https://arxiv.org/abs/2309.06180
- [R2] Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models," USENIX OSDI 2022 — https://www.usenix.org/conference/osdi22/presentation/yu
- [R3] Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," 2022 — https://arxiv.org/abs/2205.14135
- [R4] Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," 2023 — https://arxiv.org/abs/2307.08691
- [R5] Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference," 2025 — https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- [R6] Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills," 2023 — https://arxiv.org/abs/2308.16369
- [R7] Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve," 2024 — https://arxiv.org/abs/2403.02310
- [R8] Patel et al., "Splitwise: Efficient generative LLM inference using phase splitting," 2023 — https://arxiv.org/abs/2311.18677
- [R9] Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving," 2024 — https://arxiv.org/abs/2401.09670
- [R10] Aminabadi et al., "DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale," 2022 — https://arxiv.org/abs/2207.00032
- [R11] Prabhu et al., "vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention," 2024 — https://arxiv.org/abs/2405.04437
- [R12] Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" (RadixAttention), 2023 — https://arxiv.org/abs/2312.07104
- [R13] vLLM documentation (latest) — https://docs.vllm.ai/en/latest/
- [R14] vLLM documentation (stable) — https://docs.vllm.ai/en/stable/
- [R15] NVIDIA TensorRT-LLM documentation — https://nvidia.github.io/TensorRT-LLM/
- [R16] LMDeploy documentation — https://lmdeploy.readthedocs.io/en/latest/
- [R17] llama.cpp source repository — https://github.com/ggml-org/llama.cpp
- [R18] "Little's law" — https://en.wikipedia.org/wiki/Little%27s_law
- [R19] "Head-of-line blocking" — https://en.wikipedia.org/wiki/Head-of-line_blocking
- [R20] "Tail latency" — https://en.wikipedia.org/wiki/Tail_latency
- [R21] "Roofline model" — https://en.wikipedia.org/wiki/Roofline_model
- [R22] vLLM source repository — https://github.com/vllm-project/vllm
- [R23] SGLang source repository — https://github.com/sgl-project/sglang
