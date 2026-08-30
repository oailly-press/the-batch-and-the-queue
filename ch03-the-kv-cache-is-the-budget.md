# Chapter 3 — The KV Cache Is the Budget

Ask an operator how many requests their server can handle at once and you will
usually get an answer about compute — the number of GPUs, the teraflops, the batch
width. That answer is almost always wrong, or at least secondary. The real limit on
concurrency is memory, and specifically the memory consumed by the key-value cache.
"How many at once" is a memory question wearing a compute costume, and until you see
it that way you will keep being surprised by servers that refuse new work while
their expensive math units sit idle.

## What the cache is and why it must exist

When a model processes a token, attention computes, for that token, a set of keys
and values derived from every token before it. The naive way to generate a sequence
would recompute all of those keys and values from scratch at every step: to produce
the hundredth token, reprocess the previous ninety-nine. That is quadratic work and
no one does it. Instead the server computes each token's keys and values once and
stores them, so that generating the next token only needs to compute the new token's
contribution and attend against the stored history. That store is the KV cache. It
is what turns generation from quadratic into linear, and it is not optional — take it
away and decode collapses to a crawl.

The cache is per-request, because each request has its own distinct history. And it
grows: every token a request processes, whether from its prompt or its own output,
adds another entry for every layer of the model. This is the crucial property.
Weights are a fixed cost — you pay for them once when the model loads, and they do
not grow no matter how many callers arrive. The KV cache is a *per-caller, per-token*
cost, and it is the thing that actually scales with load. Two hundred callers each
holding a long conversation can consume more memory in cache than the model's
weights do, and when that memory is gone, it is gone: the server cannot admit
request two hundred and one, and no amount of idle compute changes that.

## The arithmetic you should be able to do in your head

The size of the cache is not mysterious. It follows from the model's shape by a
formula you can and should be able to estimate quickly, because it is the single
most useful back-of-the-envelope calculation in serving. For each token, the cache
must hold a key and a value — that is the factor of two — for every layer, for every
attention head that keeps its own key and value, at whatever numeric precision the
cache uses. Multiply those together and you have the bytes per token; multiply by
the context length and you have the bytes per request; divide the memory budget by
that and you have how many requests fit.

The listing does exactly this for a representative mid-size model that uses grouped
attention — where many query heads share a smaller number of key-value heads, a
design choice made largely to shrink this very cache — with a sixteen-bit cache.

```python
# How many concurrent requests fit? It is a memory question, not a compute one.
# KV bytes per token = 2 (K and V) * layers * kv_heads * head_dim * dtype_bytes
def kv_bytes_per_token(layers, kv_heads, head_dim, dtype_bytes):
    return 2 * layers * kv_heads * head_dim * dtype_bytes

# A mid-size model, GQA with 8 KV heads, fp16 cache.
per_tok = kv_bytes_per_token(layers=48, kv_heads=8, head_dim=128, dtype_bytes=2)
print(f"KV per token: {per_tok/1024:.1f} KiB")

budget_gib = 24.0                       # cache memory left after weights + activations
budget = budget_gib * (1024**3)
for ctx in (2048, 8192, 32768):
    per_seq = per_tok * ctx
    fits = int(budget // per_seq)
    print(f"  ctx={ctx:5d}: {per_seq/(1024**2):7.1f} MiB/seq  ->  "
          f"{fits:3d} concurrent sequences fit in {budget_gib:.0f} GiB")

# Prefix sharing: 40 sequences that share a 1500-token system prompt.
shared = 1500; private = 400; n = 40
naive = n * (shared + private) * per_tok
shared_once = (shared + n * private) * per_tok
print(f"shared 1500-tok prefix across {n} seqs: "
      f"{naive/(1024**2):.0f} MiB naive vs {shared_once/(1024**2):.0f} MiB shared "
      f"({100*(1-shared_once/naive):.0f}% saved)")
```

```output
KV per token: 192.0 KiB
  ctx= 2048:   384.0 MiB/seq  ->   64 concurrent sequences fit in 24 GiB
  ctx= 8192:  1536.0 MiB/seq  ->   16 concurrent sequences fit in 24 GiB
  ctx=32768:  6144.0 MiB/seq  ->    4 concurrent sequences fit in 24 GiB
shared 1500-tok prefix across 40 seqs: 14250 MiB naive vs 3281 MiB shared (77% saved)
```

Read that middle block slowly, because it is the whole chapter in three lines. With
twenty-four gigabytes of memory for cache — a plausible slice of a single GPU after
weights and working buffers are subtracted — this model serves sixty-four callers if
each keeps a two-thousand-token context, sixteen callers if each keeps eight
thousand, and only four if each keeps thirty-two thousand. Nothing about the compute
changed across those rows. The same GPU, the same weights, the same math throughput.
The only thing that changed was how much history each caller keeps, and that alone
moved the concurrency limit by a factor of sixteen. This is why "how many at once"
cannot be answered without also answering "how long a context," and why a server
that advertises a huge maximum context length and high concurrency in the same
breath is usually promising something it cannot deliver simultaneously.

## Fragmentation, and the paging idea that fixed it

Knowing the cache's total size is not the same as being able to use all of it, and
this gap was, for a while, one of the great silent wastes in serving. The trouble is
that a request does not know in advance how many tokens it will generate. If the
server reserves a contiguous block of memory big enough for each request's maximum
possible length, most of that block sits empty for most requests — reserved against
a length they will never reach. And because the blocks are contiguous and of
different sizes, the free memory between them fragments into pieces too small to
reuse, exactly the way a poorly managed heap does. Measurements in the PagedAttention
work found that this internal and external fragmentation could waste the majority of
the cache memory — that only a fraction of the reserved space held actual keys and
values [R1]. A server can be "out of memory" while most of its cache is air.

The fix, introduced by vLLM's PagedAttention, is borrowed straight from operating
systems: stop demanding contiguous memory [R1]. Divide the cache into small
fixed-size blocks, like pages of virtual memory, and let each request's history be
scattered across whatever blocks are free, linked by a table that maps the logical
sequence of tokens to their physical blocks. A request now allocates a block only
when it actually fills one, so the empty space at the end of a maximum-length
reservation never gets reserved in the first place, and because every block is the
same size, freed blocks are always reusable — external fragmentation is eliminated,
since equal-size blocks can never leave a hole too oddly shaped to reuse [R1]. What
paging does not remove is internal fragmentation: the last block a request holds is
almost never exactly full, so a partial block's worth of memory is still wasted per
sequence. That residual waste is bounded by a single block, though — a few tokens, not
a full max-length reservation — which is why the reported effect was to cut the total
wasted cache memory to a few percent rather than to literally zero, and, because more
of the memory now holds real work, to raise the achievable batch size and throughput
substantially over the previous generation of servers [R1]. Smaller blocks shrink that
last-block waste further but cost more bookkeeping, so the block size is itself a
tunable trade rather than a free win. This
idea proved so useful that it became the default mental model for cache management,
and later work has argued about the right way to implement it — vAttention, for
instance, revisits whether the paging should live in the serving framework or lean
on the hardware's own virtual-memory machinery [R11] — but the core insight, that
the cache should be paged rather than reserved in contiguous lumps, is now simply how
serving is done.

## Prefix sharing: the same tokens, cached once

Paging buys a second gift almost for free, and it is the last block of the listing
above. If two requests begin with the same tokens — the same long system prompt, the
same few-shot examples, the same document being asked ten different questions — then
the keys and values for that shared prefix are *identical*. There is no reason to
store them twice. Under a paged cache, the shared prefix's blocks can be pointed to
by every request that shares it, computed once and read by all, with copy-on-write
semantics that fork a private block only at the first token where two requests
diverge. vLLM builds this on its paged blocks directly; SGLang generalizes it with
RadixAttention, organizing cached prefixes into a radix tree so that any shared
beginning between any requests is automatically reused, not just an exact prefix
match [R12].

The savings are not marginal when prompts share structure, which in practice they
constantly do — a chat application prepends the same system prompt to every
conversation, a classification job prepends the same instructions to every document.
The listing's last line shows forty requests sharing a fifteen-hundred-token system
prompt, each adding four hundred private tokens: storing the prefix once cuts the
cache from about fourteen gigabytes to about three, a seventy-seven percent saving,
which translates directly into serving several times as many callers on the same
memory. Prefix sharing is one of the rare levers that improves throughput and
latency at once — less memory per request means more requests fit, and the shared
prefix does not need to be prefilled again, so time to first token drops too. When
you design the prompt your server sees, putting the stable, shared material first and
the variable material last is not a style preference. It is a memory optimization
worth a multiple in capacity.

## Shrinking the cache at its source

Paging and sharing use the cache more efficiently, but there is a second front: making
each token's cache entry smaller in the first place. Because concurrency is set by
cache size, anything that shrinks the per-token bytes buys concurrency directly, and
the model's own architecture is where most of that shrinking happens — often before you
ever touch the server.

The largest lever is the number of key-value heads, which sits right in the arithmetic.
Early transformer designs gave every query head its own key and value, so the cache
scaled with the full head count. Grouped-query attention, now standard in serving-
oriented models, lets many query heads share a smaller set of key-value heads, and
multi-query attention takes this to the extreme of a single shared key-value head. The
motivation was almost entirely this cache: cutting the key-value head count from,
say, sixty-four to eight cuts the cache to an eighth, which is the difference between
serving eight callers and sixty-four at the same context. When you choose a model to
serve, its key-value head count is not a footnote — it is a direct multiplier on how
many people you can serve, and two models of the same parameter count can differ
eightfold in how many callers they hold.

The second lever is the cache's numeric precision. The listing assumed a sixteen-bit
cache, but the keys and values can often be stored at eight bits with little quality
loss, halving the cache and doubling the concurrency ceiling. This is not free — it is
a quality trade, and it must be measured rather than assumed, because the cache is more
sensitive to precision in some models than others. It also comes with a warning the
bench learned the hard way in a related setting: not every low-precision cache format
is safe for every model. On one model, storing the value cache in a particular eight-
bit format corrupted the output outright, and the fix was to keep that cache at full
precision even while other parts were quantized. The general rule is that cache
quantization is a real and large lever, but one to enable deliberately and verify with
your own quality suite, not a switch to flip on faith.

A third lever changes what needs caching at all. Some models use attention variants
that do not attend over the entire history — sliding-window attention, for instance,
attends only to a recent window of tokens, which bounds the cache per request at the
window size no matter how long the conversation grows. This turns the cache from a
cost that grows without limit into a fixed cost per request, which is transformative
for long-running sessions, at the price of the model literally not being able to attend
to what fell out of the window. Whether that trade is acceptable is a model-quality
question, but its effect on the serving budget is enormous: a bounded cache means a
concurrency limit that does not erode as conversations lengthen.

None of these levers is the operator's alone to pull — the head count and attention
variant are baked into the model, and the cache precision is a setting whose quality
cost only the model can reveal. But knowing they exist reframes model selection. When
you are choosing what to serve, you are not only choosing a quality level; you are
choosing a cache footprint, and the cache footprint is your concurrency. A slightly
weaker model with a far smaller cache can serve so many more callers that it wins on
every axis that matters to the people paying the bill.

## Eviction, and admitting more than you can hold

Even with paging and sharing, demand can exceed the cache. Callers keep talking,
contexts keep growing, and eventually the sum of everyone's history does not fit.
The server then faces a choice that operating systems have faced forever: what do
you evict? The options are to refuse new admissions until memory frees up — clean but
it grows the queue — or to evict some in-flight request's cache to make room, which
means that request must reconstruct its state later. Reconstruction has two flavors:
recompute the evicted request's KV from its tokens by re-running prefill, which costs
compute but no extra memory tier, or swap the KV out to slower memory and back, which
costs bandwidth. vLLM implements both recomputation and swapping as recovery paths
for preempted requests [R1][R13]. Either way, an evicted request pays a latency
penalty when it resumes, and that penalty lands in your tail. A server that quietly
evicts and recomputes under memory pressure will show you a fine median latency and a
horrible ninety-ninth percentile, and you will not understand why until you connect
the tail to the eviction, which is the kind of connection Chapter 5 is built to help
you make.

Between the two recovery paths there is a choice with a clear logic, and it is a small
worked example of thinking in tiers. Recomputation throws away an evicted request's cache
and rebuilds it later by re-running prefill over its tokens; swapping keeps the cache but
moves it to slower memory and back. Which is cheaper depends on the relationship between
your compute and your bus. Recomputation costs a prefill, which is compute-bound and, as
Chapter 4 will show, fast; swapping costs two trips across the bus for the whole cache,
which is bandwidth work. On a machine with spare compute and a slow bus, recomputing is
often the better deal, because you are paying with the resource you have to spare rather
than the one that is already your bottleneck. On a machine with fast interconnect and
saturated compute, swapping can win. The point is not that one always beats the other
but that the choice is a tier calculation you can actually do, and a server that lets you
pick the recovery path is handing you a real lever over your tail latency under memory
pressure.

The practical consequence of all this is a discipline, not a trick. Before you serve,
do the arithmetic in the listing for your actual model and your actual memory, decide
the context length you will genuinely promise, and size your maximum concurrency to
what the cache holds at that context — not to what the compute could theoretically
sustain. On the bench, this arithmetic is what separated the fit recipes that loaded
from the ones that did not: a large model that ran happily when its cache and spilled
layers were budgeted to leave a few gigabytes of headroom per card would fail
outright, with an out-of-memory error on the very first GPU, when the split was
nudged to claim that headroom for one more layer of weights. The failure was not
gradual. Memory does not degrade; it runs out. The whole art is to know the number
before you hit it, and the number comes from the cache.
