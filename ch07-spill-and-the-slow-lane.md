# Chapter 7 — Spill and the Slow Lane

Every chapter so far has assumed, quietly, that the model and its cache live in fast
memory. On the hardware most people actually have, that assumption breaks. The model
you want to serve is bigger than the fast memory you can afford, and something has to
give. What gives is that part of the model — or part of the cache — gets pushed down
to a slower tier: system memory over the bus, or in the extreme, disk. This is spill,
and it changes the economics of everything. A server with spill is not a slightly
slower version of a server without it. It is a different machine with different
sweet spots, where techniques that help when everything fits can actively hurt, and
where a single slow request can drag an entire batch into the mud.

## Why spill happens and what it costs

The root cause is the memory hierarchy. A GPU's on-board memory is fast — hundreds of
gigabytes to a terabyte per second of bandwidth — but limited in size and expensive.
System memory is large and cheap but reaches the GPU across a bus an order of
magnitude slower. Disk is enormous and cheaper still and slower again by orders of
magnitude. When a model's weights do not fit in GPU memory, a serving system can keep
the overflow in system memory and stream it across the bus as needed, a technique
DeepSpeed-Inference formalized as heterogeneous memory management, letting a machine
serve a model far larger than its GPU memory by treating system memory, and even
disk, as extended tiers for the weights [R10]. This is what makes it possible to run
an enormous model on a modest box at all. It is also what makes that box slow in a
very specific way.

Recall from Chapter 4 that decode is memory-bandwidth-bound: each token requires
reading the whole model, and the pace of decode is set by how fast those weights
arrive. If some of the weights live across the slow bus, then every token that needs
them waits at the slow tier's speed, not the fast tier's. The spilled fraction acts
as a tax on every token. And crucially, that tax is paid *per token read of the
spilled weights*, which is the fact that makes spill interact so badly with several
of the techniques from earlier chapters.

## The batch amortization breaks down

Here is where spill overturns intuition. Chapter 2 taught that adding requests to a
decode batch is nearly free, because the weight read is shared: read the weights
once, serve everyone in the batch. That is true when the weights are in fast memory.
When the weights are spilled, the read is still shared across the batch in principle,
but the read is now so slow that it dominates everything, and the benefit of
spreading it across more requests is capped by how fast you can pull the weights over
the bus at all. The batch's throughput scaling flattens early, because the bus, not
the compute, is the wall. On the bench this showed up starkly: a model that fit
entirely in fast memory scaled its throughput steeply with concurrency, while the
large spilled model on the same box climbed from twenty-six tokens per second at one
caller only to about forty-six at four — useful, real, but a shadow of the near-linear
scaling a resident model shows. The spill set the ceiling, and no scheduler flag
could raise it, because the limit was physical.

The measured law from the bench, stated plainly, is that the payoff from concurrency
and from speculative techniques grows as spill shrinks. With a heavily spilled
configuration the gains were small; with a lightly spilled one they were larger; with
no spill at all they were largest. That ordering is not a quirk of one model. It is
what you should expect whenever the binding constraint is a slow tier: any technique
whose benefit depends on cheap, repeated weight reads gets that benefit throttled in
exact proportion to how much of the weights are stuck behind the slow bus.

## Speculation, and why draft length one wins under spill

The sharpest illustration of spill economics comes from speculative decoding, a
technique for going faster that, under spill, must be tuned against its own grain.
Speculative decoding works by having a cheap process guess several tokens ahead, then
having the full model verify those guesses in a single forward pass — if the guesses
are right, you got several tokens for the cost of one verification, and if they are
wrong, you fall back and lose a little. When the model fits in fast memory, guessing
more tokens per round is usually better, because verifying a batch of guesses is cheap
and each correct guess is nearly free.

Under spill, the arithmetic inverts, and the reason is exactly the per-token cost of
reading spilled weights. Verifying N guessed tokens in one pass requires the model to
process N positions, and on a spilled model that means reading the spilled experts
roughly N times over the slow bus. The cost of a verification step therefore grows
with the number of tokens you guessed, and it grows at the slow tier's price. Guess
too many and the verification's slow-tier reads cost more than the extra tokens are
worth. The listing models this trade for the bench's lightly spilled configuration,
with a per-extra-token surcharge fit to that box.

```python
# Speculative decoding on a spill-bound MoE: why draft length 1 wins.
# Verifying N drafted tokens re-reads the spilled expert weights ~N times over
# the slow tier (DDR5), while accepted tokens are the payoff. Model the net.
BASE_TPS = 22.2               # measured baseline decode, 14 layers spilled
def expected_speedup(n_max, accept):
    # cost of a verify step grows ~linearly with drafted tokens when spilled;
    # a spilled expert read dominates, so cost ~ (1 + k*n_max) relative units.
    k = 0.55                  # per-extra-token spill surcharge (fit to this box)
    cost = 1.0 + k * n_max
    # tokens produced per verify step = 1 (the free token) + accepted drafts
    yielded = 1.0 + sum(accept ** i for i in range(1, n_max + 1))
    return yielded / cost
for n_max, acc in ((1, 1.00), (2, 0.89), (3, 0.78)):
    s = expected_speedup(n_max, acc)
    print(f"n_max={n_max}  accept={acc:.2f}  ->  {s:.2f}x  "
          f"({BASE_TPS*s:.1f} tok/s modeled)")
print("sweet spot on this spill: n_max = 1 (first-token drafts accept ~100%)")
```

```output
n_max=1  accept=1.00  ->  1.29x  (28.6 tok/s modeled)
n_max=2  accept=0.89  ->  1.28x  (28.4 tok/s modeled)
n_max=3  accept=0.78  ->  1.08x  (24.0 tok/s modeled)
```

The model says guess exactly one token ahead, and the bench agreed. Measured on the
real server, speculating a single token gave a clean speedup with the drafted token
accepted essentially all the time — the first guessed token is the one the drafting
mechanism is trained to get right, so it lands at nearly a hundred percent — while
guessing two or three tokens ahead dropped both the acceptance rate and the net speed,
because the extra verification reads over the slow bus cost more than the occasionally
accepted extra tokens returned. The measured speedups tracked the amount of spill:
about 1.18 times faster with fourteen layers spilled, about 1.30 times with ten
spilled, and, on a small model with no spill at all, past 2 times — the same law, that
the payoff grows as the slow tier shrinks. The practical rule for anyone running a
spilled model is counterintuitive and worth pinning up: when you are spill-bound, keep
the speculation short. The aggressive setting that helps a resident model hurts a
spilled one, because it multiplies the very reads that are already your bottleneck.

## Two things can spill, and they spill differently

So far "spill" has meant weights pushed to a slow tier, but the cache can spill too, and
the two behave differently enough that conflating them leads to the wrong fix. It is
worth separating them because the symptom — a slow server — is the same while the cause
and the cure are not.

Weight spill, the case above, is a steady tax: the spilled weights are read on every
token by every request, so the penalty is uniform and predictable, a fixed reduction in
decode speed proportional to the spilled fraction. You feel it as a lower ceiling on
tokens per second that does not change much with load, and you address it by shrinking
the model's fast-memory footprint — quantizing weights, or accepting fewer resident
layers — or by adding fast memory.

Cache spill is spikier and more insidious. When the fast memory reserved for the KV
cache fills, a server that offloads cache pushes some requests' history to system memory
and pulls it back when those requests are scheduled, the swapping recovery path from
Chapter 3. The penalty here is not uniform: it lands only on the requests whose cache
was evicted, and only when they resume, so most requests see nothing while a few see a
large stall as their history is dragged back across the bus. This is a tail-latency
problem, not a throughput-ceiling problem, and it hides from any average. A server that
looks healthy in aggregate can be quietly swapping a handful of unlucky requests' caches
in and out, giving those requests a terrible experience that the mean latency washes out
entirely. The fix is different too: not shrinking the weights but shrinking the cache
demand — shorter promised contexts, cache quantization, fewer concurrent sequences — so
that the cache stops overflowing in the first place.

The reason to keep them straight is that they point at opposite levers. If your decode
is uniformly slow across all requests, you have weight spill, and giving the cache more
memory makes it worse, because you took that memory from the weights. If your decode is
fine for most requests but a minority stall badly, you have cache spill, and giving the
weights more memory makes it worse, because you took that memory from the cache. The two
compete for the same scarce fast memory, and every gigabyte you move from one to the
other trades one kind of slowness for the other. Diagnosing which one you have — from
whether the slowness is uniform or concentrated in the tail — tells you which way to move
the boundary.

Sharding the model across several devices is the structural escape from both, and it is
why multi-GPU serving exists: spread the weights and the cache across four devices and
each holds a quarter, often enough to eliminate spill entirely and jump from the slow
tier's economics back to the fast tier's. But sharding is not free either. The devices
must exchange partial results every layer, and that exchange runs across the links
between them, which are faster than the system bus but slower than a device's own memory,
so a model split across devices pays a communication tax on every token that a resident
model does not. Split too aggressively — more devices than the model needs — and the
communication tax can cost more than the spill it was meant to avoid. The bench's fit
recipes were, in the end, a search for the split that kept each device loaded with as
much as it could hold while leaving headroom, because the best configuration is almost
always the least sharding that avoids spill, not the most.

## One slow request stalls the batch

Spill also creates the slow lane of this chapter's title, and it is a scheduling
hazard, not just a throughput one. Requests in a batch advance together, in lockstep,
one token per forward pass for all of them. That lockstep is what makes batching
efficient, but it has a dark side: the batch moves at the speed of its slowest member
on each step. If one request in the batch needs a spilled expert that the others do
not — which happens naturally in a mixture-of-experts model, where different tokens
route to different experts — then the whole batch's forward pass waits for that one
request's slow-tier read. The fast requests in the batch are held to the pace of the
slow one, not because they need anything slow but because they are yoked to it for
the duration of the step.

This is head-of-line blocking from Chapter 5, reappearing inside the batch rather than
in front of it, and it is nastier because it is invisible to the queue. Your admission
policy did nothing wrong; the request was admitted fairly and is being served. It is
simply dragging its batch-mates because of where its weights happen to live. The
symptom is a batch whose per-token latency is worse than any of its members would show
alone, and the diagnosis requires knowing that spill and routing can make one request
in a batch structurally slower than the rest. The mitigations are the same in spirit
as the queue's: try not to batch a known-slow request with latency-sensitive fast
ones, and, where the framework allows, route or group requests so that the members of
a batch are more alike in what they need. But the deeper mitigation is to shrink the
spill, because a request cannot drag the batch to the slow tier if there is no slow
tier for it to reach.

## Living within the hierarchy

The honest conclusion is that spill is a compromise, not a defeat, and the operator's
job is to make the compromise deliberately rather than stumble into it. If you must
spill — and on prosumer hardware serving large models you often must — spill the parts
that are read least, keep the hottest weights and the cache in fast memory, and size
the split so a few gigabytes of headroom remain per device, because the failure mode
of getting it wrong is not slowness but an outright out-of-memory crash on load. Do
the cache arithmetic from Chapter 3 first, because the cache competes with the weights
for the same fast memory, and every gigabyte you give the cache is a gigabyte of
weights you push to the slow tier. Measure the spilled configuration's real throughput
curve rather than assuming it scales like a resident one, because it does not. And
tune the speed techniques — speculation length, batch size — against the spilled
reality, where less is often more, rather than against the resident intuition that
more is always better.

It is also worth asking, before you spill at all, whether you should serve a different
model instead. Spill is one answer to "the model does not fit," but it is not the only
one, and it is often not the best. A smaller model that fits entirely in fast memory can
serve several times the throughput of a larger model forced to spill, and for many tasks
the smaller model's answers are close enough that the throughput wins decisively on cost
per useful answer. Quantizing the model's weights to a lower precision is the other
escape: a model at four bits per weight is a quarter the size of one at sixteen, and may
fit resident where the full-precision version could not, trading a measured and often
small quality loss for the enormous economic difference between resident and spilled
serving. The bench's own comparisons made this concrete — a carefully quantized model
that kept its most sensitive weights at higher precision held nearly all of the original
quality while fitting in far less memory, and the naive alternative of re-encoding
already-compact weights onto a different grid only made them larger for no benefit. The
point is that spill should be a deliberate choice made after the alternatives are
weighed, not a default reached for the moment the model does not fit. When you find
yourself sizing a spilled configuration, pause and price the resident alternatives
first: a smaller model, a quantized one, or fewer promised tokens of context. Sometimes
spill really is the right answer, but it should win the comparison, not skip it.

Spill is where the clean abstractions of the earlier chapters meet the ceiling of a
real machine, and it rewards the same posture the whole book has argued for: measure
the actual behavior, read the load log when a number is pathological, run the control
that isolates the variable, and let the hardware tell you where its slow lane is
rather than assuming it has none. Which is a natural place to gather everything into a
protocol, because knowing the physics is only useful if it changes what you do when
you sit down in front of a real server — and that is the last chapter.
