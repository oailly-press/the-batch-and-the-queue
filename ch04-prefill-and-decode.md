# Chapter 4 — Prefill and Decode

A request lives two lives. In the first, the server reads its prompt and builds the
cache for it — the prefill. In the second, the server generates the answer one token
at a time — the decode. These two phases run on the same weights and produce the same
kind of numbers, and it is natural to assume they are the same operation at different
scales. They are not. They stress the hardware in opposite directions, they have
opposite bottlenecks, and the fact that a single request must do both, back to back,
is one of the deepest structural tensions in serving. Almost every advanced serving
technique of the last few years is, at bottom, a way of managing the collision
between prefill and decode.

## Two operations, two bottlenecks

Return to the idea from Chapter 1 that a chip has two capacities that can each run
out: how fast it can compute, and how fast it can read memory. Every operation is
limited by one or the other. An operation that does a great deal of arithmetic per
byte it reads is *compute-bound* — it keeps the math units busy and waits on them.
An operation that does little arithmetic per byte is *memory-bandwidth-bound* — the
math units finish early and sit waiting for the next data to arrive. This framing,
the roofline model, is the single most useful lens for understanding why prefill and
decode behave so differently [R21].

Prefill is compute-bound. When the server processes a prompt, it has all the prompt's
tokens at once, and it pushes them through the weights together — a big, wide matrix
multiply, hundreds or thousands of tokens against each weight. The weights are read
from memory once and reused across every token in the prompt, so the arithmetic per
byte read is high, and the math units stay busy. Prefill is the model working at its
computational best.

Decode is memory-bandwidth-bound, and painfully so. When the server generates a
single token, it has exactly one token to push through the weights. It must still
read every weight in the model to do it — there is no way to compute a token without
consulting the whole model — but now that enormous read is amortized over a single
token's worth of arithmetic. The math units do a little multiply and then wait,
starved, for the next slab of weights to arrive. Decode is the model working at its
computational worst, limited not by how fast it can think but by how fast it can
read itself out of memory.

The listing makes the asymmetry numerical for a twenty-seven-billion-parameter model
on hardware with a few hundred teraflops of compute and a terabyte per second of
memory bandwidth. The numbers are illustrative round figures chosen to show the
shape, not a benchmark of any specific chip.

```python
# Prefill and decode are opposite regimes. Prefill does one big matmul over all
# prompt tokens at once (compute-bound); decode does one token at a time and is
# dominated by re-reading the weights from memory (bandwidth-bound).
PARAMS = 27e9            # 27B params
BYTES_PER_PARAM = 2      # fp16 weights
FLOP_PER_TOK = 2 * PARAMS
weight_bytes = PARAMS * BYTES_PER_PARAM
COMPUTE = 300e12         # 300 TFLOP/s usable
BANDWIDTH = 1.0e12       # 1 TB/s usable memory bandwidth

def prefill(tokens):    # all tokens share one weight sweep: compute dominates
    return (tokens * FLOP_PER_TOK) / COMPUTE
def decode_one():       # one token, must re-read every weight: bandwidth dominates
    return weight_bytes / BANDWIDTH

pf = prefill(1000)
print(f"prefill 1000 prompt tokens: {pf*1000:.1f} ms  "
      f"({1000/pf:,.0f} tok/s -- one big matmul)")
d = decode_one()
print(f"decode one token          : {d*1000:.1f} ms  "
      f"({1/d:,.0f} tok/s -- limited by reading {weight_bytes/1e9:.0f} GB of weights)")
print(f"the weight read is {weight_bytes/BANDWIDTH*1000:.1f} ms whether you decode "
      f"for 1 request or (batched) for many -- which is why batching decode is free "
      f"throughput until compute or memory runs out")
```

```output
prefill 1000 prompt tokens: 180.0 ms  (5,556 tok/s -- one big matmul)
decode one token          : 54.0 ms  (19 tok/s -- limited by reading 54 GB of weights)
the weight read is 54.0 ms whether you decode for 1 request or (batched) for many -- which is why batching decode is free throughput until compute or memory runs out
```

Five thousand tokens per second in prefill, nineteen in decode — a difference of
more than two orders of magnitude on the same weights. That gap is not a flaw to be
fixed; it is the nature of the two phases. And it explains the last line of the
output, which is the entire economic case for batching restated in physical terms.
The fifty-four-millisecond weight read that decode is stuck behind happens whether
one request is decoding or thirty are. So adding requests to a decode batch is nearly
free until the compute finally saturates — the extra math rides along on a read you
were paying for anyway. Prefill has no such gift, because it was already keeping the
math units busy; adding more prompt tokens to a prefill adds proportional compute
and slows it down.

## The bench sees the same split

This is not only textbook arithmetic; it shows up directly in serving measurements.
On the bench, serving a large mixture-of-experts model with a memory-resident
attention path, prompt processing ran at roughly 130 tokens per second while warm
single-stream decode ran at about 26. The absolute numbers are lower than the
listing's toy figures because this is a much larger model with part of its weights
spilled to system memory, but the *ratio* tells the same story: prefill moves tokens
several times faster than decode, because prefill is doing the arithmetic the
hardware is good at and decode is waiting on memory. The split even survived a
diagnostic surprise. An early build of the server decoded at a baffling two tokens
per second, and every scheduling flag left it unchanged; the cause turned out to be
a single line in the load log revealing that one operation had landed on the CPU
instead of the GPU, throttling the whole decode path. The fix restored decode to the
mid-twenties. The lesson, which recurs, is that when a regime's number is
pathological and no batching flag moves it, the answer is in the load log, not the
scheduler — you are looking at a placement problem, not a tuning problem.

## Why mixing them is hard

Here is the collision. A running server is mostly decoding: it has a population of
requests each generating tokens, all bandwidth-bound, all riding the shared weight
read. Then a new request arrives with a two-thousand-token prompt. To admit it, the
server must prefill that prompt — a big, compute-heavy operation — and the simplest
thing to do is fold it into the next forward pass alongside the decoders. But now
that forward pass is no longer a lean bandwidth-bound decode step. It has a fat
compute-bound prefill riding in it, and it takes far longer than a decode step
should. Every decoding request in that batch waits for the prefill to finish. The
callers who were getting a smooth stream of tokens feel a stall, precisely timed to
whenever a big new prompt joins. This is the prefill-decode interference problem,
and it is why a server under mixed load has a jittery per-token latency even when its
average throughput looks fine.

You cannot escape it by simply prioritizing one phase over the other, because both
are essential and both are on the critical path of every request. Prioritize
prefills and your decoders stall in bursts; prioritize decodes and new requests wait
a long time for their first token. The field's answers, which are worth knowing even
if you never implement them, come in two families.

The first family keeps the two phases on the same hardware but stops letting prefill
arrive as an indivisible lump. Chunked prefill, from SARATHI, breaks a long prompt
into bounded-size chunks and processes one chunk per forward pass, so a single step
never carries more than a fixed budget of prefill work [R6]. The clever part is what
SARATHI does with the leftover room in those steps: it piggybacks the ongoing
decodes onto the same passes, so the compute-bound prefill chunk and the
bandwidth-bound decodes share a step and each fills what the other leaves idle — the
prefill uses the math units the decodes were starving, and the decodes use the
memory reads the prefill was not [R6]. Sarathi-Serve turns this into a scheduling
policy that holds a chosen throughput-latency balance under real mixed traffic,
bounding the per-token stall while keeping the compute units fed [R7].

The second family is more radical: refuse to mix the phases at all, by running them
on separate hardware. Splitwise and DistServe both propose disaggregating prefill
and decode onto distinct machines or pools — one set of resources does nothing but
prefill, another does nothing but decode, and a request's cache is handed from the
first to the second when its prompt is done [R8][R9]. This looks wasteful at first —
you are dedicating hardware to each phase — but it lets each pool be tuned for its
own bottleneck, sizes each independently to the load, and eliminates the
interference entirely, because a decode machine never has a prefill dropped into its
batch. DistServe frames the goal as maximizing *goodput* — the useful throughput
that meets latency targets — rather than raw throughput, which is exactly the right
target once you accept that a stalled decoder is not really being served [R9]. For
very large deployments the disaggregated design has become increasingly common; for
a single box it is usually overkill, and chunked prefill is the more practical lever.

## Decode changes regime as the batch grows

There is a wrinkle in calling decode "memory-bandwidth-bound" that matters once you
are batching seriously, and it is one of the more useful things to understand about
why throughput curves bend where they do. Decode is bandwidth-bound *at small batch
sizes*, but it does not stay that way forever as you add requests.

Recall why decode is bandwidth-bound: it reads the whole model to produce very little
arithmetic, so the math units idle while waiting on memory. But when you batch many
decoding requests together, the weight read is shared while the arithmetic is not —
each request in the batch does its own multiply against the shared weights. So the
arithmetic per byte read climbs with the batch size. Read the weights once, do one
request's worth of math: bandwidth-bound, math units idle. Read the weights once, do
thirty requests' worth of math: much more arithmetic per byte, and at some batch size
the math units stop idling and become the bottleneck. Decode has crossed from
bandwidth-bound to compute-bound, purely by growing the batch.

That crossover point is one of the most important numbers in your server, because it
tells you where batching stops paying. Below it, every request you add to a decode
batch is nearly free, because you are filling idle math with a weight read you already
paid for — this is the free-throughput regime the earlier chapters celebrated. Above
it, the math units are saturated, and each new request now competes for compute that
is fully spoken for, so adding requests slows everyone without raising aggregate
throughput. The throughput curve, climbing steeply through the bandwidth-bound region,
flattens at the crossover into the compute-bound region. Knowing roughly where your
model and hardware cross is knowing where to set your maximum batch size: pushing past
the crossover buys you nothing but worse latency.

This is also why the two ceilings from Chapter 2 — compute saturation and cache
exhaustion — race each other, and why which one binds first is a real design question.
If your crossover batch size is larger than the number of requests your cache can hold,
the cache binds first: you run out of memory to admit requests before you run out of
math to serve them, and you are permanently in the free-throughput regime, wanting only
more memory. If the crossover is smaller than what the cache holds, compute binds
first: you saturate the math units and further concurrency is wasted memory. On a
memory-rich box with a small resident model you tend to hit compute; on a memory-tight
box with a large model and long contexts you tend to hit the cache. The roofline lens
[R21] makes this legible — it is the same story of an operation sliding along the two
capacities of the hardware, except here the operation's position on the roofline moves
as you change the batch, which is a lever you control.

## What to take to your own server

You do not need to implement any of these techniques to benefit from understanding
them, because they tell you what to look for. If your callers report smooth
generation that periodically stutters, you are watching prefill interference, and the
question to ask your server is whether it supports chunked prefill and whether it is
turned on. If your time to first token is bad under load but your token pace is fine
once started, your prefills are queued behind decodes and you may want to give
prefill more of the schedule. If your token pace is bad but first tokens are quick,
your decode batches are too crowded or your model is too bandwidth-starved, which is
a memory problem, not a scheduling one. The two regimes are a diagnostic vocabulary:
almost every latency complaint maps onto one of them, and knowing which one narrows
the fix from the entire server to a single subsystem.

It helps to map the two regimes onto the two latency numbers from Chapter 1, because
the mapping is clean and it makes complaints diagnosable. Time to first token is
prefill plus whatever the request waited in the queue: the caller submits a prompt, it
waits its turn, its prompt is prefilled, and the first token emerges. Everything that
makes prefill slow or the queue long shows up here and nowhere else. Time per output
token, the pace after the first, is decode: it is set by the weight read, the batch's
crowding, and any spill in the decode path. These two numbers have almost disjoint
causes, which is exactly why separating them is powerful. A caller who says "it takes
forever to start but then it is fine" is describing a prefill-or-queue problem, and no
amount of decode tuning will touch it. A caller who says "it starts right away but
crawls" is describing a decode problem, and no amount of prefill tuning will touch that
one. The single most common diagnostic mistake in serving is to hear "it is slow,"
treat it as one problem, and tune the wrong regime; asking only whether the slowness is
in the start or the pace splits the problem in half before you have touched a flag.

The mapping also explains why the same optimization can be a triumph or a
disappointment depending on the load. Prefix sharing from Chapter 3, for instance,
slashes time to first token, because a shared prefix does not need re-prefilling — so it
is transformative for a chat workload where every request carries the same long system
prompt and callers feel the start. But it does nothing for time per output token,
because decode still reads the whole model per token regardless of any shared prefix. If
your callers were complaining about the pace rather than the start, prefix sharing would
measure as a null result, and you might wrongly conclude it does not work, when in fact
it works perfectly on a number those callers were not complaining about. An optimization
is only as good as its match to the regime your load is bound by, and the regime is
something you diagnose, not assume.

The deeper point is that a request is not homogeneous work, and a server that
pretends it is will interfere with itself. Prefill and decode want opposite things
from the hardware, and the whole game is to let each have what it wants without
starving the other — to fill the compute the decoders leave idle with prefill, and
fill the memory reads the prefill leaves idle with decode, so that a forward pass is
never half a machine working and half a machine waiting. That balance is scheduling,
and scheduling means a queue, which is where we turn next.
