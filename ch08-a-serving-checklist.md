# Chapter 8 — A Serving Checklist

Knowing the physics is worth little if it does not change what you do when a real
server is in front of you. This chapter turns the book into a protocol: an ordered
way to size a server before it exists, to load-test it once it does, and to read its
behavior under real traffic without fooling yourself. It is written as prose rather
than a card of bullet points on purpose, because each step carries a reason, and the
reason is what keeps you from applying the step where it does not belong. Follow the
order. Most serving disasters come from skipping the early steps and discovering,
under load, the constraint you could have computed in advance.

## Start with the memory arithmetic, before anything runs

Before you launch a server, before you load-test, before you tune a single flag, do
the cache arithmetic from Chapter 3 on paper. Take your model's layer count, its
number of key-value heads, its head dimension, and its cache precision, and compute
the bytes of cache per token. Take the memory you will have left after the weights and
working buffers are loaded, and divide. That quotient, against your intended context
length, is the ceiling on how many callers you can serve at once — and it is almost
always the real ceiling, well below whatever the compute could sustain. If the number
disappoints you, your levers are known and finite: a shorter promised context, a
smaller cache precision, a model with fewer key-value heads, prefix sharing if your
prompts share structure, or more memory. There is no scheduler setting that
manufactures cache out of nothing.

The sizing listing inverts the arithmetic into the two questions you actually need
answered: given a memory budget and a concurrency you want to promise, what context
length can you honestly offer; and, separately, what does a load test say your real
limit is once latency enters the picture.

```python
# A back-of-the-envelope sizer: given a memory budget and a target concurrency,
# what context length can you actually promise? Invert the KV-budget arithmetic.
def kv_per_token(layers, kv_heads, head_dim, dtype_bytes=2):
    return 2 * layers * kv_heads * head_dim * dtype_bytes
per_tok = kv_per_token(48, 8, 128)
cache_gib = 24.0
cache = cache_gib * (1024**3)
print(f"cache budget {cache_gib:.0f} GiB, KV {per_tok/1024:.0f} KiB/token")
for target_conc in (8, 16, 32, 64):
    max_ctx = int(cache / (target_conc * per_tok))
    print(f"  promise {target_conc:2d} concurrent  ->  up to {max_ctx:6d} ctx tokens each")
# The inverse question a load test answers: at what concurrency does p99 latency
# cross your SLA? That number, not the memory ceiling, is your real limit.
sla_ms = 2000
observed = {1: 120, 4: 260, 8: 520, 16: 1400, 32: 3100}
safe = max((c for c, ms in observed.items() if ms <= sla_ms), default=0)
print(f"measured: highest concurrency under {sla_ms} ms p99 = {safe}")
```

```output
cache budget 24 GiB, KV 192 KiB/token
  promise  8 concurrent  ->  up to  16384 ctx tokens each
  promise 16 concurrent  ->  up to   8192 ctx tokens each
  promise 32 concurrent  ->  up to   4096 ctx tokens each
  promise 64 concurrent  ->  up to   2048 ctx tokens each
measured: highest concurrency under 2000 ms p99 = 16
```

The two answers rarely agree, and that gap is the point. Memory says this box could
hold sixteen callers at eight thousand tokens of context each. The load test, with a
made-up but realistic latency curve, says that beyond sixteen concurrent callers the
ninety-ninth-percentile latency crosses a two-second budget. When memory and latency
disagree, latency wins, because a server that fits more callers than it can serve
within your promise is not serving them — it is enqueuing them. Your real concurrency
limit is the smaller of the two ceilings, and you only learn the latency ceiling by
measuring.

## Establish the single-stream baseline, and read the load log

With sizing done, load the server and measure it with exactly one request at a time,
warm. This single-stream baseline is your reference point for everything that
follows, and it does two jobs. It tells you the best per-token latency the server can
achieve when nothing is contending, which is the ceiling your latency will fall away
from as load rises. And it is your first honesty check against the load log, which you
should read now, before trusting any number. The load log is where the server confesses
what it actually did — which layers went to which device, whether an operation you
expected on the GPU landed on the CPU, how much memory each device actually holds. On
the bench, a decode rate stuck at an absurd two tokens per second, immune to every
scheduling flag, was explained by one line in that log revealing a key operation
running on the CPU. No amount of tuning would have found it, because it was not a
tuning problem. The rule from that experience generalizes: when a number is
pathological and no flag moves it, stop tuning and read the load log — you are almost
certainly looking at a placement or fit problem the log will name outright.

While you are reading the log, confirm the fit has headroom. A configuration that
loads with a few gigabytes free per device is safe; one that loads with almost nothing
free is a crash waiting for the first slightly larger request. The bench learned this
by nudging a working split to claim one more layer's worth of headroom for weights and
getting an out-of-memory failure on the first device instead of a slightly faster
server. Leave the headroom.

## Load-test with the traffic you will actually see

Now raise the concurrency and watch what happens, and here the discipline is to test
the load you expect rather than the load that is easy to generate. A load test that
sends identical short prompts at a fixed rate will tell you almost nothing about a
server that will face mixed prompt sizes arriving in bursts, because it exercises none
of the phenomena that matter — no prefill interference, no head-of-line blocking, no
tail. Build your test load to mirror reality: a realistic spread of prompt lengths, a
realistic spread of output lengths, and arrivals that come in bursts rather than a
metronome. Then measure at several concurrency levels, not just one, because the
throughput curve has notches. The bench saw a real one: on a particular build,
throughput at two concurrent callers dipped below the single-stream number before
recovering at three and four, a scheduling quirk that only appeared because someone
measured every point instead of just the endpoints. Assume your curve has a notch
somewhere and find it before your users do.

At every concurrency level, record percentiles, never averages. The median tells you
how a lucky request fares; the ninety-ninth percentile and the maximum tell you how
the server behaves when it matters, and those are the numbers your users actually
experience. A server whose average latency looks fine can have a tail that makes it
unusable, and the tail only appears under contention, so a percentile taken at low
load is not a smaller version of the truth — it is a different number. Watch queue
depth alongside latency, because queue depth turns up before latency does: a queue
that stops returning to empty is your early warning that you are at capacity, arriving
before the latency percentiles have climbed enough to alarm you.

## Soak it, then test the failure modes on purpose

A server that passes a five-minute load test can still fail in an hour, and the
failures that take an hour to appear are the ones that take down production at three in
the morning. Run a soak: real, sustained, mixed load for long enough that slow
problems have time to surface. Watch memory over the whole run. A stable server holds
its memory nearly flat — the bench's production configuration drifted two megabytes
over a twelve-minute soak, the signature of genuine stability — while a server that
leaks or fragments its cache climbs slowly toward the out-of-memory cliff and crosses
it only after the demo is over and everyone has gone home confident. The soak is the
only test that catches this, because the problem is defined by time.

Then break it on purpose, because you need to know how it fails before it fails on its
own. Push concurrency past the limit you found and confirm that the server degrades
gracefully — shedding or refusing load with fast, honest rejections — rather than
accepting everything and letting every caller's latency climb together into collapse.
A server without admission control does not have a high capacity followed by a clean
wall; it has a soft ceiling beyond which everyone suffers equally, which is worse than
a hard refusal because it fails quietly and for everybody at once. Confirm, too, that
a canceled request actually frees its resources, that a caller who hangs up mid-stream
does not leave a slot and a cache allocation stranded, and that a single pathological
request cannot wedge a batch indefinitely.

## Measure quality under the load you will serve, not at idle

The last step ties back to Chapter 6, and it is the one most teams skip. When you
evaluate the model's quality — its accuracy, its tool use, whatever your suite
measures — do it against the server under the batching regime you will actually run,
not against a pristine single-stream instance, and run the suite more than once. The
same prompt at temperature zero can yield different tokens depending on how the batch
was packed, so a quality number taken from one run against one batch shape is a single
draw from a distribution you have not characterized. The bench's tool suite swung
about ten points across identical back-to-back runs at temperature zero, driven by
batch-packing nondeterminism amplified by expert routing; a single number from inside
that band would have been indistinguishable from noise. Run the suite several times,
report the range, and separate the results that survive every run — the scenarios that
consistently pass or consistently fail — from the ones that flicker. Only the durable
results should drive a decision. If you need a repeatable number for a regression gate,
pin the batch to a fixed shape for that measurement and accept the throughput cost, and
be clear with yourself that you have stepped out of the production regime to get it.

## Instrument the four numbers that explain the rest

A server you cannot see into is a server you cannot operate, and the instruments worth
having are few and specific. Four numbers, watched continuously, explain almost every
behavior the earlier chapters described, and a dashboard that shows them is worth more
than one crowded with dozens of metrics that no one reads.

The first is queue depth over time — how many requests are waiting for admission at each
moment. This is your earliest warning, because it rises before latency does: a queue
that stops returning to empty is a server at capacity, and it tells you so minutes before
the latency percentiles climb enough to page anyone. The second is the two latency
percentiles split by regime: time to first token and time per output token, each at the
median and the ninety-ninth percentile. Splitting them is what turns "it is slow" into a
diagnosis, because a bad first-token tail points at prefill or the queue while a bad
per-token tail points at decode or spill, and the split tells you which subsystem to open
before you have touched anything. The third is cache occupancy — how full the KV cache is
relative to its budget. This is the number that predicts the cliff from Chapter 3: a cache
creeping toward full under sustained load is a server about to start preempting, and the
preemptions will land in your tail, so watching occupancy lets you see the cause before
the symptom. The fourth is a utilization number for the binding resource — compute if you
are compute-bound, effective memory bandwidth if you are bandwidth-bound — because it tells
you whether there is headroom to grow the batch or whether you are already at the wall.

Two derived numbers make the raw four legible. Goodput — the throughput that actually met
your latency objective, not the raw token count — is the honest measure of a server under
load, because tokens produced for requests that already blew their deadline are not
serving anyone. And a simple saturation indicator, whether queue depth and cache occupancy
are both trending up together, catches the overload spiral early enough to shed load
before it becomes a collapse. What you do not need is a wall of per-layer timings and
kernel counters; those belong in a debugging session, not a running dashboard. The
operator's continuous view is queue depth, split latency tails, cache occupancy, and the
binding resource's utilization — four numbers that, read together, tell you which of this
book's chapters your server is currently living in.

## Reading a server you did not build

Often the server is not yours to configure — it is a managed endpoint, a vendor's API,
a black box behind a URL — and yet you still need to know how it behaves before you
build on it. The same physics lets you reverse-engineer a server's character from the
outside, with nothing but a stopwatch and a willingness to send it shaped traffic.

Send it prompts of increasing length and time the first token. If time to first token
climbs roughly with prompt length, you are watching prefill, and the slope tells you
its prefill throughput. If first-token time barely moves with prompt length but jumps
under concurrency, the server is queueing your prompt behind others, and you are
measuring its scheduling rather than its prefill. Send it a fixed prompt at rising
concurrency and watch the token pace: the point where per-token latency starts climbing
is the server's crossover, the edge of its free-throughput region, and it is the
concurrency past which you should not expect the endpoint to stay fast. Send it the same
prompt twice at temperature zero and compare the outputs token for token; if they
diverge, the endpoint batches you with other traffic and does not use batch-invariant
kernels, which tells you immediately that you cannot rely on it for reproducible
results and must design your evaluation around a range.

Two shaped probes reveal the most. First, a burst: fire many requests simultaneously
after a quiet period and watch whether the first responses come back promptly while
later ones stretch, which is healthy queueing, or whether all of them stall together,
which suggests the server admitted the whole burst and let everyone's latency climb —
a sign it lacks the admission control to protect you under load. Second, a long
generation next to short ones: start one request generating thousands of tokens, then
send short requests alongside it, and see whether the short ones are served promptly or
dragged. If they are dragged, the server has head-of-line blocking you will inherit, and
a single heavy user — possibly you — can degrade everyone. None of these probes needs
access to the server's internals. They work because the behavior is forced by the same
constraints this book has laid out, and a server cannot hide its physics from traffic
shaped to expose it.

Interpreting the probes is where the earlier chapters pay off. A rising first-token time
under concurrency but flat token pace means prefill contention, addressable, if it were
your server, with chunked prefill. A flat first-token time but degrading token pace means
crowded decode batches or a bandwidth-starved model. Divergent temperature-zero outputs
mean batch-dependent reductions. Stalling shorts behind a long generation mean
head-of-line blocking. You are, in effect, running the diagnostic vocabulary of the book
against a box you cannot see inside, and the vocabulary is enough to predict how the box
will treat your production load before you commit to it.

## The protocol in one breath

The order is the message, so here it is compressed. Compute the cache budget and
derive your honest concurrency and context ceilings before anything runs. Establish a
warm single-stream baseline and read the load log to confirm the model landed where
you think it did, with headroom to spare. Load-test with realistic mixed, bursty
traffic at several concurrency levels, recording percentiles and watching queue depth,
and find the notch in the curve. Soak it long enough that leaks and creeping queues
have time to appear, watching memory drift as the tell. Break it deliberately to
confirm it sheds load rather than collapsing, and that cancellations free their
resources. And measure quality under the real batching regime, multiple times, as a
range, keeping only the findings that outlast the noise.

None of this requires exotic tools. It requires the willingness to measure before
assuming, to read the log before tuning, to run the control that isolates the
variable, and to publish the range instead of the lucky draw. That posture is the
whole of good serving practice, and it is what separates a server you can rely on from
one that merely worked the day you demonstrated it. The batch gives you throughput,
the cache sets your ceiling, the queue keeps your promises, and the measurement keeps
you honest. A hive does not thrive because any one forager is fast; it thrives because
the colony has learned, over a long season, exactly how much its shared store can bear
and never promises the field more than the entrance can carry. Serve the same way.
