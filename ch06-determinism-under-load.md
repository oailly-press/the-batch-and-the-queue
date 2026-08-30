# Chapter 6 — Determinism Under Load

Set the temperature to zero and the model is supposed to become a function: the same
prompt yields the same output, every time, because at zero temperature generation
picks the single highest-probability token at each step with no randomness left to
sample. This is what "deterministic" is supposed to mean, and it is what most people
building on top of a model assume they have. They are wrong, and the reason they are
wrong is not the model. It is the batch. The same prompt, sent to the same server
with the same weights at temperature zero, can produce different tokens on different
runs — and the difference depends on who else was in the batch. This chapter is about
why, and about the quiet damage it does to anyone trying to measure a model's quality.

## The same prompt, a different answer

Begin with the observation itself, because it is genuinely surprising the first time
you see it and it is easy to misdiagnose. I ran a tool-use evaluation on the bench
against a served model at temperature zero: fifteen scenarios, each scored pass or
fail, run several times back to back with nothing changed between runs — same server,
same weights, same prompts, same zero temperature. The suite did not return the same
score. Across identical runs it swung by about ten points out of a hundred, and the
churn was not spread evenly across scenarios; a stable majority passed or failed
consistently while a handful — five of the fifteen, in one careful accounting —
flipped between passing and failing from one run to the next. One run scored the
model at seventy-three, another at forty-three, a third at fifty, with no change to
anything I controlled. If I had run the suite once and published the number, I would
have published a coin flip and called it a measurement.

The instinct is to blame sampling — surely some randomness leaked in. But the
temperature was zero; there was no sampling to leak. The instinct's next stop is
floating-point "noise," waved at vaguely as if numerical error were a random
fog. That is closer but still wrong, and the difference between "random floating-point
noise" and the real cause is the whole point of this chapter.

## Floating point is exact but not associative

Computers represent real numbers in finite precision, and the arithmetic on those
representations has a property that trips almost everyone: addition is not
associative. Group the same numbers differently and you can get different results,
not because the computer made an error but because rounding happens at each step and
the order of steps changes where the rounding falls. This is not noise — it is
perfectly deterministic. Given the exact same operations in the exact same order, you
get the exact same answer every time. The result only changes when the *order* of the
additions changes.

The listing shows both halves of this: that reordering changes the result, and that
it does so deterministically.

```python
# Floating-point addition is not associative: the same values, combined in a
# different order, can land on a different result. In a batched matmul the
# reduction order depends on how requests were packed into the batch, so the
# same prompt at a different batch position can round differently -- and one
# different last bit, greedily argmax-ed, can pick a different token.
a = (0.1 + 0.2) + 0.3
b = 0.1 + (0.2 + 0.3)
print(f"(0.1+0.2)+0.3 = {a!r}")
print(f"0.1+(0.2+0.3) = {b!r}")
print(f"equal? {a == b}")

def chunked(xs, k):                 # sum in blocks of k, then sum the blocks
    return sum(sum(xs[i:i+k]) for i in range(0, len(xs), k))

data = [0.1] * 30
one_shot = sum(data)
print(f"one-shot sum   : {one_shot!r}")
for k in (5, 7, 8):
    c = chunked(data, k)
    print(f"chunked by {k:>2}  : {c!r}  equal-to-one-shot? {c == one_shot}")
```

```output
(0.1+0.2)+0.3 = 0.6000000000000001
0.1+(0.2+0.3) = 0.6
equal? False
one-shot sum   : 3.0
chunked by  5  : 3.0  equal-to-one-shot? True
chunked by  7  : 3.0000000000000004  equal-to-one-shot? False
chunked by  8  : 3.0  equal-to-one-shot? True
```

The first pair shows the raw fact: three numbers, added in two groupings, two
different results, and the equality test says so. The second block is the one that
matters for serving. Summing thirty copies of one-tenth is a *reduction* — exactly
the operation at the heart of every matrix multiply, where a row and a column are
multiplied elementwise and the products summed. Do that reduction in one sweep and
you get one answer; break it into chunks of seven and sum the chunks and you get a
different answer in the last bit; break it into chunks of five or eight and you are
back to the first answer. The result depends on the chunk size — and in a real matmul
kernel, the chunk size is a function of how the work was tiled across the hardware,
which is a function of how big the batch is and where in it your request landed.

## The batch decides the reduction order

Now the pieces connect. When your request is computed as part of a batch, the kernels
that do its matrix multiplies choose how to split the work — how to tile the
reductions across the hardware's parallel units — based partly on the shape of the
whole batch. A batch of two is tiled differently from a batch of thirty. Your
request's numbers are the same, but the *order in which they are summed* is set by the
batch it rode in. Different order, different rounding, different value in the last
bit of some logit. Usually that last bit does not matter, because the top token wins
by a comfortable margin. But when two candidate tokens are nearly tied, a
last-bit difference is enough to swap which one is highest, and at temperature zero
the server takes the highest and commits to it. From there the sequences diverge:
one different token early changes the context for every token after it, and two runs
that agreed for fifty tokens now write different sentences.

This is the mechanism the Thinking Machines analysis of nondeterminism in inference
lays out and names precisely [R5]. Its central point is that the culprit is not
concurrency-induced randomness in the usual sense, nor unavoidable hardware jitter,
but the *lack of batch-invariance* in the kernels: the operations give different
results for different batch sizes because their reduction strategy changes with batch
shape, so a request's output depends on what it was batched with [R5]. The analysis
goes further and shows that one can build batch-invariant kernels — reductions whose
order does not depend on batch shape — and recover bit-for-bit determinism across
batch sizes, at some performance cost [R5]. That is the real fix, and it is a
kernel-level fix, not something an operator can conjure with a flag. The point for us
is diagnostic: if your temperature-zero server is nondeterministic, you are almost
certainly looking at batch-dependent reductions, not a bug in your code and not
sampling you forgot to disable.

On the bench the effect was amplified by the model's architecture. The model was a
mixture of experts, which routes each token to a subset of expert weights, and the
routing decision is itself made from these same logits. A last-bit wobble can change
which expert a token is routed to, which changes the computation entirely, which is a
far larger perturbation than a single swapped output token. That is why the tool
suite's flips were concentrated in a handful of scenarios and why they were as large
as they were: batch-packing nondeterminism at the base, magnified by routing on top.
The scenarios that flipped were the ones where the model was genuinely near a
decision boundary; the ones that held were the ones where it was not.

## The other doors nondeterminism comes through

Batch-packing is the dominant source of temperature-zero nondeterminism in a busy
server, but it is not the only one, and a diagnostician should know the others so as
not to chase the wrong cause. They share the same root — a reduction whose order is not
fixed — but they enter through different doors.

One door is kernel selection. High-performance libraries often keep several
implementations of the same operation and pick one at runtime based on the input
shapes, sometimes even auto-tuning by trying candidates and caching the fastest. Two
implementations of a matrix multiply can sum in different orders, so which one gets
picked changes the last bit, and the pick can vary with shape or with what the tuner
happened to measure. This is why the same operation can be deterministic within a run
and vary across runs that saw different shapes: the batch changed the shape, the shape
changed the kernel, the kernel changed the order.

A second door is parallel reduction across hardware units. When many units cooperate on
one sum, the order in which their partial results combine can depend on timing — which
unit finished first — and if the combination uses an operation that races, the order is
genuinely nondeterministic, not merely batch-dependent. Atomic accumulation, where many
threads add into one location without a fixed order, is the classic offender. This is
the one case that looks like true randomness rather than hidden determinism, and it is
why some operations are nondeterministic even at a fixed batch size until they are
explicitly configured to use an ordered reduction.

A third door opens only across machines: multi-device serving splits the model across
several GPUs and combines their partial results, and the combination order can differ
from the single-device path or vary with how the model was sharded. A number computed on
four GPUs need not match the same number computed on one, for the same associativity
reason. If you compare a quality run taken on one hardware layout against a run taken on
another, some of the difference you see is the layout, not the model.

The practical recipe for reproducibility, when you truly need it, is to close as many of
these doors as you can afford. Pin the batch to a fixed shape, so the reduction order
stops moving with load. Disable auto-tuning and select kernels deterministically, so the
implementation stops varying. Prefer ordered reductions over atomic ones where the
framework offers the choice, accepting the speed cost. Hold the hardware layout fixed
across the runs you intend to compare. Each of these trades performance for
repeatability, which is exactly the trade you want for a measurement and exactly the
trade you do not want in production, and the whole discipline is knowing which of those
two situations you are in at any moment.

## What this does to evaluation

The damage this does is subtle because it does not look like a bug. Your server
works. Your evaluation runs. It produces a number. The number is simply not
repeatable, and if you do not know that, you will make decisions on noise. Suppose
you change a setting — a quantization level, a scheduler flag, a prompt template —
and rerun the suite, and the score moves by six points. Did your change help? On a
suite with ten points of run-to-run swing, you cannot say. The six points are inside
the noise. You might have improved the model, harmed it, or done nothing, and a
single before-and-after pair cannot distinguish the three. Teams ship regressions and
celebrate phantom gains this way constantly, and the root cause is that they treated a
noisy measurement as an exact one.

The discipline that defends against this is not exotic; it is the ordinary discipline
of measurement applied to a place people forget it applies. Run the suite more than
once. A surprising number gets run again; two runs that disagree get a third and a
control. Report a range, not a lucky draw — the honest summary of the bench's tool
suite is "forty-three to seventy-three across three runs," not any single number from
inside that band. And separate the signal that survives the noise from the noise
itself: on the bench, the score swung wildly, but certain scenarios passed in every
single run and certain others failed in every run, and *those* consistent results are
real findings that no amount of batch nondeterminism can explain away. A scenario that
newly passes in three runs out of three, after failing in every prior configuration,
is a genuine improvement; a two-point move in the aggregate is weather. Learning to
tell the durable result from the flicker is the core skill, and it starts with
refusing to trust any single run.

It is worth saying clearly that most of the time nondeterminism is fine, and chasing it
away would be a waste. The last-bit wobble changes an output only when the model is
genuinely near a decision boundary, and near a boundary the two candidate answers are,
almost by definition, about equally good — the model was nearly indifferent between them.
A summarization that comes out slightly reworded, a classification that lands on one of
two near-tied labels, a generation that picks a different but equally valid phrasing:
none of these is a defect, and forcing bit-exact determinism to prevent them would cost
throughput to eliminate variation that did not matter. The goal is not to make every
server deterministic. The goal is to know whether yours is, and to stop treating a
noisy measurement as a precise one. Determinism is a property you spend performance to
buy, and you buy it only where you have a concrete reason: a regression test that must
not produce false alarms, a scientific comparison whose whole point is isolating one
variable, a debugging session that needs the same failure to appear twice.

The failure to guard against, then, is not variation itself but *unacknowledged*
variation — the silent kind that corrupts a decision. The team that reruns a benchmark
after a change, sees the score move, and attributes the move to their change without ever
having measured the benchmark's own run-to-run spread has been fooled, and the fooling is
invisible because the number looks authoritative. The defense costs almost nothing: run
the baseline twice before you change anything, so you learn the noise floor of your own
measurement, and treat any change smaller than that floor as no change at all. A team
that knows its suite swings ten points will never again celebrate a six-point gain, and
that single piece of self-knowledge prevents more bad decisions than any amount of
kernel engineering. The measurement's honesty is worth more than the measurement's
precision, and honesty here means nothing more than knowing your own error bars before
you read anything into a difference.

There is a serving-side lever worth knowing, too, even though it is not free. If you
truly need reproducibility — for a regression test that must not lie, for a scientific
comparison, for a debugging session where you need the same failure twice — you can
often get much closer to it by pinning the batch. Serving a request alone, or forcing
a fixed batch shape, removes the variable that was moving the reduction order.
Reproducibility bought this way costs throughput, because a fixed or singleton batch
gives up the amortization that batching exists to provide, so it is a mode you switch
into for measurement and out of for production, not a way to run a server. The
alternative, batch-invariant kernels, buys determinism without giving up batching but
requires the kernels to exist and be enabled [R5]. Either way, the operator's job is
to *know which regime they are in* — reproducible-but-slow or fast-but-nondeterministic
— and never to confuse a number from one with a number from the other. The worst
outcome is not nondeterminism; it is nondeterminism you did not know you had,
quietly turning every measurement into a coin flip you mistook for a ruler.
