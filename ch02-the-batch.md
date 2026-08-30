# Chapter 2 — The Batch

The batch is the reason a server exists at all. Everything else — the cache, the
queue, the two regimes — is machinery in service of getting the batch right. So it
is worth being precise about what a batch is, because the word carries an old
meaning that is exactly wrong for modern serving.

In the old meaning, a batch is a fixed group of items you process together and then
release together, like a tray of bread. You gather requests, run them through the
model in one pass, wait for all of them to finish, and only then gather the next
tray. This is *static batching*, and it was how the first serving systems worked. It
captures the central insight — share the weight read across requests — but it wastes
that insight almost as fast as it captures it, because requests do not finish
together. In a tray of ten requests, one asks for three tokens and another asks for
three hundred. Static batching holds the whole tray until the three-hundred-token
request is done, and for the last two hundred and ninety-seven steps that request is
being served alongside nine empty seats. The batch shrank to one, but the machinery
kept running a batch of ten.

## From trays to a turnstile

The fix is to stop thinking in trays and start thinking in turnstiles. Instead of
admitting and releasing requests in fixed groups, admit each request the moment
there is room and release it the moment it finishes, refilling its seat immediately
with whoever is next in line. The batch is no longer a tray that fills and empties;
it is a population that churns, with new requests flowing in through the gaps left by
finished ones on every single step. This is *continuous batching*, sometimes called
dynamic or in-flight batching, and it is the single most important idea in the whole
field.

The scheduling discipline underneath it was introduced by Orca, which called it
iteration-level scheduling [R2]. The name is precise and worth unpacking. A naive
scheduler makes its decisions at the granularity of a *request*: it picks a group,
runs the group to completion, picks the next. Orca makes its decisions at the
granularity of an *iteration* — a single token step. Before every forward pass, the
scheduler looks at the current population, evicts anyone who finished on the last
step, admits anyone waiting who now fits, and only then runs the pass. Because the
decision happens on every step rather than once per group, a request that finishes
early frees its capacity within one token's time, and a request that has been
waiting gets in within one token's time. The tray's dead seats never form.

The effect is not subtle. The listing from the previous chapter's family, expanded,
makes the gap between the two disciplines something you can watch. It runs the same
six requests — arriving at different times, wanting different numbers of tokens —
under both static and continuous batching, on a machine that can advance four
sequences per forward pass.

```python
# Continuous batching vs static batching: where does a token-step go?
# A tiny discrete simulator. No GPU, no ML -- just the bookkeeping a scheduler does.
STEP_SLOTS = 4          # how many sequences the model can advance per forward pass
# (arrival_step, prompt_len, output_len) for six requests
REQS = [(0, 6, 5), (0, 6, 12), (1, 6, 3), (2, 6, 8), (2, 6, 4), (5, 6, 6)]

def run(continuous):
    pending = sorted(REQS)
    active = []          # [tokens_left]
    finished = []        # (finish_step)
    step = 0
    while pending or active:
        # admit new work
        if continuous:
            while pending and len(active) < STEP_SLOTS and pending[0][0] <= step:
                _, _, out = pending.pop(0)
                active.append(out)
        else:
            # static: only admit when the batch is completely empty
            if not active:
                while pending and len(active) < STEP_SLOTS and pending[0][0] <= step:
                    _, _, out = pending.pop(0)
                    active.append(out)
        # one forward pass advances every active sequence by one token
        active = [t - 1 for t in active]
        done = [t for t in active if t <= 0]
        finished += [step] * len(done)
        active = [t for t in active if t > 0]
        step += 1
    return step, finished

for mode in (False, True):
    steps, fin = run(mode)
    label = "continuous" if mode else "static"
    print(f"{label:>10}: all six done after {steps:2d} steps "
          f"(last finish at step {max(fin)})")
```

```output
    static: all six done after 20 steps (last finish at step 19)
continuous: all six done after 12 steps (last finish at step 11)
```

Same requests, same hardware, same forward-pass width — and continuous batching
clears the whole workload in twelve steps where static takes twenty. The difference
is entirely the dead seats: under static batching, the long twelve-token request
holds three idle slots for step after step while short requests that could have used
them wait for the tray to empty. Change the mix so the spread between shortest and
longest request is wider — the realistic case — and the gap grows accordingly.

## How tokens from different requests share one pass

It is worth being concrete about what "advancing four sequences in one forward pass"
physically means, because the phrase hides the mechanism that makes all of this pay.

A forward pass through the model is a sequence of large matrix multiplications. The
weights are the same for every request — that is the whole point of one model,
many callers — so the server stacks the current token from each active request into
a single tall input matrix and runs it through the weights once. Four requests
become four rows; the weights are read from memory one time and multiplied against
all four rows together. The memory bandwidth that a single request could not
saturate is now feeding four multiplies per byte read instead of one. That is where
the throughput comes from: not from doing each request faster, but from amortizing
the expensive weight read across more requests.

There is one operation that does not fit this tidy picture, and it is the reason
serving has kernels written specifically for it. Attention is not shared across
requests, because each request attends over its own distinct history of tokens.
Request A's token attends to request A's thousand previous tokens; request B's token
attends to B's fifty. You cannot stack those into one clean matmul against shared
weights, because there are no shared weights in attention — there is per-request
state, the KV cache, of different sizes. Orca's answer was *selective batching*:
batch the parts that share weights, and handle attention per-request with an
operation shaped for ragged, variable-length state [R2]. Modern servers lean on
attention kernels like FlashAttention that are built to compute this efficiently
without materializing the enormous intermediate matrices a naive implementation
would [R3][R4]. You do not need to write these kernels, but you do need to know that
attention is the part of the batch that does not amortize the way the rest does, and
that its cost scales with the total tokens of history across the batch, not just the
number of requests. This will matter enormously in Chapter 3, where that history is
exactly the thing filling up memory.

## The throughput-latency lever, measured

The batch size is a dial, and turning it trades the two numbers from the previous
chapter against each other. Larger batch, more throughput, worse per-request
latency. The interesting question is the *shape* of that trade: is throughput linear
in batch size, so that doubling the batch doubles the tokens per second? For a while,
yes — as long as the weight read dominates and the compute units have spare
capacity, each added request is nearly free and throughput climbs almost linearly.
Then one of two ceilings arrives. Either the compute units saturate, and adding
requests no longer finds idle math to do, so throughput flattens; or the memory
holding the KV cache fills up, and you simply cannot admit another request, so
throughput flattens for a different reason. Which ceiling you hit first depends on
the model, the hardware, and the context lengths, and knowing which one binds is
half of sizing a server.

On the bench I can show both the payoff and its limits. Serving a large
mixture-of-experts model on the four-GPU workstation described in Chapter 1,
single-stream decode ran at 26.2 tokens per second. Taken to four concurrent
callers, aggregate throughput reached about 46 tokens per second — the shared weight
read paying off, but well short of the 4× that a perfectly free batch would give,
because this particular model spills part of its weights to system memory and the
slow tier caps how much the batch can amortize. That ceiling is real and it is the
subject of Chapter 7. For contrast, a smaller model that fits entirely in fast
memory scales far more steeply. As a cross-check on the same bench, I served a
120-billion-parameter mixture model — small enough to sit resident in video memory,
with nothing spilled — under vLLM, and measured it single-stream and under load:
about 60 tokens per second single-stream, climbing past 480 at eight concurrent
callers and into the high hundreds at sixteen. Those are my own measurements on the
workstation described in Chapter 1, not a published benchmark; I report them the same
way I report the spilled model's numbers above — with the apparatus stated, so anyone
with a resident 120B-class mixture model under vLLM on comparable hardware can
reproduce the shape of the curve. Same idea, very different curve — because one server
was memory-bound on a spilled model and the other was not. The lesson is not that one number is right. It
is that the batch's payoff is bounded by whichever resource runs out first, and you
cannot know the shape of your own curve without measuring it on your own load.

One honest wrinkle from those bench runs deserves mention now, because it previews a
theme. The concurrency curve was not always monotone. On one build, throughput at
two concurrent callers dipped *below* the single-stream number before recovering at
three and four — 26 at one caller, about 17 at two, back to 25 at three. Reading the
logs showed no re-prefilling and no cache churn; it was a scheduling quirk in how
that build packed a batch of exactly two. I report it not because it is important in
itself but because it is the kind of thing serving does: the curve you expect to be
smooth has a notch in it, and the only way to find the notch is to measure every
point, not just the endpoints. A server's behavior under load is an empirical fact
about that server, not a theorem you can derive from its parts.

## The knobs a real scheduler gives you

The simulation admits requests whenever a slot opens, which is the essence of
continuous batching but not the whole of a real scheduler. Production servers expose a
handful of knobs that shape the batch, and understanding what each one does saves you
from turning dials at random when latency goes wrong.

The first knob is the maximum number of concurrent sequences — how many requests may
be in the batch at once. This is a direct cap on the batch width, and therefore on
both throughput and per-request latency: raise it and you amortize the weight read
across more callers but slow each of them; lower it and each caller gets more of the
machine while the box runs cooler than it could. It is tempting to set this as high as
the memory allows, but that ties the knob to the cache ceiling from Chapter 3, and a
server configured to admit more sequences than its cache can hold will simply preempt
and thrash. The useful setting is the smaller of what the memory holds and what your
latency budget tolerates.

The second knob is the token budget per step — a cap not on the number of requests but
on the total tokens processed in one forward pass. This one exists mostly to tame
prefill. A step that admits a four-thousand-token prompt for prefill alongside thirty
decoding requests processes far more tokens than a pure decode step, and takes
correspondingly longer, stalling every decoder. A token budget lets the scheduler say:
this step may process at most so many tokens, so a giant prompt must be split across
several steps rather than crammed into one. This is the mechanism underneath chunked
prefill, and exposing it as a knob lets you trade first-token latency for smoother
decode: a smaller budget means prompts prefill over more steps, delaying their first
token but protecting the token pace of everyone already decoding.

The third knob is the policy for what to do under pressure — whether to preempt
in-flight requests when memory runs short, and if so, whether to recompute or swap
their cache, the recovery paths Chapter 3 examines. Some servers let you bias the
scheduler toward prefill or toward decode, deciding whether a newly arrived request's
prompt jumps ahead of ongoing generation or waits behind it. There is no universally
right setting; the choice depends entirely on whether your load is the latency-bound
stream or the throughput-bound flood from the previous chapter. A server carrying
interactive traffic wants to protect the token stream of requests already in flight; a
server grinding through an offline pile wants to keep the batch as full as possible and
does not care which request's first token slips.

The lesson underneath the knobs is that continuous batching is not a single algorithm
but a family of policies over a churning population, and the defaults a framework ships
are a guess about your load, not a law. When your latency is wrong, the fix is usually
one of these three dials, and knowing which one moves which number — sequences for
batch width, token budget for prefill smoothness, preemption policy for behavior under
pressure — is the difference between tuning and flailing.

## Prefill crashes the party

There is a complication in continuous batching that the simulation above quietly
ignored, and it is the seam where this chapter meets the next two. New requests do
not arrive ready to decode. They arrive with a prompt that must be processed first —
the prefill phase — and prefill is a fundamentally different shape of work from
decode. A decoding request contributes one token to the batch. A prefilling request
contributes its entire prompt, which might be one token or might be four thousand,
all at once. When the scheduler admits a new request into a batch of quietly
decoding ones, that batch's forward pass suddenly has to also chew through a large
prompt, and every decoding request in the batch waits for it. Callers experience
this as a stutter: the steady pace of their tokens hiccups whenever a big new prompt
joins.

This is head-of-line blocking wearing a different hat, and the field has spent real
effort taming it. The main idea is to stop treating a prompt as an indivisible lump.
Chunked prefill, introduced by SARATHI, splits a long prompt into smaller pieces and
feeds one piece per forward pass, interleaving those pieces with the ongoing decodes
so that no single step is dominated by one giant prompt [R6]. The follow-on work,
Sarathi-Serve, built a scheduler around this to hold the throughput-latency trade at
a chosen point under mixed load [R7]. A more radical answer, which the next chapters
return to, is to refuse to mix the two phases in the same batch — or even on the
same hardware — at all [R8][R9]. For now the point is only that the batch is not a
uniform thing. It is a mixture of requests in different phases, and the phases fight
for the same forward pass. Understanding that fight requires understanding the two
regimes on their own terms, which is Chapter 4. But the two regimes fight over a
resource, and that resource is memory, and memory is where we go first — because
before you can schedule the batch, you have to be able to hold it, and holding it is
the tightest constraint in the whole system.
