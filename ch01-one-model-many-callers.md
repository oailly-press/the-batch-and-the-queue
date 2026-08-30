# Chapter 1 — One Model, Many Callers

A model, sitting on a disk, does nothing. It is a large table of numbers. To turn
those numbers into an answer, something has to load them into fast memory, feed a
prompt through them token by token, and hand the result back. The first time you
do this you write a loop: read a prompt, call the model, print the reply, repeat.
The loop is honest and it works, and for a single person poking at a single model
it is all you will ever need.

Then a second person shows up. And a third. And a monitoring job that fires every
thirty seconds, and a batch of ten thousand documents somebody wants classified by
morning. The loop that served one caller does not gracefully become the thing that
serves a thousand. What you need instead is a *server*: a long-running process that
holds the model in memory once and multiplexes it across every caller who arrives,
deciding moment to moment whose work advances and whose waits. That process is the
subject of this book, and the shift from the loop to the server is larger than it
looks. It is not the same code with a queue bolted on. It is a different kind of
system, with its own physics, its own failure modes, and its own arithmetic — and
almost none of that arithmetic is about the model's quality. It is about memory,
bandwidth, scheduling, and the strange things that happen when many requests share
one set of weights.

## Inference in a loop is not serving

Start with what the loop gets wrong, because the wrongness is instructive. In a
loop, each request owns the machine for its entire lifetime. It arrives, it runs to
completion, it leaves, and only then does the next request begin. The model's
weights sit in memory the whole time, but they do useful work only during the
fraction of each request when a token is actually being computed. The rest is
overhead: loading the prompt, waiting on memory, tearing down and setting up. If
one request takes four seconds and you have a hundred of them, the hundredth caller
waits almost seven minutes for a machine that was, most of that time, not busy so
much as *occupied*.

The waste has a specific shape, and naming it is the first step toward fixing it.
When a language model generates text, it does so one token at a time, and producing
each token requires reading essentially every weight in the model out of memory.
For a mid-sized model that is tens of gigabytes moved per token. The arithmetic
units that multiply those weights against your prompt are, during this phase,
mostly idle — they finish their multiply long before the next slab of weights
arrives from memory. The bottleneck is not how fast the chip can compute. It is how
fast it can *read*. A single request cannot saturate that memory bandwidth on its
own; it asks for one token's worth of work at a time, and the hardware could have
done far more with the same weight read. The loop leaves that capacity on the floor
for every request, one after another.

A server exists to reclaim it. If the expensive act is reading the weights, and the
weights get read anyway to serve one request, then serving a second request *in the
same weight read* is very nearly free. This single observation — that the dominant
cost is shared, not per-request — is the seed of everything that follows. Batching,
which the next chapter is entirely about, is the mechanism. But the mechanism only
makes sense once you have stopped thinking of a request as a function you call and
started thinking of it as an object the server manages.

## The request as a first-class object

In the loop, a request is a transient: it lives on the call stack for the duration
of one function and vanishes. In a server, a request is a durable thing with
identity, state, and a lifecycle. It arrives and is admitted, or refused. It waits
in a queue. It gets its prompt processed. It produces tokens, one at a time, across
many scheduling rounds, interleaved with other requests doing the same. Eventually
it finishes — because it hit a stop token, or reached its length limit, or the
caller hung up — and the resources it held are reclaimed and handed to whoever is
next in line.

Every serious serving system, from vLLM to TensorRT-LLM to the one you might write
yourself, is organized around this object and its states. Orca, the system that
introduced the scheduling discipline most modern servers now use, framed the whole
problem at the granularity of a single iteration — one token step — precisely so
that the scheduler could make decisions about each request on every step rather
than once at the start [R2]. The request is what the scheduler schedules. It is
what holds a slice of the memory budget. It is what shows up in your latency
percentiles. If you cannot name a request's current state and say what it is
waiting for, you cannot reason about the server at all.

Here is that lifecycle made concrete. The listing models two requests moving
through the states a real server tracks — queued, prefilling, decoding, finished —
one token step at a time. It is not the model; it is the bookkeeping that surrounds
one.

```python
# The request is a first-class object with a lifecycle, not a function call.
# A server holds many of these at once, each in one of a few states.
STATES = ["queued", "prefilling", "decoding", "finished"]
class Request:
    def __init__(self, rid, prompt_len, max_new):
        self.rid = rid; self.prompt_len = prompt_len
        self.max_new = max_new; self.produced = 0
        self.state = "queued"
    def step(self):
        if self.state == "queued":
            self.state = "prefilling"
        elif self.state == "prefilling":
            self.state = "decoding"          # KV for the prompt now exists
        elif self.state == "decoding":
            self.produced += 1
            if self.produced >= self.max_new:
                self.state = "finished"
        return self.state

pool = [Request(i, prompt_len=6, max_new=n) for i, n in enumerate((2, 3))]
tick = 0
while any(r.state != "finished" for r in pool):
    snapshot = [f"r{r.rid}:{r.step()}" for r in pool]
    print(f"t={tick}  " + "  ".join(snapshot))
    tick += 1
print("all finished; the server frees their KV and admits the next from the queue")
```

Running it prints the two requests advancing together, each finishing when it has
produced the tokens it asked for:

```output
t=0  r0:prefilling  r1:prefilling
t=1  r0:decoding  r1:decoding
t=2  r0:decoding  r1:decoding
t=3  r0:finished  r1:decoding
t=4  r0:finished  r1:finished
all finished; the server frees their KV and admits the next from the queue
```

Two things in that trace matter more than they seem. The requests advance *in
step*: at each tick, both take one token's worth of progress. That lockstep is the
batch, and it is what lets the shared weight read serve both at once. And `r0`
finishes before `r1`, and the moment it does, its resources are free. A server that
waits for the whole batch to drain before admitting new work wastes exactly the gap
between `r0`'s finish and `r1`'s. A server that admits new work into `r0`'s freed
slot the instant it opens does not. That difference — filling slots continuously
rather than in fixed batches — is worth a large multiple in throughput, and it is
the whole content of Chapter 2.

## What the server is actually made of

Strip a serving system down and you find a small number of moving parts, each of
which gets its own chapter in this book because each has physics you can measure and
get wrong.

There is the *batch*: the set of requests advancing together in a single forward
pass. Its size is the throughput lever, and its composition — which requests, in
which phase — determines whether a given pass is well spent or half wasted.

There is the *KV cache*: the memory that holds, for each request, the intermediate
attention state for every token it has seen so far. This is the quietly dominant
resource. It grows with every token every request produces, it cannot be recomputed
cheaply, and when it runs out the server cannot admit another request no matter how
idle the compute units are. How many callers you can serve at once is, in the end, a
question about this cache — a point important enough that Chapter 3 is devoted to it.

There are the *two regimes*, prefill and decode. Processing a prompt and generating
a token look like the same operation but stress the hardware in opposite ways:
prefill is compute-bound, decode is memory-bandwidth-bound. Mixing them in one batch
means one of them is always being under-served, and the attempts to reconcile them —
chunking prefill, splitting the two phases onto different hardware — are an active
and fascinating corner of the field [R6][R8][R9].

There is the *queue*: the line of requests waiting for admission, and the policy
that decides who gets in and in what order. The queue is where fairness lives, where
head-of-line blocking hides, and where your tail latency is quietly decided long
before any token is computed.

And there is the load itself, which refuses to be uniform. Some prompts are short,
some enormous. Some requests want three tokens, some want three thousand. They
arrive in bursts. They cancel halfway through. The server's job is to keep the
shared resource busy and the callers reasonably treated across all of that
variation, and the tension between those two goals — utilization and fairness — is
the tension you will spend your serving career managing.

## Two shapes of load: the stream and the flood

Not all serving is the same kind of serving, and the difference changes which numbers
you optimize. There are two broad shapes, and it is worth naming them because a
technique that is right for one can be wrong for the other.

The first shape is online, interactive load: a person or an application is waiting on
the other end of each request, right now, for a response they will read as it
arrives. A chat interface is the clearest case. Here latency is sovereign. A caller
who waits four seconds for the first token has a bad time no matter how efficient the
server was in aggregate, and a server that streams tokens at a pace slower than
reading speed feels broken even if its throughput is enormous. Online serving lives
and dies by time to first token and by the steadiness of the token stream, and it will
often accept lower hardware utilization to keep those good. It also faces the hardest
version of the load problem, because interactive traffic is bursty and unpredictable —
everyone opens their laptop at nine in the morning — and the server must hold its
latency promises through the bursts, which is why admission control and tail latency,
the subjects of Chapter 5, matter most here.

The second shape is offline, batch load: a pile of work with no human waiting on any
individual item, only on the pile as a whole. Classifying ten million documents
overnight, generating embeddings for a corpus, scoring a dataset — these have a
deadline for the batch but no per-item latency requirement at all. Here throughput is
sovereign and latency is nearly irrelevant. You want every scrap of the hardware busy
every moment, so you run the largest batches the memory allows, pack the queue full,
and never leave a slot idle waiting to be gentle to some individual request, because
no individual request is waiting. Offline serving is, in a sense, the easy case: with
no latency promise to keep, you can push utilization to the ceiling and let the tail
fall where it may.

Most real deployments are a mixture, and the mistake is to tune one as if it were the
other. Running interactive traffic with offline settings — huge batches, full queues —
gives you superb throughput and a chat that stutters and lags, because you optimized
the number no interactive user cares about at the expense of the two they feel.
Running offline work with interactive settings — small batches, generous headroom —
gives you a snappy server that takes all night to finish a job that should have taken
an hour, because you paid for latency nobody was waiting to collect. When you sit down
to tune a server, the first question is not which flags to set. It is which shape of
load this server actually carries, because that answer decides which of the two
numbers you are allowed to sacrifice.

## The two numbers everyone actually argues about

Every serving decision eventually reduces to a fight between two numbers, and it
helps to have them named before the fight starts. The first is *throughput*: how
many tokens the server produces per second across all callers combined. This is the
number the person paying for the hardware cares about, because it sets the cost per
token — a server that produces twice the tokens per second on the same box halves
the bill. The second is *latency*: how long an individual caller waits, which
itself splits into two useful sub-measurements. Time to first token is how long
from submitting a prompt until the first word comes back, and it is dominated by
prefill and by how long the request sat in the queue. Time per output token is the
pace of the words after that, and it is dominated by decode and by how crowded the
batch is. The person using the product cares about these, because they are what the
experience feels like.

The uncomfortable truth is that these two numbers pull against each other. The
lever that raises throughput is a bigger batch: more requests sharing each weight
read means more tokens per second in aggregate. But a bigger batch means each
request's token now waits behind more company on every step, so its time per output
token gets worse. Push the batch large enough and throughput is magnificent while
every individual caller feels the server crawl. Shrink the batch toward one and each
caller gets the machine's full attention while the box sits mostly idle and the cost
per token soars. There is no setting that maximizes both; there is only a chosen
point on a curve, and choosing it wisely for your particular load is much of the job.

Keep those two numbers, and the tension between them, in the front of your mind for
the rest of the book. Nearly every mechanism I describe is a way of buying more of
one without giving up too much of the other, and nearly every failure I describe is
one of them collapsing while nobody was watching the other.

## Why the numbers are not about the model

It is tempting, coming from the model side, to assume that a better model makes a
better server. It does not, not in the sense that matters here. The quality of the
answers and the economics of serving them are almost orthogonal. A brilliant model
served badly gives brilliant answers to the first ten callers and timeouts to the
rest. A modest model served well gives adequate answers to everyone: fast, cheaply,
predictably. The skills in this book are the second kind, and they transfer across
models. The KV-cache arithmetic is the same whether the weights are good or bad, the
queueing behavior is the same, the prefill-decode split is the same.

I will make this concrete throughout the book with measurements from a specific
machine, described honestly as observations rather than cited as authority. The
apparatus is a workstation I will call the bench: a Threadripper 9970X with 128 GB
of DDR5-4800 system memory and four RTX PRO 4500 Blackwell GPUs providing 128 GB of
aggregate video memory, running llama.cpp [R17] and, for one cross-check, vLLM
[R22]. On that bench, serving a large mixture-of-experts model, single-stream
decoding runs at roughly 26 tokens per second. That number is the model's quality
made visible only in the sense that it is the same regardless of whether the answer
is right. Push the same server to four concurrent callers and aggregate throughput
rises into the mid-forties of tokens per second — not four times as much, but close
to double, and the shape of that curve, why it bends where it bends, is a serving
fact, not a model fact. When I report such numbers I will say how they were taken
and what varied between runs, because a serving number without its conditions is
noise, and this press has learned the hard way to publish the conditions.

## What this book assumes and what it promises

You can read Python and hold your own in a shell. You have called a model, probably
through an API, and gotten text back. You have not necessarily thought about what
happened on the other side of that call, and you have never had to size a server,
read its latency percentiles under load, or explain why the same prompt gave two
different answers on two different days. You do not need to know how a GPU kernel is
written or how a model is trained; where the internals matter, I will describe them
in the terms a server operator needs and no more.

The promise is narrow and I intend to keep it. When you close the book you will
understand serving as a resource-allocation problem with a handful of levers, you
will know which lever moves which number, and you will have a protocol for measuring
a real server's real behavior instead of guessing. Every listing runs on the
standard library alone, and the publisher's acceptance gate re-executes the runnable
ones before this text reaches you; the small simulations exist so that the arithmetic
is something you can run and perturb, not something you take on faith. Listings
marked as fragments are illustrative and were not meant to execute.

One more framing before we begin, and it is the one to keep. A colony of bees does
not assign one forager per flower and wait for each to return before dispatching the
next. It sends the whole foraging force out at once against a shared field, pools
what they bring into one store, and meters the store against the entire hive's
needs. The interesting problems are never about a single bee's speed. They are about
contention for the store, the width of the entrance, and how the colony behaves when
the field is thin. A model serving many callers is exactly this: one shared store of
weights and cache, a crowd of independent requests, and an entrance only so wide.
The rest of this book is the physics of that entrance.
