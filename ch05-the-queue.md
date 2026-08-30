# Chapter 5 — The Queue

The batch is what the server runs; the queue is what decides who gets into the batch
and when. It is the least glamorous part of a serving system and the part that most
determines how the server *feels* to the people using it. Throughput lives in the
batch, but latency — and especially the latency that generates complaints — lives in
the queue. A server can have a perfectly tuned batch and a beautiful cache and still
deliver a miserable experience because of how requests wait before they ever reach
the model. This chapter is about that waiting: what governs it, what it hides, and
how to read it.

## The queue is where demand meets a finite server

A queue forms for one reason: work arrives faster, sometimes, than it can be served.
If requests trickled in one at a time, each fully served before the next appeared,
there would be no queue and no scheduling to speak of. Real load is not like that.
Requests arrive in bursts, they vary wildly in size, and the server has a finite
number of slots — set, as Chapter 3 established, by the cache. When more requests
want in than there are slots, the surplus waits. The queue is that waiting line, and
admission control is the policy that governs it: who is let in, who waits, and who is
turned away.

There is a law that governs queues so generally that it holds regardless of how
requests arrive or how long they take, and it is worth internalizing because it
turns a vague intuition into arithmetic. Little's law states that the average number
of items in a system equals the average arrival rate multiplied by the average time
each item spends in the system: L equals lambda times W [R18]. For a server, the
number of requests in flight equals how fast they arrive times how long they stay.
Rearranged, it says something you can act on: if you want to hold latency W down at a
given arrival rate lambda, you must keep the number in the system L bounded — and
since your slots are finite, that means either serving faster or admitting less. You
cannot wish the relationship away. A server that admits everything at an arrival rate
its service rate cannot match will see W grow without bound, which is the formal
statement of the thing everyone has felt: an overloaded server does not get slightly
slower, it falls over, because the queue and therefore the wait grow without limit.

The practical corollary is that admission control is not optional cruelty; it is the
only thing standing between a busy server and a collapsed one. A server that says
"no" quickly to some requests when it is saturated keeps its promises to the
requests it accepted. A server that says "yes" to everything keeps its promises to
no one, because everyone's wait climbs together. Refusing or shedding load under
pressure — returning a fast, honest rejection rather than an eventual timeout — is a
feature, and its absence is the most common way a serving deployment turns a traffic
spike into an outage.

## Head-of-line blocking, the quiet latency thief

Assume you are admitting sensibly and requests are waiting in line. The order you
serve them in now matters enormously, and the default order — first in, first out —
has a specific pathology named head-of-line blocking [R19]. It happens whenever a
slow item at the front of a line delays faster items behind it that could otherwise
have been served. The classic image is a single supermarket checkout: one shopper
with a full cart holds up five shoppers each holding a single item, even though those
five together would clear in less time than the one. The five are blocked not because
the server is busy with useful work but because they are stuck behind the wrong
request.

Serving is full of head-of-line blocking, and it wears several disguises. A long
prompt prefilling at the front of a batch blocks the decodes behind it, which is the
interference from Chapter 4 seen from the queue's side. A request generating three
thousand tokens holds a slot that three hundred short requests could have cycled
through. A single slow request — one that spilled to slow memory, or hit a
pathological input — drags the pace of the whole batch it rides in, a problem Chapter
7 examines in detail. In each case the resource is not overloaded; it is
mis-scheduled, held hostage by one request at the head of a line.

The listing makes the effect and its cure visible. It runs three hundred requests
through a four-slot server under stable load, where most requests are short but about
eight percent are very long, and compares first-in-first-out admission against a
policy that lets shorter requests jump the line.

```python
# Head-of-line blocking: one long request stuck at the front delays everyone
# behind it. Same workload, two admission policies, measured queueing delay.
import random
random.seed(7)
SLOTS = 4                 # concurrent decode slots
GAP = 8                   # a new request arrives every GAP time-units
N = 300
jobs = []
t = 0
for i in range(N):
    length = 8 if random.random() < 0.92 else 200   # 8% are long
    jobs.append({"id": i, "arrive": i * GAP, "len": length})

def simulate(size_aware):
    active = []           # dicts with "left"
    pending = list(jobs)
    done = []
    clock = 0
    while pending or active:
        # admit whatever fits, from the head of the queue (FIFO) or shortest-first
        ready = [j for j in pending if j["arrive"] <= clock]
        if size_aware:
            ready.sort(key=lambda j: j["len"])
        for j in ready:
            if len(active) >= SLOTS:
                break
            j["start"] = clock
            j["left"] = j["len"]
            active.append(j)
            pending.remove(j)
        # advance every active slot by one token
        for j in active:
            j["left"] -= 1
        for j in [x for x in active if x["left"] <= 0]:
            j["wait"] = j["start"] - j["arrive"]
            done.append(j)
        active = [x for x in active if x["left"] > 0]
        clock += 1
    waits = sorted(j["wait"] for j in done)
    pick = lambda q: waits[min(len(waits) - 1, int(q * len(waits)))]
    return pick(0.50), pick(0.99), max(waits)

for name, sa in (("FIFO", False), ("shortest-first", True)):
    p50, p99, mx = simulate(sa)
    print(f"{name:>14}: queue-wait p50={p50:3d}  p99={p99:4d}  max={mx:4d}")
```

```output
          FIFO: queue-wait p50=  0  p99= 160  max= 160
shortest-first: queue-wait p50=  0  p99= 112  max= 192
```

The median wait under both policies is essentially zero — the system is not
overloaded, and a typical request sails through. The story is entirely in the tail.
Under first-in-first-out, the ninety-ninth-percentile wait is a hundred and sixty
time units, inflated by short requests stuck behind the occasional long one. Letting
short requests pass trims that tail wait to a hundred and twelve — but look at the
maximum: it rose from a hundred and sixty to a hundred and ninety-two. Shortest-first
made the common case better by making the worst case worse. The long requests, always
passed over, now wait even longer than they did under fair ordering. This is the
central, unavoidable trade of scheduling: you can move latency around between classes
of request, but you cannot make it vanish. Every policy is a decision about *whose*
latency to protect, and there is no policy that protects everyone's at once.

## Tail latency is the number that matters

Notice that the medians told us nothing. Both policies delivered a median wait of
zero, and if you had judged the server by its average or its median you would have
called both perfect and missed the entire difference between them. This is the single
most important habit in reading a server under load: the average is a comforting lie,
and the tail is the truth [R20]. Users do not experience the average. A user issues
many requests over a session, and their experience is dominated by the worst ones —
the request that took ten times as long as the others is the one they remember and
the one that breaks the interaction. In a system where a page render waits on several
model calls, the slowest call sets the page's speed, so the more calls a user makes,
the more certainly they meet your tail. Reporting a serving system's latency as a
single average is not just imprecise; it actively hides the behavior that determines
whether the system is usable.

Always look at percentiles — the median, the ninety-ninth, and the maximum — and look
at them under the load you actually expect, not at idle. A server measured with one
request at a time will show you a beautiful, tight latency that has nothing to do with
how it behaves when forty callers arrive at once. The tail is an emergent property of
contention, and contention only exists under load, so a latency number taken without
load is not a smaller version of the truth; it is a different quantity that happens to
share a unit.

## Fairness when the callers are not one caller

The simulation treated every request as interchangeable, differing only in length. Real
servers rarely have that luxury, because the requests come from different callers with
different claims on the machine, and the queue is where those claims are honored or
betrayed. A single server might carry a paying customer's interactive traffic, an
internal batch job, and a free tier all at once, and first-in-first-out treats them
identically — which means the batch job's thousand queued items can bury the paying
customer's one urgent request behind an hour of work that nobody was waiting on. Left
alone, a shared queue lets the heaviest user set everyone else's latency.

The defense is priority, but priority is a loaded gun. A strict priority scheme that
always serves the high-priority class first will, under sustained high-priority load,
*starve* the low-priority class completely — its requests wait forever because there is
always something more important. That is sometimes what you want and usually not, and
the middle grounds are where the craft lives: weighted fair sharing, where each class
gets a guaranteed fraction of the capacity so no one starves; rate limiting per caller,
so no single tenant can flood the queue and crowd out the rest; and reserved capacity,
where a slice of the slots is held for a class no matter how busy the others are. Each
of these is a way of deciding, in advance and explicitly, whose latency you protect
when the machine is contended — which is the same decision the shortest-first
experiment made, now made along the axis of who is asking rather than how big the ask.

Underneath all of these sits the service-level objective, the promise you are actually
trying to keep. "Ninety-nine percent of interactive requests get their first token
within one second" is a real objective, and it is measurable, and it tells the
scheduler what to optimize. Without a stated objective, you cannot even say whether a
scheduling policy is good, because good is defined relative to a promise. The most
common failure here is an unstated objective: a team runs a shared server, never
decides whose latency matters, and discovers under load that the answer the system
chose by default — first come, first served, heaviest user wins — is not the answer
anyone would have chosen on purpose. Decide the objective, then choose the policy that
serves it, then measure whether the policy keeps the promise. In that order.

A last, easily missed point: a request that will miss its objective anyway is often
better dropped than served. If an interactive request has waited past the point where
its answer is still useful — the user has given up, the page has moved on — then serving
it consumes a slot and a weight read to produce something no one will read, while
delaying a request that could still be served in time. Deadline-aware scheduling, which
sheds requests whose objective is already lost, protects the requests that can still be
saved. It feels wasteful to discard work you could complete, but completing work whose
value has expired is the deeper waste, because it is paid for out of the capacity that
would have kept the next promise.

## What a queue reveals when you watch it over time

A queue is also a diagnostic instrument, and its length over time tells you things no
single latency number can. A queue that is usually empty and occasionally spikes is a
healthy server meeting bursty demand — the spikes are the bursts, the emptiness
between them is headroom. A queue that is never empty is a server running at or past
capacity, and its latency is only going to grow; Little's law guarantees it. A queue
whose length grows monotonically and never recovers is a server in the process of
falling over, and the time to shed load was several minutes ago. Watching queue depth
is often a better early warning than watching latency, because queue depth turns up
before latency does — the requests are piling up before they have finished waiting
long enough to show in your percentiles.

The bench offers a small but honest picture of what healthy behavior looks like when
you watch it long enough to trust it. A production configuration of the
mixture-of-experts server, run at two concurrent slots for a twelve-minute soak,
handled eighty-six requests with zero errors, held full draft acceptance throughout,
and drifted its memory use by two megabytes over the entire run. That last number is
the interesting one: a server that leaks memory or fragments its cache under
sustained load will show a slow, monotone climb in memory that eventually crosses the
cliff from Chapter 3, and it will do so only after minutes or hours, invisible to any
short test. Two megabytes of drift over twelve minutes is the signature of a system
that is genuinely stable, not merely stable for the length of a demo. The soak test —
running real load for long enough that slow leaks and creeping queues have time to
appear — is the only test that catches this class of problem, and Chapter 8 makes it
a required step.

## Backpressure: telling the source to slow down

There is one more thing a queue can do that a naive queue does not, and it is the
difference between a system that bends under overload and one that shatters.
Backpressure is the queue pushing back on whoever is feeding it — signaling upstream to
slow down, or refusing new work — when it is falling behind, rather than silently
accepting everything and letting the backlog grow. A queue without backpressure is an
unbounded buffer, and an unbounded buffer under sustained overload does the worst
possible thing: it accepts every request, so nothing is refused and everyone waits,
with the wait growing without limit exactly as Little's law predicts. The requests at
the back of such a queue often wait so long that the caller has given up before they are
served, so the server spends its scarce capacity producing answers nobody is waiting for
anymore, which deepens the backlog further. This is the classic overload collapse, and
its cause is almost always a missing bound on the queue.

The cure is to bound the queue and act when the bound is reached. A bounded queue that
rejects new work when full converts a slow-motion collapse into a fast, honest signal:
callers learn immediately that the server is saturated and can back off, retry later, or
shed their own load, instead of hanging on a request that will never arrive in time.
The rejection feels like a failure, but it is the system telling the truth about its
capacity, and a truthful refusal is worth far more than a dishonest acceptance. Systems
that propagate this signal upstream — a client that slows its send rate when it sees
rejections, a load balancer that stops routing to a saturated instance — turn the whole
pipeline into one that degrades gracefully, each stage protecting the next from more
than it can bear.

Retries deserve a specific warning here, because they are where good intentions make
overload worse. When a server is struggling and starts rejecting or timing out, a client
that immediately retries doubles the load precisely when the server can least afford it,
and if many clients retry in unison they synchronize into a thundering herd that turns a
brief hiccup into a sustained outage. The discipline that prevents this — retrying with
increasing delays and a little randomness so the retries spread out rather than pile up,
and giving up after a bounded number of attempts — is not the server's to enforce, but it
is the server operator's to demand of the clients, because a server with perfect
admission control can still be knocked over by clients that retry without restraint. The
queue can only protect the server from honest load; protecting it from its own clients'
panic is a contract that has to be agreed on both sides.

The queue, then, is where the server's promises are kept or broken. It is where
admission control decides whether a spike becomes graceful degradation or a
collapse, where scheduling decides whose latency you protect, and where the tail —
the number that actually governs the user's experience — is set. Get the batch and
the cache right and you have a fast server; get the queue right and you have a server
people can rely on. But there is a subtler way a server can betray trust, one that
has nothing to do with speed and everything to do with load, and it is strange enough
to deserve a chapter of its own.
