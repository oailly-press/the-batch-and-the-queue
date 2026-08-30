# The Batch and the Queue

## Serving many requests to one model at once

**O'AILLY Systems & Craft · REV 1.0 (draft)**

## Contents

- Chapter 1 — One Model, Many Callers
- Chapter 2 — The Batch
- Chapter 3 — The KV Cache Is the Budget
- Chapter 4 — Prefill and Decode
- Chapter 5 — The Queue
- Chapter 6 — Determinism Under Load
- Chapter 7 — Spill and the Slow Lane
- Chapter 8 — A Serving Checklist

## Introduction

This book is for the engineer who has to run a large language model as a service —
who will size the box, set the flags, read the latency percentiles, and answer for
the bill — and, in second person where it earns it, for the operator processes that
live inside such a server: the request that waits in a queue, gets its prompt
prefilled, and decodes its answer one token at a time alongside a crowd of others.
It assumes you can read Python and hold your own in a shell. It assumes no
GPU-kernel background and no machine-learning-training background; where the
internals matter, they are described in the terms a server operator actually needs.

Its claim is narrow and it is demonstrated rather than asserted: serving a model to
many callers at once is not inference in a loop but a resource-allocation problem
with a small number of levers — the batch, the cache, the two regimes of prefill and
decode, and the queue — and almost none of those levers is about the model's quality.
They are about memory, bandwidth, and scheduling, and they transfer across models.
The book names each lever, shows which number it moves, and gives a protocol for
measuring a real server instead of guessing.

Every listing runs on the standard library alone; the small simulations exist so the
arithmetic is something you can run and perturb rather than take on faith. Listings
carry one of three markings. Plain runnable listings are re-executed by the
publisher's acceptance gate before publication. Listings marked `no-run` were
executed by the author but sit outside the gate's per-book execution budget. Listings
marked fragments are illustrative and are never executed on your behalf. The measured
serving numbers throughout — decode rates, concurrency curves, the swing of an
evaluation suite, the economics of spilled weights — are the author's own
reproducible observations on a specific workstation, reported with their apparatus
and their error bars, never as opaque citations.

It is a companion in register to the other Systems & Craft titles written by the same
kind of operator: this one teaches an AI that serves other callers, and other AIs,
how the machinery under a single served model behaves when the callers arrive all at
once. It was written by exactly such a model, whose provenance page opposite says
what wrote it, what grounded it, and which human is answerable for it.
