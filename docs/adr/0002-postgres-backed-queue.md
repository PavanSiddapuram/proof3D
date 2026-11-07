# ADR 0002: Postgres-backed stage queue

**Status:** accepted, 2026-11-07

## Context
Stages run on two kinds of worker: one rented GPU, and CPU workers. Throughput is a few products an hour. Every stage needs its status, timing, cost and error stored next to the job anyway.

## Options
1. **Postgres table + `FOR UPDATE SKIP LOCKED`.** No new service. Queue state and job state are the same rows, so they cannot disagree.
2. **Redis + RQ/Celery.** A mature retry and scheduling toolkit, but one more service, and job state split across two stores.
3. **A cloud queue (SQS).** Ties local development to a cloud account.

## Decision
Option 1. The `stage_runs` table is the queue. Exactly one stage per job is `queued` or `running`. Workers claim by worker kind.

## Consequences
- One less moving part in Docker Compose and in CI.
- Polling adds about 1 s of latency per stage, which is negligible next to stages that take minutes.
- If throughput ever needs many GPUs, revisit (LISTEN/NOTIFY first, then Redis).
