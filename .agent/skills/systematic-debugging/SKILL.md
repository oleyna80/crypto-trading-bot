---
name: systematic-debugging
description: Diagnose and repair a reproducible defect in bybit_grid_bot through evidence, a narrow hypothesis, and a regression test. Use for failing tests, incorrect strategy behavior, or provider parsing faults after a scoped fix is approved.
---

# Systematic Debugging

Turn a reported symptom into a verified root cause and a minimal repair. Start
as read-only analysis; write only after the Owner or Orchestrator approves a
bounded fix work block.

## Workflow

1. Capture the symptom, expected behavior, reproduction command and affected
   ticket. Preserve the original failure output.
2. Read the nearest tests and implementation; trace inputs, outputs, branches
   and external-provider assumptions.
3. State one falsifiable hypothesis and test it with the smallest diagnostic.
   Do not apply speculative fixes or stack unrelated changes.
4. Once the root cause is supported, obtain/confirm the fix write-set.
5. Add a regression test that fails before the repair and passes after it.
6. Run the focused test and relevant smoke-check; report the evidence and any
   remaining uncertainty.

## Escalation

After three disproved hypotheses, stop and return the evidence, eliminated
causes and the smallest useful next investigation. Do not widen the scope to
dependencies, provider configuration, secrets or deployment without approval.
