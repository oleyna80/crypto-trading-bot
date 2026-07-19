---
name: architecture-discovery
description: Read-only discovery of module boundaries, data flow, provider contracts, and test seams in bybit_grid_bot. Use before planning a non-trivial feature, integration, migration, or architecture change.
---

# Architecture Discovery

Produce evidence for a plan without modifying repository files. Use it before
new DEX/LP capabilities, provider integrations, strategy changes or cross-layer
refactors.

## Workflow

1. Read `AGENTS.md`, memory-bank files, active-ticket artifacts and the
   relevant source/tests.
2. Map entry points, callers, data models, external provider boundaries,
   configuration inputs and failure paths.
3. Identify the smallest viable change boundary and existing test seams.
4. Separate confirmed facts from assumptions; cite file paths and symbols for
   every material conclusion.
5. Return a short implementation brief: impacted files, interfaces, risks,
   non-goals, acceptance criteria and proposed test commands.

## Project focus

For DEX LP work, distinguish provider payload parsing, scoring/strategy
decisions, range operations and alert delivery. Confirm that a proposed path
does not silently add trade execution, secrets handling or live network calls
to unit tests.

## Boundaries

This is a read-only Analyst procedure. It must not create code, alter a plan,
change dependencies or turn unverified provider behavior into a requirement.
