---
name: security-audit-triage
description: Triage suspected security weaknesses in Python, configuration, provider integration, and alert paths of bybit_grid_bot. Use for a requested security review or before a risky integration; operate read-only unless a separate fix scope is approved.
---

# Security Audit Triage

Assess security-relevant evidence without exposing secrets or changing code.
This procedure triages engineering controls; it is not a substitute for a
financial, smart-contract or infrastructure audit.

## Review areas

- credential and `.env` handling, logging and error messages;
- HTTP/provider input validation, timeouts, retries and malformed payloads;
- unsafe URL construction, injection surfaces and untrusted deserialization;
- alert delivery, rate limits and accidental external side effects;
- dependency/configuration changes and whether tests avoid live credentials.

## Workflow

1. Define scope, threat assumptions and sensitive files excluded from reading.
2. Inspect code, tests and documentation; cite concrete `file:line` evidence.
3. Classify each candidate as `confirmed`, `partial`, `not confirmed` or
   `needs evidence`. Do not treat a missing test as a proven exploit.
4. Rank confirmed issues by impact and likelihood, and propose the smallest
   safe remediation scope plus verification.

## Boundaries

Do not print secrets, read full `.env` files, send live transactions, alter
external systems or apply fixes. Hand confirmed remediation to a separate,
approved Coder stage.
