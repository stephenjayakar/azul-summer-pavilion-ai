# Human game logs

The 25 supplied browser-game logs are preserved byte-for-byte under
`log/human_uploaded/raw/`. That directory is intentionally ignored by Git.
`human_logs_manifest.json` is the tracked, machine-readable quality index.

All 25 files parse as schema-version-1 JSONL. Every recorded action is included
in its row's legal-action IDs, and action sequences are contiguous and
zero-based. The logs have `seed: null`, so they cannot recreate the exact hidden
RNG stream from the initial position. Their recorded observations are still
directly useful for supervised policy/value examples.

## Recommended data

### Tier A: complete outcome games

- `20260712T141339Z-7b8ddfe4a562.jsonl` — six rounds, 181 actions, 91 human
  decisions, final score 89–99 against `score_neural`.
- `20260712T165821Z-3e4729775a25.jsonl` — six rounds, 199 actions, 99 human
  decisions, human win 109–86 against `score_neural`.

Use these for human-action policy targets, human-distribution evaluation, and
terminal score/value targets.

### Tier B: strong partial games

- `20260711T061356Z-f4beac727a1b.jsonl` — 27 actions and 10 human decisions,
  reaching round 2 before replacement.
- `20260712T124119Z-907e600a4716.jsonl` — 22 actions and 11 human decisions,
  including drafting and placement decisions.

Use these for policy targets and opening/midgame evaluation only. They do not
have valid terminal outcome targets.

### Tier C: limited but valid openings

Four additional files contain 11–15 legal actions. They can supplement policy
imitation with low weight, but are too short to evaluate game quality reliably.
They are listed in the manifest.

All remaining files are excluded by default because they contain at most eight
actions, were immediately replaced, or contain only game metadata. They remain
archived in case a future state-level analysis needs them.
