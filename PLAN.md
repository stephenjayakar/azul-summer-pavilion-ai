# AlphaZero v2 implementation and training plan

## Objective

Build a substantially stronger Azul: Summer Pavilion agent by implementing all
seven identified upgrades, prove improvement with reproducible short gates, and
then launch a durable long-running self-play job. The existing verified reference
is `checkpoints/best_search_policy.pt` with the following fixed results:

- raw neural policy vs strategic heuristic, 300 games: 37.67% win rate,
  -8.77 mean margin;
- policy + 8-simulation PUCT + 0.5 tactical root prior, 300 games: 55.83%
  win rate, +1.52 mean margin;
- raw fixed-seed self-play score, 500 games: 100.302 mean.

No v2 checkpoint is promoted as "improved" unless it clears the gates below.

## Design decision: search first, train between searches

The network is frozen for an entire self-play generation. PUCT stores priors
and values produced by that immutable snapshot. Training occurs only after the
games finish and policy/value targets are finalized. Updating the network inside
an active search would invalidate already stored node statistics and prevent
efficient batched inference.

## 1. Reversible, low-copy simulator path

Implement a compact `GameSnapshot` plus `snapshot()`, `restore()`, and trusted
`step_fast()` path in `game.py`. A PUCT simulation will keep one scratch game per
root, apply actions down a path, evaluate the leaf, and undo to the root. Tree
nodes store statistics/actions rather than deep-copied `AzulGame` instances.

Acceptance evidence:

- randomized parity tests show normal `step()` and fast apply/undo produce the
  same observation, legal actions, scores, RNG state, and terminal result;
- undo restores a byte-for-byte-equivalent state signature;
- a benchmark reports at least a material reduction in search time and clone
  count versus the current lazy-`deepcopy` PUCT.

## 2. Deep, GPU-batched PUCT

Run 100–400 simulations per move with batched leaf inference across games.
Use virtual visit accounting, PUCT priors, root Dirichlet noise for training,
temperature sampling early in games, and deterministic visits for evaluation.
Reuse the selected child subtree within a game when its state signature matches.

Acceptance evidence:

- legal normalized visit policies at 1, 100, and 400 simulations;
- turn-aware backup tests cover retained turns, switched turns, and terminal
  nodes;
- benchmark demonstrates that 100 simulations is practical for short training
  runs on the RTX 3080.

## 3. Multi-head two-player value prediction

Replace the scalar value with four outputs from the acting player's perspective:

1. own final score / 200;
2. opponent final score / 200;
3. clipped score margin / 50;
4. win/draw/loss target in [-1, 1].

PUCT combines predicted margin, score difference, and win value. This preserves
the general-sum scoring signal while still optimizing competitive play.

Acceptance evidence:

- target-orientation tests for both seats;
- finite multi-head losses and gradients;
- held-out value errors are written to every training metric row.

## 4. Chance handling for future factory draws

Do not let search inspect the live game's future RNG stream. Each simulation
uses a fresh determinization of future bag draws. When an action consumes RNG—
either across a round boundary or while refilling the architectural-bonus
supply—treat the resulting draw as a sampled chance leaf and average repeated
outcomes on the incoming edge. Do not reuse deterministic children beyond that
chance boundary.

Acceptance evidence:

- changing the live game's future RNG sequence does not change fixed-seed search
  results when the search determinization seed is held constant;
- repeated simulations across a round boundary observe multiple factory fills;
- same-round deterministic paths remain exactly reproducible.

## 5. Structured residual/attention network

Tokenize the current 282-element observation into:

- one global/pool token;
- five factory tokens;
- two player-summary tokens;
- twelve outer-star tokens;
- twelve center-slot tokens;
- six architectural-claim tokens.

Use segment projections, learned positional/type embeddings, residual attention
blocks, and separate policy/value heads. Bootstrap the new architecture by
distilling the verified checkpoint before self-play, rather than discarding the
existing learned strategy.

Acceptance evidence:

- exact observation-slice coverage test (all 282 inputs consumed once);
- checkpoint save/load round trip;
- teacher-policy agreement and held-out human-action likelihood are reported.

## 6. Persistent replay, opponent league, and human games

Persist compressed replay generations under the run directory and train from a
bounded window sampled across recent and older accepted generations. The league
contains the current incumbent, prior accepted checkpoints, the verified v1
checkpoint, score champion, and strategic heuristic/tactical-prior variants.

Human logs are indexed by `human_logs_manifest.json`:

- Tier A complete games contribute low-weight human policy examples and final
  score/value targets;
- Tier B strong partial games contribute low-weight policy examples only;
- Tier C openings are optional and down-weighted further;
- excluded logs never receive terminal targets.

Acceptance evidence:

- replay survives process restart and enforces its generation/example cap;
- league seat alternation is deterministic;
- human-log loader rejects malformed/illegal rows and never assigns terminal
  values to partial games.

## 7. Multi-gate promotion

A candidate is promoted only if all hard safety gates pass and the composite
score improves:

- candidate vs incumbent league arena: at least 52% over the configured gate;
- no material fixed-seed score collapse (initial floor: 98.5 mean over 200 games);
- heuristic match margin does not regress beyond its confidence allowance;
- human Tier A/B action cross-entropy does not regress materially;
- NaN/legality/invariant checks are clean.

Track, but do not overfit to, a composite of league Elo, heuristic margin,
fixed-seed score, value calibration, and human-action likelihood. Rejected
candidates restore both model and optimizer state.

## Short-run proof before long training

1. Distill the verified v1 policy into the structured model.
2. Benchmark shallow and deep budgets, then use the strongest measured point.
   The first calibrated sweep found that 8 simulations was stronger than 32
   while the value head was noisy. After offline value calibration, 32 became
   the stronger teacher and generated denser visit targets. Search depth is
   therefore a property of the policy/value pair, not a monotonic knob.
3. Run a small number of full v2 generations.
4. Compare on identical seeds against the v1 reference, heuristic, league, score
   gate, and human log gate.
5. Launch the long run only after at least one accepted candidate improves the
   composite gate without violating score/human floors.

## Long run and verification

The replacement long run writes to `training_runs/az2_long_v2/`:

- `config.json` — immutable run configuration;
- `metrics.jsonl` — generation, search throughput, losses, gates, promotion;
- `replay/` — bounded compressed generations;
- `league.json` — accepted checkpoint roster and ratings;
- `latest.pt` and `best.pt` — resumable and best accepted checkpoints;
- `run.log`, `pid.json`, and `heartbeat.json` — process monitoring.

Verification commands will be recorded verbatim in the README and final handoff.
At minimum they will include process/heartbeat checks, live metric tailing, fixed
arena reproduction, score gate, human-log evaluation, and clean resume after a
controlled short interruption.

## Calibration evidence (2026-07-12)

- compact snapshot/restore is 13.18x faster than deepcopy in a 5,000-cycle
  microbenchmark;
- 16-game, 32-simulation self-play runs at 37.6-38.2 searched positions/second;
- 64-game, 8-simulation self-play runs at 137-140 searched positions/second;
- policy-distillation candidates at 32 and 8 simulations were rejected by the
  incumbent arena (40.6-43.0% and 41.8% respectively), despite preserving score;
- a value-only candidate improved the tiny human-value set but regressed on a
  separate 24,928-position validation set, so per-head validation is now hard;
- on the same 128 seeds against the strategic heuristic, 8-simulation search
  scored 54.30%/+0.59 with neural value disabled, 49.22%/-1.38 at full value,
  and 57.42%/+1.72 at 25% value influence.
- a value-only candidate passed a 64-game preliminary arena at 54.69%, but lost
  a fresh 128-game confirmation at 42.19%; it is excluded, and every future
  preliminary pass now requires a separate confirmation arena before promotion.

The original `training_runs/az2_long/` curriculum was stopped after nine
completed updates with zero promotions. Its arena win rates ranged from 44.53%
to 52.73%; the sole preliminary pass failed confirmation at 48.63%. Held-out
value MAE also drifted from roughly 0.3442 to 0.3452. More wall time on that
configuration was therefore unlikely to help.

The failure audit found that the 8-simulation replay targets averaged 3.50
non-zero actions, were one-hot 18.84% of the time, and had mean target entropy
0.884. Removing root noise alone did not fix transfer. Before recalibration,
32 simulations was worse than 8 on paired seeds because search trusted a nearly
constant value predictor.

A direct observation-to-value residual was then calibrated offline using eight
complete replay generations for training and two held-out generations. The
selected checkpoint reduced held-out weighted MSE from 0.1292 to 0.1063 and
beat the old value checkpoint 74-53-1 across two independent 32-simulation
arenas. Continuing calibration to a still lower regression error made play
worse, so value checkpoints are selected by search strength rather than loss.

With the selected evaluator, a noise-free 32-simulation teacher produced 5.23
non-zero target actions on average, only 9.28% one-hot targets, and entropy
1.066. Its shared-policy candidate went 101-87-4 (53.65%) against its parent
over three independent arenas while preserving fixed-seed score. Fully freezing
the backbone and heavy value anchoring both failed independent confirmation.

The replacement curriculum is consequently two-timescale:

1. Calibrate value offline on complete games, with policy frozen, held-out
   early stopping, and final selection by deep-search arenas.
2. Run fresh 32-simulation self-play generations with zero Dirichlet target
   contamination, train policy/shared features at `1e-5`, and disable value
   outcome loss. Compare value MAE to one fixed calibrated reference with a 2%
   allowance rather than allowing per-generation drift.

`training_runs/az2_long_v2/` starts from the arena-audited policy candidate and
retains the 25% value utility blend. Every promotion requires a preliminary
arena, an independent confirmation, the fixed score/value/human gates, and the
opponent league. Rejected generations remain useful search data, but model and
optimizer state are restored.

## Verified champion (2026-07-13)

The generic `az2_long_v2` update was stopped after its first complete candidate
tied its parent 31-32-1. Target density was healthy, but one full shared-network
epoch was still too disruptive. Subsequent ablations established:

- persistent subtree recovery across observed opponent moves is required for
  practical evaluation and strengthens later decisions without extra network
  updates;
- nonlinear value residuals reduced held-out error but failed independent
  game confirmation, so value regression alone remains non-promotable;
- many-epoch soft visit distillation improved KL but collapsed raw and searched
  play through distribution shift;
- one frozen-value policy-head epoch at `1e-4`, using temperature-0.5 visits
  mixed 50/50 with visit argmax, transferred reliably.

That conservative candidate beat its parent 74-48-6 (60.16%) in a fresh
128-game 32-simulation arena. At the final deployment settingâ€”128 simulations,
75% tactical root prior, and 25% neural-value utilityâ€”it scored 83-44-1
(65.23%, +7.34 margin) against the strategic heuristic on seed block 1,530,000.
A disjoint 64-game preliminary scored 42-20-2 (67.19%, +10.97), giving combined
evidence of 125-64-3 (65.89%).

The champion is aliased as `checkpoints/best_az2_search.pt`. Its raw fixed-seed
500-game score is 100.407 versus 100.232 for its parent, human top-1 is unchanged
at 38.86%, and all value outputs are bit-exact. This is the current accepted
policy-plus-search configuration; failed nonlinear-value, over-distilled,
75%-teacher, repeated-balanced-teacher, and opponent-conditioned candidates are
retained only as negative research artifacts.
