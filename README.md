# Azul: Summer Pavilion AI

A reproducible **2–4 player** simulator for the normal (colored-star) side of **Azul: Summer Pavilion**, plus a masked-action PPO trainer that learns through two-player neural-network self-play on an NVIDIA GPU. Neural observations, training, evaluation, and the human-versus-AI CLI are intentionally two-player; the rules engine supports the official 5/7/9-factory player counts. The monochrome-board variant is not implemented.

The engine implements all six rounds, the rotating wild color, factory and center drafting, first-player penalty, separate placement turns, natural/wild payment choices, connected-star scoring, the normal board's transcribed orange→red→blue→yellow→green→purple architectural geometry, supply bonuses for all 18 windows/statues/pillars, passing and keeping up to four tiles, bag/tower recycling, and final star/number bonuses. It intentionally does not reproduce the publisher's artwork.

## Quick start

```powershell
python -m pip install -e .
pytest

# Highest-scoring learned policy (raw neural play)
python -m azul.cli play --checkpoint checkpoints/best_score.pt --mode neural

# Strongest measured head-to-head mode (policy plus tactical guardrails)
python -m azul.cli play --checkpoint checkpoints/best_competitive.pt

# Or play the built-in strategic baseline immediately
python -m azul.cli play --difficulty heuristic
```

## Play in the web interface

From the project folder, start the local game server:

```powershell
python -m azul.web
```

Then open <http://127.0.0.1:8000> in a browser. The opponent menu includes the competitive hybrid, raw score and competitive neural policies, score hybrid, final-round rollout search, the experimental multi-star policy, the strategic heuristic, and random play. The interface runs entirely on this computer and uses the same rules engine and checkpoint files as the command-line game.

`checkpoints/best.pt` and `checkpoints/latest.pt` currently alias the high-score policy. The separate `best_competitive.pt` is preserved because its tactical hybrid remains the stronger match-play opponent.

The CLI abbreviates colors in inventories (`p`, `g`, `o`, `y`, `b`, `r`) and lists every legal move. When placing a tile, “using N natural” is meaningful because you may choose how many current-round wild tiles to spend. One natural tile is always required unless placing the wild color on its own matching star.

## Score-maximizing self-play

The high-score model was trained with undiscounted Monte Carlo returns. For each player, the sum of the episode rewards is exactly `(final score - initial score) / 20`; this avoids weakening six-round final bonuses through discounting.

```powershell
python -m azul.training --iterations 20 --games-per-iteration 512 `
  --batch-size 8192 --ppo-epochs 4 --objective score `
  --gamma 1 --gae-lambda 1 --shaping-weight 1 `
  --output checkpoints/score_run --resume checkpoints/best_score.pt
```

The trainer automatically chooses CUDA when available. Every generated game has the current neural policy playing both seats. PPO uses legal-action masks, and checkpoints plus JSONL metrics are written after every iteration. On the installed RTX 3080, 512 simultaneous games and 8192-sample PPO batches provide the best measured throughput; simulator-side action generation remains the limiting part of utilization.

Run the fixed-seed score gate or alternating-seat opponents with:

```powershell
python -m azul.cli score checkpoints/best_score.pt --games 500
python -m azul.cli evaluate checkpoints/best_score.pt --opponent random --games 300 --mode neural
python -m azul.cli evaluate checkpoints/best_competitive.pt --opponent heuristic --games 100
```

The score winner averages **101.077** over the fixed 500-game self-play gate, up from **89.149** for the prior competitive checkpoint and **99.268** for the first score champion. It also averages **114.88** against random play and won all 300 games. See `training_report.json` for distributions and match-play results.

## Theoretical ceiling and Pareto frontiers

The exact mixed-integer board optimizer enumerates all normal-board construction patterns, optimal connected placement order, final bonuses, and all 18 architectural rebates:

```powershell
python -m azul.theory --output theoretical_frontier.json
```

The absolute printed-board ceiling is **304 points**: 5 starting + 147 placement + 152 final bonuses. Filling all 42 spaces costs 147 tiles gross and triggers 36 architectural reward tiles, for a relaxed net construction cost of 111. This proves the board ceiling, but does not prove a six-round drafting sequence can attain it.

The generated report contains 83 exact single-board score/cost breakpoints and 57 nondominated two-player upper-bound pairs. With the two-player game's 120 shared factory tiles, the relaxed frontier ranges from 304–18 at a 111/9 net split to 147–147 at 60/60. It deliberately relaxes color availability, round timing, supply timing, and alternating draft turns, so it is an optimistic upper bound.

Sample a complementary lower bound using only fully legal games:

```powershell
python -m azul.empirical_frontier --checkpoint checkpoints/best_score_v1.pt `
  --checkpoint checkpoints/best_score.pt --games 1000 --device cuda `
  --output empirical_frontier.json
```

Across 8,000 sampled score-pair orientations, the observed frontier reached an individual maximum of 142, a combined maximum of 250 (139–111), and a best balanced result of 113–113. More search can expand this empirical frontier; it cannot invalidate the theoretical ceiling.

## Exploration curriculum

Pure score optimization initially converged on a center-star strategy and never completed an outer star. Elite-only/CEM training and a direct outer-star auxiliary reward did not solve that exploration failure. The successful route was a short outer-score teacher curriculum followed by exact-score self-play recovery:

```powershell
python -m azul.curriculum --base checkpoints/best_competitive.pt `
  --teacher outer-score --output checkpoints/outer_seed.pt --games 120 --epochs 12
python -m azul.training --iterations 30 --games-per-iteration 512 `
  --objective score --gamma 1 --gae-lambda 1 --shaping-weight 1 `
  --batch-size 8192 --ppo-epochs 4 --output checkpoints/outer_recovery `
  --resume checkpoints/outer_seed.pt
```

The final exact-score phase—not the teacher reward—determines whether the seeded behavior survives. In the winner, it did: outer-star completions rose from zero to 0.437 per player while mean score increased.

## Strategy telemetry

```powershell
python -m azul.analysis --checkpoint checkpoints/best_score.pt `
  --agent neural --opponent neural --games 100
```

The high-score policy averages 98.26 in this independent 100-game profile. It completes the purple outer star in 40% of player-games, all cost-1/2/3 spaces in 98%/100%/93%, and all cost-4 spaces in 28%. `STRATEGY.md` describes the learned regime shift and practical lessons.

## Project layout

- `src/azul/game.py` — deterministic state machine, legal action mask, scoring, observations, invariants.
- `src/azul/agents.py` — random, heuristic, score heuristic, and checkpoint-backed neural agents plus evaluation.
- `src/azul/training.py` — batched GPU self-play, exact-score objective, and PPO updates.
- `src/azul/curriculum.py` — optional balanced or outer-score curriculum distillation.
- `src/azul/score_cem.py` — reproducible elite-score experiment (not the winning method).
- `src/azul/dagger.py` — adversarial state aggregation for correcting neural-visited states.
- `src/azul/analysis.py` — empirical strategy telemetry from simulated games.
- `src/azul/theory.py` — exact board optimizer and relaxed resource Pareto frontier.
- `src/azul/empirical_frontier.py` — observed legal-game score-pair frontier.
- `src/azul/cli.py` — interactive play, opponent evaluation, and fixed-seed score gate.
- `src/azul/web.py` / `src/azul/web_static/` — local 1v1 browser interface and JSON game API.
- `training_report.json` — hardware, run sizes, score distributions, and evaluation evidence.
- `theoretical_frontier.json` / `empirical_frontier.json` — full upper/lower frontier data.
- `STRATEGY.md` — learned strategy, stronger/weaker patterns, and caveats.
- `tests/` — rule cases, geometry, observations, conservation, and randomized full-game tests.

## Rules reference

Implementation was checked against the supplied English rulebook and the photographed normal board. The downloaded PDF is excluded from version control because the game and artwork remain their owners' property.

This is an independent research/education project and is not affiliated with Plan B Games or Next Move Games.
