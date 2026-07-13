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

Then open <http://127.0.0.1:8000> in a browser. The opponent menu includes the verified AZ2 search champion, competitive hybrid, raw score and competitive neural policies, score hybrid, final-round rollout search, the experimental multi-star policy, the strategic heuristic, and random play. The interface runs entirely on this computer and uses the same rules engine and checkpoint files as the command-line game.

`checkpoints/best.pt` and `checkpoints/latest.pt` currently alias the high-score policy. The separate `best_competitive.pt` is preserved because its tactical hybrid remains the stronger match-play opponent.

## AI opponents

The opponent menu contains several policies with different goals. “Hybrid” means a hand-written, rules-aware heuristic first narrows the legal moves to strategically sensible candidates, then a neural network chooses among close alternatives. “Neural” means the network chooses directly from the legal-action mask. “Score” policies optimize the player's own final score; “competitive” policies were trained or selected for head-to-head play.

| Opponent | How it plays | Best for |
| --- | --- | --- |
| **AZ2 search champion** | Structured policy plus persistent 128-simulation stochastic search, a tactical root prior, and calibrated value guidance. It is the strongest verified match-play mode, but much slower. | The hardest research challenge |
| **Competitive hybrid** | Tactical guardrails plus the competitive neural policy as a tie-breaker. It avoids obvious resource mistakes while responding quickly. | The default fast challenge |
| **Score champion** | A pure neural policy trained to maximize personal score and final bonuses. It is ambitious about building a high-scoring pavilion and is less explicitly defensive. | Seeing the highest-scoring style |
| **Score champion + search** | The score champion, with deterministic full-game rollout search added in the final round. It can make sharper endgame decisions, at the cost of more thinking time. | A slower, stronger endgame |
| **Score hybrid** | The score-focused neural policy constrained by tactical guardrails. It keeps the score objective while filtering out obviously poor trades. | A score-oriented middle ground |
| **Competitive neural** | The raw competitive neural policy without the hand-written tactical filter. It is fast and direct, but can make less robust tactical choices. | Comparing raw learned play |
| **Experimental multi-star** | A research policy seeded to develop multiple outer stars rather than concentrating on one dominant scoring plan. It is intentionally not fully tuned. | Variety and experimentation |
| **Strategic heuristic** | A fast, transparent hand-written baseline that values immediate scoring, connected placements, architectural rewards, and useful drafts. | Quick games and a readable baseline |
| **Random** | Uniformly selects among legal moves. It knows the rules but has no strategy. | A gentle demo or rules testing |

The learned policies are not “difficulty levels” in a perfectly ordered ladder: they optimize different objectives. For a normal game, start with **Competitive hybrid**; choose **AZ2 search champion** for the strongest measured opponent when longer thinking time is acceptable, use **Strategic heuristic** for speed, **Random** for a relaxed introduction, and the score policies when you want to watch pavilion-building and bonus-chasing behavior.

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

## Experimental policy plus search self-play

`azul.alphazero` provides a batched PUCT loop modeled on AlphaZero: freeze the
policy/value network, use search visit counts as improved policy targets, finish
the self-play games to obtain score-margin value targets, and train only between
search generations. Updating weights inside an active tree is intentionally not
supported because it would make stored priors and values inconsistent.

When starting from a score-objective PPO checkpoint, the trainer first preserves
the policy and recalibrates only its value head for competitive margins. An
optional alternating-seat arena rejects candidate updates that do not beat the
incumbent.

```powershell
python -m azul.alphazero --resume checkpoints/best_score.pt `
  --output checkpoints/alphazero_run --iterations 8 `
  --games-per-iteration 32 --simulations 8 --learning-rate 2e-5 `
  --epochs 1 --replay-iterations 2 --arena-games 64

# Deploy policy plus search; 8 is interactive, 32 is slower research mode.
python -m azul.cli play --checkpoint checkpoints/best_search_policy.pt `
  --mode puct --simulations 8 --heuristic-prior-weight 0.5
```

The first search experiment is recorded in `search_training_report.json`. The
selected raw policy improved from **33.5% to 37.7%** against the strategic
heuristic over 300 alternating-seat games while retaining a **100.302** mean in
the fixed 500-game score gate (the score champion is 101.077). This is a
measurable but still modest raw-policy gain. The recommended deployed
policy-plus-search configuration is much stronger: with 8 simulations and a
50/50 learned/tactical root prior, it won **55.8%** over 300 games with a
**+1.52** average margin, versus 33.5%/−9.50 for the original raw policy and
42.5%/−2.08 for the existing hybrid. A full heuristic prior is harmful; the
measured benefit comes from combining complementary priors with value search.
The original mode remains available for comparison. The second-generation
search below removes deepcopy from the simulation loop and handles future draws
as sampled chance boundaries.

## Structured policy + stochastic search (AZ2)

`azul.az2` adds compact snapshot/restore, batched low-copy PUCT, subtree reuse,
sampled factory/supply chance outcomes, four player-oriented value heads, and a
structured residual transformer. `PLAN.md` records the seven-part design and
acceptance evidence. Training freezes a network during each complete search
generation and updates it only between generations; changing weights inside a
live tree would invalidate its cached priors and values.

The fast interactive deployment point remains 8 simulations, a 50/50 learned
and tactical root prior, and 25% neural-value influence. On the same 128 seeds
against the strategic heuristic it scored **57.42% with a +1.72 margin**, versus
54.30%/+0.59 with value disabled and 49.22%/-1.38 with full value influence.
This is why simulation depth and value weight are gated empirically rather than
assumed to improve monotonically. A 64-game candidate promotion that failed a
fresh 128-game confirmation also led to a mandatory second confirmation arena.

The first long run confirmed that simply training longer was not enough: nine
complete generations produced no promotion, held-out value error drifted, and
the only preliminary arena pass failed its independent confirmation. Its
8-simulation targets averaged only 3.50 non-zero actions and were one-hot in
18.84% of positions. The replacement uses a two-stage schedule:

1. Calibrate value offline on complete replay generations with a direct
   observation-to-value residual; freeze policy and select the value checkpoint
   by search arenas, not regression loss alone.
2. Freeze outcome-value training and distill a stronger 32-simulation search
   teacher into the shared policy network between complete self-play
   generations. Search targets contain no Dirichlet-noise mixture.

The selected value stage beat the uncalibrated value checkpoint by 74-53-1
(58.20%) across two independent 32-simulation arenas. With that evaluator,
32-simulation teacher targets became substantially denser (5.23 non-zero
actions on average, 9.28% one-hot). The selected policy candidate went 101-87-4
(53.65%) against its parent across three independent arenas. These samples are
promising rather than proof of convergence, so the durable run retains paired
arena, confirmation, score, human, fixed-reference value, and league gates.

The first replacement generation was also stopped after its candidate tied its
parent at 31-32-1. The successful curriculum used a smaller, safer update:

- generate one fresh, noise-free 32-simulation generation with a balanced 50%
  tactical root prior;
- sharpen visits with temperature 0.5 and mix them 50/50 with the visit-argmax
  action;
- train only the residual policy head for one epoch at `1e-4`, leaving every
  value output bit-exact;
- deploy with persistent subtree reuse, a 75% tactical prior, 25% value utility,
  and 128 simulations.

The resulting checkpoint is `checkpoints/best_az2_search.pt` (SHA-256
`33C83E5FEA14B81893A3677E6249646E78DBFDBC5EDF589D25E7813BF9DE403E`). It
beat the strategic heuristic **83-44-1, or 65.23%, with a +7.34 margin** over a
fresh 128-game alternating-seat confirmation. The disjoint 64-game preliminary
was 42-20-2 (67.19%, +10.97); combined evidence is 125-64-3 (65.89%). Its raw
500-game score mean is 100.407 versus 100.232 for its parent, human top-1 is
unchanged at 38.86%, and value outputs are bit-for-bit identical.

The accepted policy update itself is reproducible from the preserved replay:

```powershell
python -m azul.az2_policy_distillation `
  --base training_runs/az2_long_v2/starting.pt `
  --replay training_runs/az2_long_v2/replay/generation_0001.npz `
  --output training_runs/az2_curriculum/champion_65.pt `
  --epochs 1 --patience 1 --learning-rate 1e-4 --batch-size 2048 `
  --target-temperature 0.5 --hard-weight 0.5
```

Reproduce the independent confirmation (about 18 minutes on an RTX 3080):

```powershell
python -m azul.az2_evaluate checkpoints/best_az2_search.pt `
  --games 128 --score-games 500 --simulations 128 `
  --heuristic-prior-weight 0.75 --value-utility-weight 0.25 `
  --seed 1530000 --output training_runs/az2_curriculum/verification.json
```

Play against the same deployed configuration with:

```powershell
python -m azul.cli play --checkpoint checkpoints/best_az2_search.pt `
  --mode az2 --simulations 128 --heuristic-prior-weight 0.75 `
  --value-utility-weight 0.25
```

Use 32 or 64 simulations for faster interactive play; 128 is the verified
65% research configuration.

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
