# Strategy findings

These findings combine fixed-seed score gates, fixed-opponent evaluation, and telemetry from the corrected simulator. They describe the normal colored-star board in two-player games. The learned policies are strong against their measured benchmarks, but are not claimed to be optimal.

## The score breakthrough

The previous competitive policy averaged 89.149 points in the deterministic 500-game self-play gate. Direct score PPO improved this only into the mid-90s and stayed entirely center-star focused. Elite-only self-imitation and a direct outer-star auxiliary reward also failed to discover a stable outer-star strategy.

A short outer-score demonstration curriculum supplied the missing exploration. Exact, undiscounted score self-play then discarded unproductive imitation while retaining the useful behavior. The winning checkpoint reached:

- **101.077 mean**, 104 median, 84 p10, 113 p90, and 139 maximum over 1,000 player results in the fixed 500-game gate;
- **+11.928 mean points (+13.38%)** and +14 median points over the competitive checkpoint;
- **+1.809 mean points** over the first 99.268 score champion;
- 0.459 outer-star completions per player, up from zero;
- 114.88 average score and a 300–0 record against random play.

This is a real policy change rather than merely higher variance: the 10th percentile improved by five points at the same time as the median and upper tail improved.

## Two useful checkpoints

The objectives produce different best agents:

- `best_score.pt` (also `best.pt` and `latest.pt`) is the highest-scoring raw neural policy. Use `--mode neural` when playing or evaluating it.
- `best_competitive.pt` is the prior match-play checkpoint. Combined with the default tactical guardrails, it remains the stronger measured human-facing opponent: 57–39–4 against the strong heuristic, versus 50–50 for the score model's hybrid.

The raw score policy also improved against the strong heuristic—from a 16.75% to a 36.25% match score—but maximizing one's own total is not identical to maximizing win probability or denying an opponent.

## Learned style of the score policy

Across an independent 100-game neural self-play profile, each player averaged:

- 99.59 points and 27.06 placed tiles;
- the purple outer star in 44% of games and no center-star completions;
- all cost-1/2 spaces in 100%, all 3s in 95%, and all 4s in 30%;
- 0.44 windows, 4.56 statues, and 2.47 pillars;
- 11.23 wild tiles spent, 13.19 tiles carried between rounds, and 2.95 first-player tokens.

The previous model completed the center star in 82% of games, no outer stars, no windows, and all 4s in only 7%. The score run therefore found a distinct purple/window/number-set economy rather than simply refining the previous center-heavy line.

## Better strategic principles

1. **Finish the cheap number sets.** All 1s, 2s, and usually 3s combine attainable final bonuses with progress toward statues and pillars. All 4s became more valuable once the policy learned a coherent outer-star plan; 5s and 6s still need a specific star or window target.
2. **Choose one realistic outer star.** The learned winner overwhelmingly chooses purple. Spreading expensive placements across several stars gives up the completion bonus and architectural rebate.
3. **Build connected runs, not isolated petals.** Every new tile scores its whole connected component. Growing a chain repeatedly scores two, three, or four points; scattering repeatedly scores one.
4. **Carry next round's wild color.** Keeping up to four tiles is frequently better than forcing inefficient placements. The score policy carries about 13 tiles over a game and spends about 12 wilds.
5. **Trigger bonuses early in the placement phase.** Supply choice is strongest before the opponent removes a needed color, and the reward tile can be spent in that same round.
6. **Coordinate architecture with end-game bonuses.** A window costs 11 tiles and returns three; a statue costs 10 and returns two; pillars cost 8–16 and return one. Windows become worthwhile when their 5/6 spaces also finish the chosen outer star. Statues naturally align with cheap-number goals.
7. **Treat first player as a price, not a goal.** The center penalty must be justified by both the draft and next-round initiative.
8. **Deny narrowly in match play.** Taking one critical color or supply tile can prevent an opponent's star or number bonus. Broad hate-drafting wastes your own development. The competitive checkpoint is better at this than the pure score winner.

## Common losing patterns

- spreading tiles across every star without a completion target;
- paying for 5s/6s before the associated star/window can realistically finish;
- forcing center completion when a coherent outer-star plus window line is available;
- spending next round's wild color on marginal current-round points;
- passing with more than four low-value leftovers and absorbing avoidable waste;
- taking a large first center group only for initiative;
- completing a feature late in the phase, after useful supply colors are gone.

## What did and did not work in training

- **Worked:** exact total-score rewards, `gamma=1`, `GAE lambda=1`, large GPU PPO batches, an exploration-only outer curriculum, then long exact-score recovery and a zero-entropy consolidation gate.
- **Did not work:** competitive rewards alone for absolute score, discounted final bonuses, elite/CEM self-imitation, or an outer-progress auxiliary reward without demonstrated outer completions.
- **Key lesson:** shaping can open a behavior basin, but only unshaped objective evaluation should promote a score checkpoint.

## Theoretical versus observed frontier

The exact printed-board ceiling is 304: 147 placement points plus 152 end bonuses and the five starting points. Its 147 gross payment tiles are offset by all 36 architectural reward tiles, giving a relaxed net cost of 111.

For two players sharing 120 factory tiles, the relaxed resource model produces 57 nondominated upper-bound score pairs. Important points are 304–18 at a 111/9 split, 171–123 at 70/50, and 147–147 at 60/60. These are optimistic because color counts, wild rounds, bonus timing, and alternating drafts are relaxed.

The empirical lower frontier from 4,000 fully legal games (8,000 mirrored orientations) currently reaches 142–87 at the individual-score extreme, 139–111 for the maximum combined score of 250, and 113–113 for the highest observed minimum player score. The true game frontier lies between the generated empirical and theoretical frontiers.

## Reproduce the telemetry

```powershell
python -m azul.cli score checkpoints/best_score.pt --games 500
python -m azul.analysis --checkpoint checkpoints/best_score.pt --agent neural --opponent neural --games 100
python -m azul.cli evaluate checkpoints/best_score.pt --opponent random --games 300 --mode neural
python -m azul.cli evaluate checkpoints/best_competitive.pt --opponent heuristic --games 100
python -m azul.theory --output theoretical_frontier.json
python -m azul.empirical_frontier --checkpoint checkpoints/best_score_v1.pt --checkpoint checkpoints/best_score.pt --games 1000 --device cuda
```
