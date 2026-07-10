# Azul: Summer Pavilion strategy findings

This document separates three kinds of evidence:

1. recommendations imported from the human-play article that helped shape the exploration curriculum;
2. behavior learned and retained by neural self-play in the corrected simulator;
3. mathematical upper bounds that are useful for understanding tradeoffs but are not necessarily achievable in a legal six-round game.

The experiments use the normal colored-star board in two-player games. The strongest score checkpoint is useful evidence, but it is not proof of optimal play.

## The article that kickstarted the outer-star curriculum

The external reference was Mike Rhea's [Azul: Summer Pavilion – Ultimate Strategy Guide (With Data)](https://boostyourplay.com/azul-summer-pavilion-ultimate-strategy-guide-with-data/). It reports statistics from strong Board Game Arena players and makes the following recommendations.

- Complete one or two outer stars rather than spreading expensive placements everywhere. The article especially favors the 20-point fuchsia star, called `purple` in this simulator.
- Do not plan to finish the multicolor center star. Use its cheap spaces selectively.
- Complete all cost-1 and cost-2 spaces and usually all cost-3 spaces. Treat the cost-4 set as optional.
- Place tiles in connected runs, because placement points make up more of a strong player's score than end-game bonuses alone.
- Favor statues and windows over pillars when planning architectural rebates. Statues return two tiles and windows return three.
- Carry up to four tiles when they will become wild next round rather than forcing inefficient current-round placements.
- Take first player when the center group and next-round initiative justify the penalty; do not pursue it automatically.
- Trigger useful architectural rewards early in the placement phase, while the desired supply colors remain available.
- In two-player match play, deny a critical opponent color when the denial is cheap and specific.

The article's central strategic picture is therefore: cheap number sets plus one high-value outer star, connected efficiently and supported by statues/windows.

## Did the AI discover the purple-star strategy independently?

No—not in the strong sense of discovering it from an unseeded policy.

The chronology matters:

1. The original competitive self-play policy became center-heavy. In a 100-game profile it completed the center star in 82% of games and completed no outer stars.
2. Direct score-maximizing PPO improved into the mid-90s but still completed no outer stars.
3. Elite/CEM self-imitation and a smooth outer-progress reward also failed to create stable outer-star completion.
4. A short scripted curriculum then demonstrated outer-star play. Its design was influenced by the article's purple-star recommendation and by the printed star bonuses. One seat was explicitly biased toward purple during this exploration phase.
5. Exact, undiscounted score self-play resumed with the curriculum reward removed. It discarded much of the teacher's weak play but retained the purple-star behavior, eventually reaching 101.077 mean in the fixed 500-game gate.

So the honest conclusion is:

- **Purple completion was seeded, not independently invented.**
- **Its continued use was independently validated by exact-score self-play.** The policy was free to abandon purple once the curriculum disappeared, but instead made the strategy more efficient.
- The fine structure around the seed—how often to finish purple, which cheap number sets to pair with it, when to carry tiles, and when a window is worth its 5/6 costs—was learned during self-play recovery rather than copied move-for-move from the article.

This distinction is important. Curriculum can reveal a useful behavior basin; it does not prove that everything later learned inside that basin came from the source that suggested it.

## What the strongest score policy learned

The promoted raw score policy reached the following fixed-gate result over 500 self-play games, or 1,000 player scores:

| Metric | Result |
|---|---:|
| Mean | 101.077 |
| Median | 104 |
| 10th percentile | 84 |
| 90th percentile | 113 |
| Maximum | 139 |
| Outer stars per player | 0.459 |
| Center stars per player | 0.000 |

An independent 100-game strategy profile showed:

| Behavior | Per player-game |
|---|---:|
| Score | 99.59 |
| Placed board tiles | 27.06 |
| Purple star completion | 44% |
| All cost-1 spaces | 100% |
| All cost-2 spaces | 100% |
| All cost-3 spaces | 95% |
| All cost-4 spaces | 30% |
| Windows | 0.44 |
| Statues | 4.56 |
| Pillars | 2.47 |
| Wild tiles spent | 11.23 |
| Tiles carried between rounds | 13.19 total per game |
| First-player tokens | 2.95 |

### Findings that agree with the article

- The cheap number sets are extremely reliable value. The learned policy completes 1s and 2s essentially every game and 3s in nearly every game.
- Completing the center star is usually inferior to using its low-cost spaces as part of number-set and feature plans.
- Carrying future wilds is a core resource-conversion mechanism, not an emergency fallback.
- First player is situational. The learned frequency is close to three rounds per game, not zero and not six.
- Statues are the most frequent architectural reward because their 1/2/3/4 geometry overlaps naturally with cheap number-set development.

### Findings that add nuance or differ from the article

#### One efficient star beat forced multi-star play

The article recommends completing one or two stars. In these experiments, forcing broader outer-star exploration produced a policy with **1.374 outer stars per player**, but it averaged only **95.67**, below the 101.077 one-star policy.

That does not prove two stars are bad. It shows that star count alone is a poor objective. A second star can consume the same 5/6 payments needed for windows, connected scoring, carries, and cheap-number completion. Finish a second star only when the tile flow makes the whole package efficient.

#### A window is best treated as part of a star plan

The learned policy does not chase windows generically. It averages only 0.44 windows, almost exactly matching its 0.44 purple-star completion rate in the independent profile. The useful pattern is usually not "build a window because three free tiles are good"; it is "the purple 5/6 placements finish both a valuable star and its window."

#### Pillars are not primary targets, but they are not irrelevant

The article correctly identifies pillars as worse raw rebates. The policy still averages 2.47 pillars because some are incidental consequences of completing cheap number spaces and center low costs. A weak rebate can still be good when most of its prerequisites were already justified by another scoring plan.

#### Completing all 4s became more attractive inside a coherent outer plan

The center-heavy checkpoint completed all 4s in only 7% of games. The purple-star score policy increased that to 30%. Cost-4 completion is still situational, but it becomes more realistic after low-number coverage, statues, and one outer-star plan have already aligned the board.

#### Lower variance was part of the improvement

The move from the original competitive checkpoint to the score checkpoint raised the mean from 89.149 to 101.077, but it also raised the 10th percentile from 76 to 84. The stronger policy was not merely taking riskier shots at a large star bonus; it reduced bad games as well.

#### Absolute score and winning are different objectives

The score checkpoint is the best raw policy for maximizing its own total. The older competitive checkpoint combined with tactical guardrails remains the stronger measured head-to-head opponent. Denial moves can improve win probability while lowering the total number of points created at the table.

## Practical strategy derived from the experiments

### 1. Start with the 1/2/3 skeleton

Treat all 1s and 2s as default targets. Plan to complete the 3s unless the draft becomes unusually hostile. These spaces score number bonuses, develop statues, and create inexpensive connected runs.

### 2. Choose one outer star early, but require evidence before committing to a second

Purple is the default because its 20-point bonus gives the best payoff for the same printed costs. Commitment should still depend on draft access. A half-built purple star scores no completion bonus, so switch only before the expensive 5/6 investment has made the plan irreversible.

### 3. Connect placements deliberately

Within a star, grow an existing component whenever the broader bonus plan is equal. A completed six-space star can yield 21 placement points when built as a connected run. Scattered cells give up repeated adjacency scoring even if the final occupied set is identical.

### 4. Couple expensive spaces to two rewards

A cost-5 or cost-6 placement should usually advance at least two of:

- an outer-star completion;
- a window;
- the all-4 set or another realistic number goal;
- a large connected component.

Paying 5/6 solely for isolated immediate points is one of the fastest ways to make a promising board inefficient.

### 5. Use architectural rewards as timing tools

Trigger a feature early when the supply already contains a needed color. Delay it when the current supply is poor and an opponent's reward may force a refill. The number of tiles returned is only half the value; access to the correct color at the correct time is the other half.

### 6. Carry with a purpose

Future-wild tiles are the best default carry. A carried natural color is also justified when it closes a specific expensive star/window placement next round. Avoid carrying four unrelated leftovers merely because four slots are available.

### 7. Price the first-player token as part of the draft

The token is worthwhile when the chosen center group itself is efficient and first access next round matters. Do not pay a large per-tile penalty for initiative without a concrete next-round target.

### 8. Separate score play from match play

For raw score, prefer constructive drafts and architectural chains. For winning a two-player match, one narrow denial can be worth more than a small constructive gain. Do not import the cooperative score-search experiments into normal competitive advice: their support player was intentionally sacrificing resources.

## Common losing patterns observed in training

- Starting several stars and completing none.
- Treating completed-star count as the objective instead of total score.
- Paying for isolated 5/6 spaces without a star/window connection.
- Forcing the center star because it is visibly close, despite its lower bonus and mixed-color constraint.
- Spending next round's wild color for a marginal current placement.
- Triggering a feature after the useful supply colors have already disappeared.
- Taking a large first center group mainly for initiative.
- Optimizing immediate placement points while silently giving up a reachable end-game bonus.
- Using a pure score policy for competitive denial play, or a denial policy when measuring raw score.

## Pareto frontier: what "how well can an AI do?" means

A two-player score frontier has two objectives: maximize P0's score and maximize P1's score. Because both players share the same 120 factory tiles, improving one score can reduce the resources available to the other. A pair is Pareto-optimal when neither player can be improved without reducing the other under the model being used.

### Exact printed-board ceiling

If tile-flow feasibility is ignored, a completely filled board scores exactly 304 at most:

- 5 starting points;
- 147 optimally ordered connected-placement points;
- 100 points for all six outer stars;
- 12 points for the center star;
- 40 points for all four number sets.

The full board costs 147 payment tiles. Completing all 18 architectural features returns 36 supply tiles, producing a relaxed net construction cost of 111.

This **304 is an exact board-scoring ceiling**, not a claim that a legal six-round draft can fill the board.

### Relaxed two-player upper frontier

The mixed-integer optimizer credits feature rebates and shares the two-player game's 120 factory tiles, but relaxes colors, wild-round timing, supply timing, and alternating draft order. It finds 57 nondominated upper-bound pairs. Representative points are:

| Relaxed net allocation | Upper-bound score pair | Combined |
|---:|---:|---:|
| 60 / 60 | 147 / 147 | 294 |
| 70 / 50 | 171 / 123 | 294 |
| 79 / 41 | 203 / 98 | 301 |
| 111 / 9 | 304 / 18 | 322 |

The shape matters more than the headline 304. In this relaxation, a 150 average means at least 300 combined points, and those solutions are asymmetric—approximately 203/98 or more extreme. That observation motivated the experimental builder/support policies. It does **not** imply that sacrificing one player is good competitive Azul strategy.

### Empirical legal-game lower frontier

The stored empirical frontier sampled 4,000 fully legal games and mirrored both seat orientations. It found:

- 142 as the highest individual score in that broad sample;
- 250 as the highest combined score, from 139/111;
- 113/113 as the highest observed minimum-player pair;
- seven nondominated observed pairs.

Later targeted rollout-search experiments reached an individual score of 149 and a combined score of 251 on smaller samples. Across much larger stochastic tail audits, neither the one-star nor multi-star neural family produced a 200-point game.

The empirical frontier is a lower bound: better search can expand it. The relaxed optimizer is an upper bound: adding real round, color, and drafting constraints can only shrink it. The true legal-game frontier lies between them, and the large gap is an open research problem rather than evidence that 200 is attainable.

## Training lessons

### What worked

- Correct board geometry and a Markov observation, including the public next-start-player holder.
- Exact score-delta rewards with `gamma=1` and `GAE lambda=1`, so sixth-round bonuses retain full credit.
- Large GPU rollout and PPO batches.
- A short behavior curriculum to cross an exploration barrier, followed by a much longer phase with the curriculum removed.
- Fixed-seed distribution gates rather than selecting checkpoints from noisy training means.
- Preserving separate score and competitive checkpoints.

### What did not work

- Hoping pure self-play would discover outer-star completion from the center-heavy policy.
- Discounting final bonuses across roughly 90 decisions.
- Elite/CEM imitation, which amplified lucky trajectories rather than reproducible decisions.
- Rewarding outer progress without demonstrations.
- Forcing more completed stars without preserving placement efficiency.
- Shared cooperative PPO for permanent builder/support roles; the relative observation is intentionally symmetric.
- Simple search distillation, because small state-dependent corrections caused distribution shift when executed without search.
- Deep nested terminal rollout, which became computationally impractical.

## Reproduce the evidence

```powershell
# Promoted raw score gate
python -m azul.cli score checkpoints/best_score.pt --games 500

# Learned behavior profile
python -m azul.analysis --checkpoint checkpoints/best_score.pt `
  --agent neural --opponent neural --games 100

# Competitive benchmark
python -m azul.cli evaluate checkpoints/best_competitive.pt `
  --opponent heuristic --games 100

# Exact ceiling and relaxed resource frontier
python -m azul.theory --output theoretical_frontier.json

# Fully legal empirical score-pair frontier
python -m azul.empirical_frontier `
  --checkpoint checkpoints/best_score_v1.pt `
  --checkpoint checkpoints/best_score.pt `
  --games 1000 --device cuda
```

The full numeric artifacts are in `training_report.json`, `theoretical_frontier.json`, and `empirical_frontier.json`.
