# Blog post notes

The outside reference was Mike Rhea's [Azul: Summer Pavilion – Ultimate Strategy Guide (With Data)](https://boostyourplay.com/azul-summer-pavilion-ultimate-strategy-guide-with-data/). It analyzes strong Board Game Arena play. The recommendations below are the article's claims, not discoveries credited to this project's AI.

- Finish one or two outer stars instead of spreading expensive placements across the board. The article particularly favors the 20-point fuchsia star, called `purple` in this simulator.
- Do not make the multicolor center star a primary goal. Its cheap spaces can still be useful.
- Complete every cost-1 and cost-2 space, usually complete the cost-3 set, and treat the cost-4 set as optional.
- Build connected groups so placements score repeatedly through adjacency.
- Prefer statues and windows over pillars when planning architectural rewards. Their rebates are more efficient for the cost of the surrounding spaces.
- Carry up to four tiles when they will become useful wild tiles next round instead of forcing weak placements now.
- Take the first-player token only when the center draft and next-round initiative justify its penalty.
- Trigger architectural rewards while the supply contains colors you actually need.
- In two-player games, deny a critical color when the denial is cheap and specifically disrupts the opponent.

The article's overall plan is therefore: build the cheap number sets, efficiently connect one high-value outer star—often purple—and use statues or windows to sustain the plan.

# Learned AI, tendencies, score

These are measured behaviors of the strongest raw-score checkpoint in the corrected two-player simulator. They should not be read as proof of optimal play.

One provenance limit matters: **the AI did not independently invent the purple-star idea.** Purple completion was introduced through a blog-informed curriculum. The AI later retained and refined it while optimizing exact game score, but purple itself must be treated as a seeded idea. The frequencies, combinations, and tradeoffs below are what the trained policy learned around that seed.

## Score

The fixed evaluation used 500 self-play games, producing 1,000 player scores.

| Metric | Result |
|---|---:|
| Mean score | 101.077 |
| Median | 104 |
| 10th percentile | 84 |
| 90th percentile | 113 |
| Maximum | 139 |
| Outer stars per player | 0.459 |
| Center stars per player | 0.000 |

A separate 100-game behavior audit found:

| Learned behavior | Per player-game |
|---|---:|
| Score | 99.59 |
| Board tiles placed | 27.06 |
| Purple star completed | 44% |
| All cost-1 spaces completed | 100% |
| All cost-2 spaces completed | 100% |
| All cost-3 spaces completed | 95% |
| All cost-4 spaces completed | 30% |
| Windows completed | 0.44 |
| Statues completed | 4.56 |
| Pillars completed | 2.47 |
| Wild tiles spent | 11.23 |
| Tiles carried between rounds | 13.19 total per game |
| First-player tokens taken | 2.95 |

Targeted rollout search has produced a 149 individual score and a 251 combined score in smaller samples. Those are observed peaks, not representative policy averages.

## Learned tendencies

- **The 1/2/3 number skeleton is extremely reliable.** The policy completes every 1 and 2 in essentially every game and nearly always completes the 3s. These cheap spaces simultaneously create number bonuses, connected scoring, and statue progress.
- **The center star is a source of useful cells, not a completion target.** The policy uses low-cost center spaces without completing the center star in the measured games.
- **One coherent outer star performed better than forced star volume.** A policy pushed toward broader star completion reached 1.374 outer stars per player but averaged only 95.67, below the 101.077 score policy. Completed-star count is therefore a poor objective by itself. A second star is worthwhile only when its colors and expensive spaces fit the rest of the board plan.
- **Windows work best as part of a star plan.** The policy's 0.44 windows per game almost exactly matches its 44% purple completion rate. It tends to make purple's expensive 5/6 placements pay twice by closing both the star and the adjacent window.
- **Statues are the default feature; pillars are useful overlap.** Statues occur 4.56 times per game because their geometry aligns with cheap-number development. Pillars still occur 2.47 times despite their weaker rebate, usually because most prerequisite spaces were already justified by another goal.
- **Cost-4 completion is more viable than a simple “optional” label suggests.** The policy completes the 4s in 30% of games when an outer-star and feature plan naturally supplies those cells. It does not force them in every game.
- **Carrying tiles is routine resource conversion.** The policy carries about 13.19 tiles across round boundaries per full game, especially colors about to become wild. Carrying is strongest when each tile has a named next-round use.
- **First player is situational.** Taking the token about 2.95 times per game indicates neither automatic pursuit nor avoidance. The center group and a concrete next-round target must pay for the penalty.
- **Expensive placements need multiple jobs.** A cost-5 or cost-6 space is most efficient when it advances a star plus a window, number set, or connected component. Isolated expensive placements are rarely attractive.
- **Raw score and match strength are different objectives.** The raw-score checkpoint is best at maximizing its own total. A competitive checkpoint with tactical guardrails can be stronger head-to-head because a denial move may improve win probability while reducing the table's total score.

The clearest AI contribution beyond the blog's headline advice is not “complete purple.” That idea was seeded. It is the narrower finding that **one efficient star package beats indiscriminate star completion**, and that purple is strongest when its expensive cells also close a window, extend a connected component, and preserve the cheap-number structure. The AI also shows that incidental pillars and situational cost-4 completion can be worthwhile even though neither should be a primary objective.

# Addendum: verified AZ2 search champion (2026-07-13)

The project's strongest measured match-play agent is now the structured AZ2
policy plus stochastic search in `checkpoints/best_az2_search.pt`. Its verified
deployment uses 128 simulations per decision, persistent subtree reuse, a 75%
tactical root prior, and 25% learned-value influence. It beat the unchanged
strategic heuristic 83–44–1 (**65.23%**, +7.34 average margin) on a fresh
128-game alternating-seat confirmation. Combined with a disjoint preliminary,
the result is 125–64–3 (**65.89%**) over 192 games.

This agent is not a clean tabula-rasa source of strategic discoveries. Its
policy descends from the score curriculum described above, and its root search
deliberately blends the learned prior with the hand-written tactical heuristic.
Purple was seeded by the outside strategy article, and connected scoring,
feature rewards, wild preservation, and useful draft size are explicit parts of
the tactical prior. The findings below are therefore best read as evidence
about which combinations survived training and deep adversarial search.

## Champion behavior sample

A separate 32-game directional telemetry sample used the full verified
128-simulation configuration against the strategic heuristic. It scored
24–7–1 in that small block. The match result is too small to replace the
128/192-game verification above, but the action and board frequencies are
useful descriptions of how the champion plays.

| Champion tendency | 32-game sample |
|---|---:|
| Mean score | 94.84 |
| Mean margin | +9.22 |
| Board tiles | 27.38 per game |
| Purple stars completed | 53.1% |
| Other outer stars completed | 0% |
| Center stars completed | 0% |
| Center placements | 4.03 per game |
| All cost-1 spaces completed | 100% |
| All cost-2 spaces completed | 100% |
| All cost-3 spaces completed | 75% |
| All cost-4 spaces completed | 12.5% |
| Windows / statues / pillars | 0.53 / 4.22 / 2.94 per game |
| Wild tiles spent | 11.81 per game |
| Tiles carried | 12.66 per game |
| First-player tokens | 4.00 per game |
| Drafts taken from center | 54.4% |
| Mean tiles gained per draft | 1.94 |

The placement-cost distribution was 25.6% cost 1, 25.6% cost 2, 24.7% cost 3,
17.9% cost 4, 3.5% cost 5, and 2.7% cost 6. In other words, **93.7% of its
placements cost four or less**. The 5/6 spaces are not routine construction;
they are the small expensive capstone of a much larger cheap-space plan.

## What the champion's play suggests

- **Build the cheap skeleton before chasing prestige.** Costs 1 and 2 remain
  effectively mandatory, and cost 3 remains the normal extension. These cells
  create connected points, number bonuses, and statue progress at the same
  time. Cost 4 is conditional; 5 and 6 need unusually strong overlap.
- **Treat purple plus its window as one package.** Purple-star and window
  completion were both 53.1% in the sample. This near-perfect coupling is more
  informative than purple completion alone: the expensive placements are
  justified when the star bonus, window rebate, and adjacency all pay together.
- **Do not confuse using the center with completing it.** The champion placed
  about four center cells per game while never completing the center star. Cheap
  center cells can close number sets, connect groups, and trigger features
  without turning center completion into the plan.
- **Statues are the rebate engine.** More than four statues per game survived
  both score training and competitive search. Pillars are frequent secondary
  overlap. Windows are rarer and usually tied to the chosen outer star.
- **Carry with intent.** Roughly 12.7 carried tiles and 11.8 wilds spent per
  game show that round boundaries are part of the resource plan. A tile worth
  keeping should have a named placement or a valuable next-round wild role;
  carrying arbitrary leftovers is not the same strategy.
- **Tempo matters more in match play.** The champion took the first-player token
  four times per game, more often than the older score policy's 2.95. It also
  drafted from the center slightly more than half the time. Search appears
  willing to pay the token penalty when initiative, denial, and the center pile
  jointly compensate for it.
- **Large drafts are not automatically best.** Its average gain was only 1.94
  tiles per draft. That does not prove every small draft was a denial, but it is
  consistent with selecting for exact color, timing, and opponent impact rather
  than maximizing immediate tile count.
- **Deep tactics change the value of the same strategic plan.** The verified
  checkpoint reached 62.5% at 64 simulations on its independent block and
  65.23% at 128. The board plan did not change; the deeper search improved
  sequencing, draft timing, payment choices, and responses to the opponent.

## A practical playbook

Before each draft, ask four questions in order:

1. What exact placement does this color buy this round?
2. If it is carried, what does it buy next round, and will it become wild?
3. Does taking it also remove a critical affordable placement from the
   opponent?
4. If it comes from the center, are the pile and next-round initiative worth
   the first-player penalty?

During placement:

1. Complete cheap connected cells and statue prerequisites first.
2. Time a feature so its supply reward contains colors that extend the same
   plan; a rebate of unusable colors is much weaker than its printed tile count.
3. Spend wilds on bottlenecks, not merely on the first legal substitution.
4. Place a cost-5/6 cell only when it does at least two jobs—ideally star plus
   window, with adjacency or number progress as a third.
5. Keep up to four tiles rather than forcing an isolated placement, but name
   their next-round jobs before committing to the carry.

In the final rounds, compare moves by **score margin**, not just personal score.
A lower-scoring draft can be correct when it removes the opponent's star,
number-set, or feature completion while preserving your own coherent finish.
This is the main strategic distinction between the score champion and the AZ2
match champion.

## Playing against the champion

- Contest purple costs 5/6 or the adjacent window colors when the denial is
  cheap. Denying a random purple tile is less useful than breaking the package.
- Watch its next-round wild conversion before leaving a convenient center pile.
- Force a choice between initiative and a weak first-player pile; the champion
  actively values tempo and will otherwise collect both.
- Pressure the cost-3 skeleton and statue intersections. They are more central
  to its engine than flashy cost-6 placements.
- Do not race a second outer star merely because the champion has one. Its own
  evidence says star count without efficient overlap lowers score.

## Limits of the evidence

The 32-game telemetry block is descriptive and noisy. The 65.23% result is a
stronger match-strength measurement, but it is against one fixed strategic
heuristic, not a proof of optimal play or universal 65% strength. Several
candidates passed 32/64-game preliminary arenas and failed fresh confirmation;
only independently confirmed results are promoted in this project. The supplied
human logs currently contain two complete games and two useful partial games;
they are valuable safety and imitation examples, but far too small to support
new population-level strategy claims.

# Pareto frontier

“How well can an AI do?” has two different answers in a two-player game. Competitive play tries to beat the opponent. A score-pair frontier instead asks how high both final scores can be when the players share the same 120 factory tiles. A score pair is Pareto-optimal when neither player's score can be improved without lowering the other's under the model being measured.

## Exact printed-board ceiling

Ignoring whether the required colors can legally arrive in six rounds, a completely filled board has a maximum score of 304:

- 5 starting points;
- 147 connected-placement points;
- 100 points for all six outer stars;
- 12 points for the center star;
- 40 points for all four number sets.

Filling the board costs 147 payment tiles. All 18 architectural features return 36 supply tiles, giving a relaxed net construction cost of 111. The 304 figure is an exact board-scoring ceiling, not an achievable-game claim.

## Relaxed two-player upper frontier

The optimizer shares the two-player game's 120 factory tiles and credits architectural rebates, but it ignores color availability, wild-color timing, supply timing, and alternating draft order. It found 57 nondominated upper-bound pairs. Representative points are:

| Relaxed net allocation | Upper-bound score pair | Combined score |
|---:|---:|---:|
| 60 / 60 | 147 / 147 | 294 |
| 70 / 50 | 171 / 123 | 294 |
| 79 / 41 | 203 / 98 | 301 |
| 111 / 9 | 304 / 18 | 322 |

The shape is more informative than the 304 endpoint. Even in the relaxed model, averaging 150 requires at least 300 combined points and strongly asymmetric resource use—roughly 203/98 or more extreme. That is what “designated builder” means in the project: one seat deliberately receives most resources to explore the maximum possible individual score. It is a ceiling experiment, not a recommended way to win a normal game.

## Empirical legal-game lower frontier

The broad legal-game sample contains 4,000 games, mirrored across both seat orientations. It found:

- 142 as the highest individual score in that sample;
- 250 as the highest combined score, from a 139/111 game;
- 113/113 as the highest observed minimum-player pair;
- seven nondominated observed score pairs.

Smaller targeted rollout searches later reached 149 individually and 251 combined. Much larger stochastic tail audits did not produce a 200-point game from either tested neural policy family.

The empirical frontier is a lower bound because better policies and search can expand it. The relaxed mathematical frontier is an upper bound because enforcing real colors, rounds, wilds, and draft competition can only shrink it. The true legal frontier lies somewhere between the observed 149/251 peaks and the relaxed bounds; the current evidence does not establish that a 200-point individual game is attainable.
