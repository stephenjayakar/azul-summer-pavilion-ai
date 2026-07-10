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
