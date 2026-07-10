import { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const COLORS = ["purple", "green", "orange", "yellow", "blue", "red"];
const BOARD_CLOCKWISE = ["orange", "red", "blue", "yellow", "green", "purple"];
const STAR_BONUSES = {
  purple: 20,
  green: 18,
  orange: 17,
  yellow: 16,
  blue: 15,
  red: 14,
};
const STAR_POSITIONS = BOARD_CLOCKWISE.map((color, index) => {
  // The printed board starts orange on the upper-left spoke. Each following
  // star advances clockwise by 60 degrees.
  const angle = -120 + index * 60;
  const radians = (angle * Math.PI) / 180;
  return {
    color,
    angle,
    x: 50 + Math.cos(radians) * 31.5,
    y: 50 + Math.sin(radians) * 31.5,
  };
});
const FEATURE_COPY = {
  window: {
    label: "Window",
    mark: "W",
    reward: 3,
    copy: "Complete costs 5 and 6 on one colored star.",
  },
  statue: {
    label: "Statue",
    mark: "S",
    reward: 2,
    copy: "Bridge the four spaces at the junction of two neighboring stars.",
  },
  pillar: {
    label: "Pillar",
    mark: "P",
    reward: 1,
    copy: "Connect costs 2 and 3 on an outer star to the two center spaces on its spoke.",
  },
};

function classNames(...values) {
  return values.filter(Boolean).join(" ");
}

function polarPoint(radius, angle) {
  const radians = (angle * Math.PI) / 180;
  return {
    x: 50 + Math.cos(radians) * radius,
    y: 50 + Math.sin(radians) * radius,
  };
}

function featureKey(feature) {
  return `${feature.kind}-${feature.index}`;
}

function featureCells(feature) {
  const boardIndex = feature.index;
  const star = BOARD_CLOCKWISE[boardIndex];
  const next = BOARD_CLOCKWISE[(boardIndex + 1) % 6];
  if (feature.kind === "window") {
    return [
      { star, slot: 4 },
      { star, slot: 5 },
    ];
  }
  if (feature.kind === "statue") {
    return [
      { star, slot: 0 },
      { star, slot: 1 },
      { star: next, slot: 2 },
      { star: next, slot: 3 },
    ];
  }
  return [
    { star, slot: 1 },
    { star, slot: 2 },
    { star: "center", slot: (boardIndex - 1 + 6) % 6 },
    { star: "center", slot: boardIndex },
  ];
}

function cellKey(star, slot) {
  return `${star}-${slot}`;
}

function featurePosition(feature) {
  const baseAngle = -120 + feature.index * 60;
  if (feature.kind === "window") return polarPoint(47, baseAngle);
  // Statues occupy the inward junction between neighboring star centers.
  if (feature.kind === "statue") return polarPoint(27.5, baseAngle + 30);
  // Pillars sit halfway along each center-to-outer-star spoke.
  return polarPoint(16, baseAngle);
}

function tileLetter(color) {
  return color === "yellow" ? "Y" : color?.[0]?.toUpperCase();
}

function TileFace({ color, count, small = false }) {
  return (
    <span
      className={classNames("tile-face", color, small && "small-tile")}
      aria-hidden="true"
    >
      <span>{tileLetter(color)}</span>
      {count !== undefined && <b>{count}</b>}
    </span>
  );
}

function TileCluster({ color, count, companionWild = false, wildColor }) {
  return (
    <span className="tile-cluster" aria-hidden="true">
      <span className="tile-stack-visual">
        {Array.from({ length: Math.min(count, 3) }, (_, index) => (
          <span
            className={classNames("tile-layer", color)}
            style={{ "--layer": index }}
            key={index}
          >
            {tileLetter(color)}
          </span>
        ))}
      </span>
      <b className="tile-count">×{count}</b>
      {companionWild && (
        <span className={classNames("wild-companion", wildColor)}>
          + {tileLetter(wildColor)} wild
        </span>
      )}
    </span>
  );
}

function TileGroup({
  color,
  count,
  action,
  busy,
  onAction,
  companionWild = false,
  wildColor,
  context,
}) {
  const actionable = Boolean(action) && !busy;
  const label =
    action?.description ||
    `${count} ${color} tile${count === 1 ? "" : "s"}${context ? ` in ${context}` : ""}`;
  if (!action) {
    return (
      <span className="tile-group is-static" aria-label={label} title={label}>
        <TileCluster
          color={color}
          count={count}
          companionWild={companionWild}
          wildColor={wildColor}
        />
      </span>
    );
  }
  return (
    <button
      type="button"
      className="tile-group is-actionable"
      disabled={!actionable}
      onClick={() => onAction(action.id)}
      aria-label={label}
      title={label}
    >
      <TileCluster
        color={color}
        count={count}
        companionWild={companionWild}
        wildColor={wildColor}
      />
    </button>
  );
}

function Inventory({
  player,
  legalActions,
  busy,
  onAction,
  interactive = false,
}) {
  const keepActions = legalActions.filter((action) => action.kind === "keep");
  const total = Object.values(player.inventory).reduce(
    (sum, count) => sum + count,
    0,
  );
  const storedTotal = Object.values(player.stored || {}).reduce(
    (sum, count) => sum + count,
    0,
  );
  return (
    <section
      className="inventory-shelf"
      aria-label={`${player.name} available tiles`}
    >
      <div className="shelf-heading">
        <span>Available tiles</span>
        <b>{total}</b>
      </div>
      <div className="inventory-tiles">
        {COLORS.filter((color) => player.inventory[color]).map((color) => {
          const action = interactive
            ? keepActions.find((item) => item.color === color)
            : null;
          return (
            <TileGroup
              key={color}
              color={color}
              count={player.inventory[color]}
              action={action}
              busy={busy}
              onAction={onAction}
              context="your inventory"
            />
          );
        })}
        {!total && <span className="empty-copy">No tiles in hand</span>}
      </div>
      {!!storedTotal && (
        <div className="stored-row">
          <span>Kept for next round</span>
          <span>
            {COLORS.filter((color) => player.stored[color]).map((color) => (
              <TileFace
                color={color}
                count={player.stored[color]}
                small
                key={color}
              />
            ))}
          </span>
        </div>
      )}
    </section>
  );
}

function BoardSlot({
  star,
  color,
  slot,
  position,
  occupied,
  centerColor,
  actions,
  busy,
  highlighted,
  onChooseActions,
}) {
  const cost = slot + 1;
  const actionable = actions.length > 0 && !busy;
  const placedColor = star === "center" ? centerColor : color;
  const label = occupied
    ? `${placedColor} tile on ${star === "center" ? "center star" : `${color} star`} cost ${cost}`
    : actionable
      ? `Place on ${star === "center" ? "center star" : `${color} star`} cost ${cost}`
      : `Empty ${star === "center" ? "center star" : `${color} star`} cost ${cost}`;
  const Tag = actionable ? "button" : "span";
  return (
    <Tag
      type={actionable ? "button" : undefined}
      className={classNames(
        "board-slot",
        `slot-${cost}`,
        `position-${position}`,
        occupied && "is-occupied",
        actionable && "is-playable",
        highlighted && "is-feature-target",
      )}
      onClick={
        actionable
          ? () =>
              onChooseActions({
                label,
                actions,
                color: star === "center" ? actions[0].color : color,
                cost,
              })
          : undefined
      }
      aria-label={label}
      title={label}
    >
      <span className="slot-cost">{cost}</span>
      {occupied && (
        <span className={classNames("placed-tile", placedColor)}>
          {tileLetter(placedColor)}
        </span>
      )}
    </Tag>
  );
}

function StarNode({
  star,
  player,
  legalActions,
  busy,
  highlightedCells,
  onChooseActions,
}) {
  const position = STAR_POSITIONS.find((item) => item.color === star.color);
  return (
    <div
      className={classNames("star-node", `star-${star.color}`)}
      style={{ left: `${position.x}%`, top: `${position.y}%` }}
    >
      <div className={classNames("star-core", star.color)}>
        <strong>{star.color}</strong>
        <span>+{STAR_BONUSES[star.color]}</span>
      </div>
      {star.slots.map((occupied, slot) => {
        // Costs rotate once per star around the printed board. This keeps the
        // 5-6 edge outside, the 2-3 edge toward the center, and the remaining
        // pairs on the two neighboring-statue edges.
        const boardIndex = BOARD_CLOCKWISE.indexOf(star.color);
        const position = ((boardIndex + slot + 1) % 6) + 1;
        const actions = legalActions.filter(
          (action) =>
            action.kind === "place_outer" &&
            action.star === star.color &&
            action.cost === slot + 1,
        );
        return (
          <BoardSlot
            key={slot}
            star={star.color}
            color={star.color}
            slot={slot}
            position={position}
            occupied={occupied}
            actions={actions}
            busy={busy}
            highlighted={highlightedCells.has(cellKey(star.color, slot))}
            onChooseActions={onChooseActions}
          />
        );
      })}
    </div>
  );
}

function CenterStar({
  player,
  legalActions,
  busy,
  highlightedCells,
  onChooseActions,
}) {
  return (
    <div className="star-node center-node">
      <div className="star-core center-core">
        <strong>center</strong>
        <span>+12</span>
      </div>
      {player.center.map((color, slot) => {
        const actions = legalActions.filter(
          (action) =>
            action.kind === "place_center" && action.cost === slot + 1,
        );
        return (
          <BoardSlot
            key={slot}
            star="center"
            slot={slot}
            position={slot + 1}
            occupied={Boolean(color)}
            centerColor={color}
            actions={actions}
            busy={busy}
            highlighted={highlightedCells.has(cellKey("center", slot))}
            onChooseActions={onChooseActions}
          />
        );
      })}
    </div>
  );
}

function ArchitectureBadge({ feature, focused, compact, onFocus }) {
  const meta = FEATURE_COPY[feature.kind];
  const position = featurePosition(feature);
  const near = !feature.complete && feature.progress === feature.required - 1;
  if (compact && !feature.complete && !near) return null;
  return (
    <button
      type="button"
      className={classNames(
        "architecture-badge",
        `kind-${feature.kind}`,
        feature.complete && "is-complete",
        near && "is-near",
        focused && "is-focused",
      )}
      style={{ left: `${position.x}%`, top: `${position.y}%` }}
      onClick={() => onFocus(featureKey(feature))}
      aria-pressed={focused}
      aria-label={`${feature.name}: ${feature.requirement}. Reward ${feature.reward} tiles. Progress ${feature.progress} of ${feature.required}.`}
      title={`${feature.name}: ${feature.requirement}`}
    >
      <span className="architecture-mark">{meta.mark}</span>
      <b>+{feature.reward}</b>
      <small>
        {feature.complete ? "✓" : `${feature.progress}/${feature.required}`}
      </small>
    </button>
  );
}

function PavilionMap({
  player,
  legalActions = [],
  busy = false,
  compact = false,
  focusedFeatureKey,
  onFocusFeature = () => {},
  onChooseActions = () => {},
}) {
  const focusedFeature = player.architecture.find(
    (feature) => featureKey(feature) === focusedFeatureKey,
  );
  const highlightedCells = new Set(
    (focusedFeature ? featureCells(focusedFeature) : []).map((cell) =>
      cellKey(cell.star, cell.slot),
    ),
  );
  const polygonPoints = STAR_POSITIONS.map(
    (position) => `${position.x},${position.y}`,
  ).join(" ");
  return (
    <div className={classNames("pavilion-map", compact && "is-compact")}>
      <svg className="board-geometry" viewBox="0 0 100 100" aria-hidden="true">
        <circle cx="50" cy="50" r="31.5" />
        <polygon points={polygonPoints} />
        {STAR_POSITIONS.map((position) => (
          <line
            x1="50"
            y1="50"
            x2={position.x}
            y2={position.y}
            key={position.color}
          />
        ))}
      </svg>
      {player.outer.map((star) => (
        <StarNode
          key={star.color}
          star={star}
          player={player}
          legalActions={legalActions}
          busy={busy}
          highlightedCells={highlightedCells}
          onChooseActions={onChooseActions}
        />
      ))}
      <CenterStar
        player={player}
        legalActions={legalActions}
        busy={busy}
        highlightedCells={highlightedCells}
        onChooseActions={onChooseActions}
      />
      {player.architecture.map((feature) => (
        <ArchitectureBadge
          key={featureKey(feature)}
          feature={feature}
          compact={compact}
          focused={featureKey(feature) === focusedFeatureKey}
          onFocus={onFocusFeature}
        />
      ))}
    </div>
  );
}

function ArchitectureGuide({ features, focusedFeatureKey, onFocusFeature }) {
  const focused =
    features.find((feature) => featureKey(feature) === focusedFeatureKey) ||
    features.find((feature) => feature.kind === "pillar");
  const targets = focused ? featureCells(focused) : [];
  const focusKind = (kind) => {
    const next =
      features.find((feature) => feature.kind === kind && !feature.complete) ||
      features.find((feature) => feature.kind === kind);
    if (next) onFocusFeature(featureKey(next));
  };
  return (
    <section
      className="architecture-guide"
      aria-label="Architectural bonus guide"
    >
      <div className="guide-types">
        {Object.entries(FEATURE_COPY).map(([kind, meta]) => {
          const complete = features.filter(
            (feature) => feature.kind === kind && feature.complete,
          ).length;
          return (
            <button
              type="button"
              className={classNames(
                "guide-type",
                focused?.kind === kind && "is-active",
              )}
              onClick={() => focusKind(kind)}
              key={kind}
            >
              <span>{meta.mark}</span>
              <span>
                <strong>{meta.label}</strong>
                <small>{meta.copy}</small>
              </span>
              <b>+{meta.reward}</b>
              <em>{complete}/6</em>
            </button>
          );
        })}
      </div>
      {focused && (
        <div className="pattern-lens" aria-live="polite">
          <div className="pattern-lens-heading">
            <span className={classNames("lens-mark", `kind-${focused.kind}`)}>
              {FEATURE_COPY[focused.kind].mark}
            </span>
            <span>
              <span className="eyebrow">Pattern lens</span>
              <strong>{focused.name}</strong>
            </span>
            <span className="reward-pill">
              +{focused.reward} supply tile{focused.reward === 1 ? "" : "s"}
            </span>
          </div>
          <p>
            {focused.requirement}. The required spaces are glowing on the board.
          </p>
          <div className="target-cells">
            {targets.map((target) => (
              <span key={cellKey(target.star, target.slot)}>
                {target.star === "center" ? "Center" : target.star}{" "}
                <b>{target.slot + 1}</b>
              </span>
            ))}
          </div>
          <div className="pattern-progress">
            <span
              style={{
                width: `${(focused.progress / focused.required) * 100}%`,
              }}
            />
          </div>
          <small>
            {focused.complete
              ? "Claimed"
              : `${focused.progress} of ${focused.required} spaces filled`}
          </small>
        </div>
      )}
    </section>
  );
}

function PlayerPanel({
  player,
  opponentName,
  legalActions = [],
  busy,
  onAction,
  onChooseActions,
  focusedFeatureKey,
  onFocusFeature,
  compact = false,
}) {
  const isKeeping = legalActions.some((action) => action.kind === "keep");
  return (
    <article
      className={classNames(
        "player-pavilion",
        player.index ? "opponent-pavilion" : "your-pavilion",
        compact && "is-compact",
      )}
      aria-label={`${player.name} pavilion`}
    >
      <header className="pavilion-header">
        <div className="player-identity">
          <span className="avatar">{player.index ? "AI" : "YOU"}</span>
          <span>
            <span className="eyebrow">
              {player.index ? "Opponent pavilion" : "Your pavilion"}
            </span>
            <strong>
              {player.index ? opponentName : "Build the pavilion"}
            </strong>
          </span>
        </div>
        <div className="score-box">
          <strong>{player.score}</strong>
          <span>points</span>
        </div>
      </header>
      <Inventory
        player={player}
        legalActions={legalActions}
        busy={busy}
        onAction={onAction}
        interactive={!player.index && isKeeping}
      />
      <PavilionMap
        player={player}
        legalActions={player.index ? [] : legalActions}
        busy={busy}
        compact={compact}
        focusedFeatureKey={focusedFeatureKey}
        onFocusFeature={onFocusFeature}
        onChooseActions={onChooseActions}
      />
      {!compact && (
        <ArchitectureGuide
          features={player.architecture}
          focusedFeatureKey={focusedFeatureKey}
          onFocusFeature={onFocusFeature}
        />
      )}
    </article>
  );
}

function DraftSurface({ game, busy, onAction }) {
  const draftActions = game.legal_actions.filter(
    (action) => action.kind === "draft",
  );
  const bonusActions = game.legal_actions.filter(
    (action) => action.kind === "bonus",
  );
  const draftAction = (source, color) =>
    draftActions.find(
      (action) => action.source === source && action.color === color,
    );
  return (
    <section
      className="draft-surface"
      aria-label="Factory displays and center market"
    >
      <header className="section-heading">
        <span>
          <span className="eyebrow">Direct drafting</span>
          <h2>Workshops</h2>
        </span>
        <span className="interaction-hint">
          <i /> Click a tile color to take it
        </span>
      </header>
      <div className="factories">
        {game.factories.map((factory, index) => {
          const total = Object.values(factory).reduce(
            (sum, count) => sum + count,
            0,
          );
          return (
            <article
              className={classNames("factory-dish", !total && "is-empty")}
              aria-label={`Workshop ${index + 1}`}
              key={index}
            >
              <span className="factory-number">
                {String(index + 1).padStart(2, "0")}
              </span>
              <div className="factory-tiles">
                {COLORS.filter((color) => factory[color]).map((color) => {
                  const action = draftAction(index, color);
                  const companionWild = Boolean(
                    action && color !== game.wild && factory[game.wild],
                  );
                  return (
                    <TileGroup
                      key={color}
                      color={color}
                      count={factory[color]}
                      action={action}
                      busy={busy}
                      onAction={onAction}
                      companionWild={companionWild}
                      wildColor={game.wild}
                      context={`workshop ${index + 1}`}
                    />
                  );
                })}
                {!total && <span className="empty-copy">Empty</span>}
              </div>
            </article>
          );
        })}
      </div>
      <div className="market-row">
        <article className="center-dish">
          <div className="market-title">
            <span>
              <span className="eyebrow">Shared market</span>
              <strong>Center</strong>
            </span>
            <small>
              {game.next_start_player === null
                ? "First-player marker available"
                : `${game.next_start_player === 0 ? "You" : "AI"} starts next round`}
            </small>
          </div>
          <div className="center-tiles">
            {game.next_start_player === null && (
              <span
                className="first-player-token"
                title="Taking from the center claims next round's start and costs points"
              >
                1
              </span>
            )}
            {COLORS.filter((color) => game.center_pool[color]).map((color) => (
              <TileGroup
                key={color}
                color={color}
                count={game.center_pool[color]}
                action={draftAction(9, color)}
                busy={busy}
                onAction={onAction}
                companionWild={Boolean(
                  draftAction(9, color) &&
                    color !== game.wild &&
                    game.center_pool[game.wild],
                )}
                wildColor={game.wild}
                context="the center"
              />
            ))}
            {!Object.values(game.center_pool).some(Boolean) && (
              <span className="empty-copy">
                Tiles from workshops collect here
              </span>
            )}
          </div>
        </article>
        <article
          className={classNames(
            "supply-tray",
            game.pending_bonus && "is-active",
          )}
        >
          <div className="market-title">
            <span>
              <span className="eyebrow">Architect rewards</span>
              <strong>Supply</strong>
            </span>
            {game.pending_bonus ? (
              <small>Choose {game.pending_bonus} more</small>
            ) : (
              <small>Earned by patterns</small>
            )}
          </div>
          <div className="supply-tiles">
            {COLORS.filter((color) => game.supply[color]).map((color) => (
              <TileGroup
                key={color}
                color={color}
                count={game.supply[color]}
                action={bonusActions.find((action) => action.color === color)}
                busy={busy}
                onAction={onAction}
                context="the supply"
              />
            ))}
          </div>
        </article>
      </div>
    </section>
  );
}

function PaymentChooser({ choice, wild, busy, onAction, onClose }) {
  if (!choice) return null;
  return (
    <section className="payment-chooser" aria-label="Choose tile payment">
      <div>
        <span className="eyebrow">Choose payment</span>
        <strong>{choice.label}</strong>
        <p>
          At least one matching tile is required. Spend more wilds now or
          preserve them for another star.
        </p>
      </div>
      <div className="payment-options">
        {choice.actions.map((action) => {
          const wildCount = action.cost - action.natural;
          const paymentLabel = `Pay ${action.natural} ${choice.color} tile${action.natural === 1 ? "" : "s"}${wildCount ? ` and ${wildCount} ${wild} wild tile${wildCount === 1 ? "" : "s"}` : ""}`;
          return (
            <button
              type="button"
              onClick={() => onAction(action.id)}
              disabled={busy}
              aria-label={paymentLabel}
              title={paymentLabel}
              key={action.id}
            >
              <TileFace color={choice.color} count={action.natural} />
              {wildCount > 0 && (
                <>
                  <span className="payment-plus">+</span>
                  <TileFace color={wild} count={wildCount} />
                </>
              )}
            </button>
          );
        })}
      </div>
      <button
        type="button"
        className="icon-button"
        onClick={onClose}
        aria-label="Close payment choices"
      >
        ×
      </button>
    </section>
  );
}

function TurnConsole({ game, busy, onAction }) {
  const actions = game.legal_actions || [];
  const passAction = actions.find((action) => action.kind === "pass");
  const finishAction = actions.find((action) => action.kind === "keep_finish");
  const isKeeping =
    actions.some((action) => action.kind === "keep") || Boolean(finishAction);
  let title = "Waiting for the AI";
  let copy = "The board will unlock when the AI finishes.";
  let step = "AI";
  if (game.done) {
    title = "Pavilion complete";
    copy = "Start a new game from the header when you are ready.";
    step = "Done";
  } else if (game.pending_bonus) {
    title = "Choose from the supply";
    copy = `Your architecture earned ${game.pending_bonus} bonus tile${game.pending_bonus === 1 ? "" : "s"}. Click a color in the supply tray.`;
    step = "Reward";
  } else if (isKeeping) {
    title = "Keep up to four tiles";
    copy =
      "Click tiles on your inventory shelf to carry them forward, then finish keeping.";
    step = "Pass";
  } else if (game.phase === "draft") {
    title = "Draft from the table";
    copy =
      "Click a color directly inside a workshop or the center. Current-round wilds join a non-wild draft automatically.";
    step = "Draft";
  } else if (game.phase === "place") {
    title = "Build on your board";
    copy =
      "Glowing numbered spaces are legal placements. Click one, then choose a payment if there is more than one way to pay.";
    step = "Place";
  }
  return (
    <section className="turn-console" aria-live="polite">
      <span className="turn-step">{step}</span>
      <div>
        <span className="eyebrow">Your interaction</span>
        <strong>{title}</strong>
        <p>{copy}</p>
      </div>
      <span className="legal-count">
        {actions.length}
        <small>legal moves</small>
      </span>
      {passAction && (
        <button
          type="button"
          className="secondary-button"
          disabled={busy}
          onClick={() => onAction(passAction.id)}
        >
          Pass & choose keeps
        </button>
      )}
      {finishAction && (
        <button
          type="button"
          className="primary"
          disabled={busy}
          onClick={() => onAction(finishAction.id)}
        >
          Finish keeping
        </button>
      )}
    </section>
  );
}

function StatusStrip({ game }) {
  const status = game.done
    ? [
        game.winner === "human"
          ? "You win the pavilion"
          : game.winner === "ai"
            ? `${game.opponent_name} wins`
            : "The game ends in a tie",
        `Final score ${game.players[0].score}–${game.players[1].score}`,
      ]
    : game.pending_bonus && game.current_player === 0
      ? [
          "Architectural reward",
          `Take ${game.pending_bonus} bonus tile${game.pending_bonus === 1 ? "" : "s"} from the supply`,
        ]
      : [
          game.current_player === 0 ? "Your move" : "AI is thinking",
          `${game.phase === "draft" ? "Draft from the workshops" : "Build your pavilion"} · playing ${game.opponent_name}`,
        ];
  return (
    <section className="status-strip" aria-live="polite">
      <div className="round-medallion">
        <span>{game.round}</span>
        <small>round</small>
      </div>
      <div className="status-copy">
        <span className="eyebrow">Now playing</span>
        <p id="status-line">{status[0]}</p>
        <p id="status-detail">{status[1]}</p>
      </div>
      <div className="status-metrics">
        <span>
          <small>Phase</small>
          <b>{game.phase}</b>
        </span>
        <span>
          <small>Wild color</small>
          <b>
            <TileFace color={game.wild || "purple"} small /> {game.wild || "—"}
          </b>
        </span>
      </div>
    </section>
  );
}

function RewardEvent({ events }) {
  if (!events?.length) return null;
  return (
    <section className="reward-event" aria-live="assertive">
      {events.map((event) => (
        <div
          className="reward-event-card"
          key={`${event.player}-${event.kind}-${event.index}`}
        >
          <span className="reward-event-icon">✦</span>
          <span>
            <strong>
              {event.player === 0 ? "You completed" : "AI completed"}{" "}
              {event.name}
            </strong>
            <small>{event.requirement}</small>
          </span>
          <b>
            +{event.reward} tile{event.reward === 1 ? "" : "s"}
          </b>
        </div>
      ))}
    </section>
  );
}

function App() {
  const [game, setGame] = useState(null);
  const [opponents, setOpponents] = useState([]);
  const [opponent, setOpponent] = useState("competitive_hybrid");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [paymentChoice, setPaymentChoice] = useState(null);
  const [focusedFeatureKey, setFocusedFeatureKey] = useState("pillar-0");
  const selectedOpponent = useMemo(
    () => opponents.find((item) => item.id === opponent),
    [opponents, opponent],
  );

  const request = async (path, options = {}) => {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    const text = await response.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error(
        `The game server returned an unreadable response (${response.status}).`,
      );
    }
    if (!response.ok) throw new Error(data.error || "Something went wrong");
    return data;
  };

  const loadGame = async (path, body) => {
    setBusy(true);
    setError("");
    setPaymentChoice(null);
    try {
      const next = await request(path, {
        method: "POST",
        body: JSON.stringify(body),
      });
      setGame(next);
      setOpponents(next.opponents || []);
      if (next.opponent_id) setOpponent(next.opponent_id);
    } catch (caught) {
      setError(caught.message);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    request("/api/state")
      .then((initial) => {
        const availableDefault =
          initial.opponents?.find(
            (item) => item.id === initial.opponent_id && item.available,
          ) ||
          initial.opponents?.find(
            (item) => item.recommended && item.available,
          ) ||
          initial.opponents?.find((item) => item.available);
        const opponentId = availableDefault?.id || "heuristic";
        setOpponents(initial.opponents || []);
        setOpponent(opponentId);
        if (initial.started) setGame(initial);
        else loadGame("/api/new", { opponent: opponentId });
      })
      .catch((caught) => setError(caught.message));
  }, []);

  const playAction = (actionId) =>
    loadGame("/api/action", { action: actionId });
  const choosePlacement = (choice) => {
    if (choice.actions.length === 1) playAction(choice.actions[0].id);
    else setPaymentChoice(choice);
  };

  if (!game)
    return (
      <div className="loading-screen">
        <span className="loading-mark">✦</span>
        {error ? (
          <>
            <strong>Could not prepare the pavilion</strong>
            <p>{error}</p>
            <button
              type="button"
              className="primary"
              onClick={() => window.location.reload()}
            >
              Try again
            </button>
          </>
        ) : (
          <>
            <span className="spinner" />
            <strong>Preparing the pavilion…</strong>
          </>
        )}
      </div>
    );

  return (
    <>
      <a className="skip-link" href="#game-table">
        Skip to game table
      </a>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">✦</span>
          <span>
            <span className="eyebrow">Azul · Summer Pavilion</span>
            <h1>Pavilion Atelier</h1>
          </span>
        </div>
        <div className="game-controls">
          <div className="opponent-field">
            <label htmlFor="opponent-select">
              Opponent{" "}
              {selectedOpponent?.recommended && (
                <span className="best-label">Recommended</span>
              )}
            </label>
            <select
              id="opponent-select"
              value={opponent}
              onChange={(event) => setOpponent(event.target.value)}
              disabled={busy}
            >
              {opponents.map((item) => (
                <option
                  key={item.id}
                  value={item.id}
                  disabled={!item.available}
                >
                  {item.name}
                  {item.score_champion ? " · score champion" : ""}
                </option>
              ))}
            </select>
          </div>
          <button
            type="button"
            className="primary"
            disabled={busy}
            onClick={() => loadGame("/api/new", { opponent })}
          >
            New game
          </button>
        </div>
      </header>
      <main id="game-table">
        <StatusStrip game={game} />
        <RewardEvent events={game.reward_events} />
        <div className="game-layout">
          <div className="left-rail">
            <DraftSurface game={game} busy={busy} onAction={playAction} />
            {game.last_ai_actions?.length > 0 && (
              <aside className="ai-log">
                <span className="ai-log-icon">AI</span>
                <span>
                  <span className="eyebrow">Opponent's last move</span>
                  <p>{game.last_ai_actions.join(" · ")}</p>
                </span>
              </aside>
            )}
            <PlayerPanel
              player={game.players[1]}
              opponentName={game.opponent_name}
              compact
              focusedFeatureKey={null}
            />
          </div>
          <div className="right-stage">
            <PlayerPanel
              player={game.players[0]}
              opponentName={game.opponent_name}
              legalActions={game.legal_actions || []}
              busy={busy}
              onAction={playAction}
              onChooseActions={choosePlacement}
              focusedFeatureKey={focusedFeatureKey}
              onFocusFeature={setFocusedFeatureKey}
            />
            <PaymentChooser
              choice={paymentChoice}
              wild={game.wild}
              busy={busy}
              onAction={playAction}
              onClose={() => setPaymentChoice(null)}
            />
            <TurnConsole game={game} busy={busy} onAction={playAction} />
          </div>
        </div>
      </main>
      {busy && (
        <div className="thinking">
          <div className="thinking-card">
            <span className="spinner" />
            <strong>AI is planning…</strong>
            <small>
              Search opponents can take a little longer late in the game.
            </small>
          </div>
        </div>
      )}
      {error && (
        <div className="toast" role="alert">
          <span>{error}</span>
          <button
            type="button"
            onClick={() => setError("")}
            aria-label="Dismiss error"
          >
            ×
          </button>
        </div>
      )}
    </>
  );
}

createRoot(document.getElementById("root")).render(<App />);
