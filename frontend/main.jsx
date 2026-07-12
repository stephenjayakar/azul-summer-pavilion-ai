import { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const COLORS = ["purple", "green", "orange", "yellow", "blue", "red"];
const WILD_ORDER = COLORS;
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
const MOTION_STORAGE_KEY = "pavilion-atelier-animations";

function initialMotionPreference() {
  try {
    const stored = window.localStorage.getItem(MOTION_STORAGE_KEY);
    if (stored !== null) return stored === "true";
  } catch {
    // Storage can be unavailable in privacy-restricted browser contexts.
  }
  return !window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

function colorCountsChanged(previous = {}, next = {}) {
  return COLORS.some((color) => previous[color] !== next[color]);
}

function describeGameChanges(previous, next) {
  const changes = {
    board: new Set(),
    scores: new Set(),
    inventories: new Set(),
    factories: new Set(),
    center: colorCountsChanged(previous.center_pool, next.center_pool),
    supply: colorCountsChanged(previous.supply, next.supply),
  };

  next.players.forEach((player, playerIndex) => {
    const priorPlayer = previous.players[playerIndex];
    if (player.score !== priorPlayer.score) changes.scores.add(playerIndex);
    if (
      colorCountsChanged(priorPlayer.inventory, player.inventory) ||
      colorCountsChanged(priorPlayer.stored, player.stored)
    ) {
      changes.inventories.add(playerIndex);
    }
    player.outer.forEach((star, starIndex) => {
      star.slots.forEach((occupied, slot) => {
        if (occupied && !priorPlayer.outer[starIndex].slots[slot]) {
          changes.board.add(`${playerIndex}-${star.color}-${slot}`);
        }
      });
    });
    player.center.forEach((color, slot) => {
      if (color && !priorPlayer.center[slot]) {
        changes.board.add(`${playerIndex}-center-${slot}`);
      }
    });
  });

  next.factories.forEach((factory, index) => {
    if (colorCountsChanged(previous.factories[index], factory)) {
      changes.factories.add(index);
    }
  });
  return changes;
}

function actionConfirmation(action) {
  if (!action) return "Move complete";
  if (action.kind === "draft") return `${action.color} tiles drafted`;
  if (action.kind === "place_outer") {
    return `Tile placed · ${action.star} ${action.cost}`;
  }
  if (action.kind === "place_center") {
    return `Tile placed · center ${action.cost}`;
  }
  if (action.kind === "bonus") return `${action.color} reward claimed`;
  if (action.kind === "keep") return `${action.color} tile kept`;
  if (action.kind === "keep_finish") return "Keeps confirmed";
  if (action.kind === "pass") return "Placement turn complete";
  return "Move complete";
}

function elementRect(element) {
  if (!element) return null;
  const rect = element.getBoundingClientRect();
  return {
    left: rect.left,
    top: rect.top,
    width: rect.width,
    height: rect.height,
  };
}

function captureAiFlightSpecs(descriptions = []) {
  return descriptions.flatMap((description) => {
    let match = description.match(/^take (\w+) from (center|factory (\d+))$/);
    if (match) {
      const [, color, source, factoryNumber] = match;
      const sourceIndex = source === "center" ? 9 : Number(factoryNumber) - 1;
      const sourceElement = document.querySelector(
        `[data-source-index="${sourceIndex}"] [data-tile-color="${color}"]`,
      );
      const from = elementRect(sourceElement);
      if (!from) return [];
      return [{
        color,
        count: Number(sourceElement.dataset.tileCount) || 1,
        from,
        targets: [
          `.opponent-pavilion .inventory-shelf [data-tile-color="${color}"]`,
          ".opponent-pavilion .inventory-tiles",
        ],
      }];
    }

    match = description.match(/^place (\w+) in center cost (\d+)/);
    if (match) {
      const [, color, cost] = match;
      const from = elementRect(document.querySelector(
        `.opponent-pavilion .inventory-shelf [data-tile-color="${color}"]`,
      ));
      if (!from) return [];
      return [{
        color,
        count: 1,
        from,
        targets: [
          `.board-slot[data-player-index="1"][data-board-star="center"][data-board-slot="${Number(cost) - 1}"]`,
        ],
      }];
    }

    match = description.match(/^place (\w+) on cost (\d+)/);
    if (match) {
      const [, color, cost] = match;
      const from = elementRect(document.querySelector(
        `.opponent-pavilion .inventory-shelf [data-tile-color="${color}"]`,
      ));
      if (!from) return [];
      return [{
        color,
        count: 1,
        from,
        targets: [
          `.board-slot[data-player-index="1"][data-board-star="${color}"][data-board-slot="${Number(cost) - 1}"]`,
        ],
      }];
    }

    match = description.match(/^take (\w+) bonus tile$/);
    if (match) {
      const color = match[1];
      const from = elementRect(document.querySelector(
        `[data-source-index="supply"] [data-tile-color="${color}"]`,
      ));
      if (!from) return [];
      return [{
        color,
        count: 1,
        from,
        targets: [
          `.opponent-pavilion .inventory-shelf [data-tile-color="${color}"]`,
          ".opponent-pavilion .inventory-tiles",
        ],
      }];
    }

    match = description.match(/^keep one (\w+)$/);
    if (match) {
      const color = match[1];
      const from = elementRect(document.querySelector(
        `.opponent-pavilion .inventory-shelf [data-tile-color="${color}"]`,
      ));
      if (!from) return [];
      return [{
        color,
        count: 1,
        from,
        targets: [
          ".opponent-pavilion .stored-row",
          ".opponent-pavilion .inventory-shelf",
        ],
      }];
    }
    return [];
  });
}

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
      <span
        className="tile-group is-static"
        data-tile-color={color}
        data-tile-count={count}
        aria-label={label}
        title={label}
      >
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
      data-tile-color={color}
      data-tile-count={count}
      disabled={!actionable}
      onClick={(event) =>
        onAction(action.id, {
          sourceElement: event.currentTarget,
          color,
          count,
        })
      }
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
  changed = false,
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
      className={classNames("inventory-shelf", changed && "is-updated")}
      data-player-index={player.index}
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
  playerIndex,
  star,
  color,
  slot,
  position,
  occupied,
  centerColor,
  actions,
  busy,
  highlighted,
  justPlaced,
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
        justPlaced && "is-newly-placed",
      )}
      data-player-index={playerIndex}
      data-board-star={star}
      data-board-slot={slot}
      onClick={
        actionable
          ? (event) =>
              onChooseActions({
                label,
                actions,
                color: star === "center" ? actions[0].color : color,
                cost,
                targetRect: elementRect(event.currentTarget),
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
  recentBoardChanges,
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
            playerIndex={player.index}
            star={star.color}
            color={star.color}
            slot={slot}
            position={position}
            occupied={occupied}
            actions={actions}
            busy={busy}
            highlighted={highlightedCells.has(cellKey(star.color, slot))}
            justPlaced={recentBoardChanges?.has(
              `${player.index}-${star.color}-${slot}`,
            )}
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
  recentBoardChanges,
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
            playerIndex={player.index}
            star="center"
            slot={slot}
            position={slot + 1}
            occupied={Boolean(color)}
            centerColor={color}
            actions={actions}
            busy={busy}
            highlighted={highlightedCells.has(cellKey("center", slot))}
            justPlaced={recentBoardChanges?.has(
              `${player.index}-center-${slot}`,
            )}
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
  recentBoardChanges,
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
          recentBoardChanges={recentBoardChanges}
          onChooseActions={onChooseActions}
        />
      ))}
      <CenterStar
        player={player}
        legalActions={legalActions}
        busy={busy}
        highlightedCells={highlightedCells}
        recentBoardChanges={recentBoardChanges}
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
  recentChanges,
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
        <div
          className={classNames(
            "score-box",
            recentChanges?.scores.has(player.index) && "is-updated",
          )}
        >
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
        changed={recentChanges?.inventories.has(player.index)}
      />
      <PavilionMap
        player={player}
        legalActions={player.index ? [] : legalActions}
        busy={busy}
        compact={compact}
        focusedFeatureKey={focusedFeatureKey}
        onFocusFeature={onFocusFeature}
        onChooseActions={onChooseActions}
        recentBoardChanges={recentChanges?.board}
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

function DraftSurface({ game, busy, onAction, recentChanges }) {
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
              className={classNames(
                "factory-dish",
                !total && "is-empty",
                recentChanges?.factories.has(index) && "is-updated",
              )}
              aria-label={`Workshop ${index + 1}`}
              data-source-index={index}
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
        <article
          className={classNames(
            "center-dish",
            recentChanges?.center && "is-updated",
          )}
          data-source-index="9"
        >
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
            recentChanges?.supply && "is-updated",
          )}
          data-source-index="supply"
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
              onClick={() =>
                onAction(action.id, {
                  color: choice.color,
                  targetRect: choice.targetRect,
                })
              }
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

function MotionToggle({ enabled, onChange, previewing }) {
  return (
    <label
      className={classNames(
        "motion-control",
        previewing && "is-previewing",
      )}
    >
      <span className="motion-control-copy">
        <span>Animations</span>
        <small>{enabled ? "On" : "Off"}</small>
      </span>
      <input
        type="checkbox"
        checked={enabled}
        onChange={(event) => onChange(event.target.checked)}
        aria-label="Enable action animations"
      />
      <span className="motion-switch" aria-hidden="true">
        <span>✦</span>
      </span>
    </label>
  );
}

function WildOrder({ round }) {
  return (
    <section className="wild-order" aria-label="Wildcard order by round">
      <span className="wild-order-label">
        <span>Wild order</span>
        <small>rounds 1–6</small>
      </span>
      <ol>
        {WILD_ORDER.map((color, index) => {
          const roundNumber = index + 1;
          return (
            <li
              className={classNames(
                index < round - 1 && "is-past",
                index === round - 1 && "is-current",
              )}
              title={`Round ${roundNumber}: ${color} is wild`}
              aria-current={index === round - 1 ? "step" : undefined}
              key={color}
            >
              <span className={classNames("wild-order-tile", color)}>
                {tileLetter(color)}
              </span>
              <small>{roundNumber}</small>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function TileFlight({ flight }) {
  if (!flight) return null;
  return (
    <span
      className="tile-flight"
      style={{
        "--flight-x": `${flight.x}px`,
        "--flight-y": `${flight.y}px`,
        "--flight-dx": `${flight.dx}px`,
        "--flight-dy": `${flight.dy}px`,
        "--flight-mx": `${flight.mx}px`,
        "--flight-my": `${flight.my}px`,
      }}
      aria-hidden="true"
      key={flight.sequence}
    >
      <span className={classNames("tile-flight-face", flight.color)}>
        {tileLetter(flight.color)}
      </span>
      {flight.count > 1 && <b>×{flight.count}</b>}
    </span>
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
  const [motionEnabled, setMotionEnabled] = useState(initialMotionPreference);
  const [actionPhase, setActionPhase] = useState("");
  const [recentChanges, setRecentChanges] = useState(null);
  const [actionCue, setActionCue] = useState(null);
  const [showThinking, setShowThinking] = useState(false);
  const [motionPreview, setMotionPreview] = useState(false);
  const [tileFlight, setTileFlight] = useState(null);
  const [pendingAiFlightSpecs, setPendingAiFlightSpecs] = useState([]);
  const actionTimer = useRef(null);
  const thinkingTimer = useRef(null);
  const previewTimer = useRef(null);
  const flightTimer = useRef(null);
  const activeFlight = useRef(null);
  const aiFlightTimers = useRef([]);
  const actionSequence = useRef(0);
  const flightSequence = useRef(0);
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

  useEffect(() => {
    document.documentElement.classList.toggle(
      "motion-disabled",
      !motionEnabled,
    );
    try {
      window.localStorage.setItem(
        MOTION_STORAGE_KEY,
        String(motionEnabled),
      );
    } catch {
      // The preference still works for the current session without storage.
    }
  }, [motionEnabled]);

  useEffect(
    () => () => {
      if (actionTimer.current) window.clearTimeout(actionTimer.current);
      if (thinkingTimer.current) window.clearTimeout(thinkingTimer.current);
      if (previewTimer.current) window.clearTimeout(previewTimer.current);
      if (flightTimer.current) window.clearTimeout(flightTimer.current);
      aiFlightTimers.current.forEach((timer) => window.clearTimeout(timer));
    },
    [],
  );

  useEffect(() => {
    if (thinkingTimer.current) window.clearTimeout(thinkingTimer.current);
    if (!busy) {
      setShowThinking(false);
      return undefined;
    }
    thinkingTimer.current = window.setTimeout(
      () => setShowThinking(true),
      450,
    );
    return () => window.clearTimeout(thinkingTimer.current);
  }, [busy]);

  const loadGame = async (path, body, animateAction = false) => {
    if (actionTimer.current) window.clearTimeout(actionTimer.current);
    aiFlightTimers.current.forEach((timer) => window.clearTimeout(timer));
    aiFlightTimers.current = [];
    setPendingAiFlightSpecs([]);
    setActionPhase(animateAction ? "is-action-pending" : "");
    if (!animateAction) {
      setRecentChanges(null);
      setActionCue(null);
      activeFlight.current = null;
      setTileFlight(null);
    }
    setBusy(true);
    setError("");
    setPaymentChoice(null);
    try {
      const next = await request(path, {
        method: "POST",
        body: JSON.stringify(body),
      });
      const aiFlightSpecs =
        animateAction && motionEnabled
          ? captureAiFlightSpecs(next.last_ai_actions)
          : [];
      setGame(next);
      setPendingAiFlightSpecs(aiFlightSpecs);
      if (animateAction) {
        const playedAction = game?.legal_actions?.find(
          (action) => action.id === body.action,
        );
        actionSequence.current += 1;
        setRecentChanges(describeGameChanges(game, next));
        setActionCue({
          sequence: actionSequence.current,
          label: actionConfirmation(playedAction),
        });
        setActionPhase("is-action-complete");
        actionTimer.current = window.setTimeout(
          () => {
            setActionPhase("");
            setRecentChanges(null);
            setActionCue(null);
          },
          1550,
        );
        if (activeFlight.current) {
          const elapsed = Date.now() - activeFlight.current.startedAt;
          flightTimer.current = window.setTimeout(
            () => {
              activeFlight.current = null;
              setTileFlight(null);
            },
            Math.max(180, 880 - elapsed),
          );
        }
      }
      setOpponents(next.opponents || []);
      if (next.opponent_id) setOpponent(next.opponent_id);
    } catch (caught) {
      setActionPhase("");
      setRecentChanges(null);
      setActionCue(null);
      activeFlight.current = null;
      setTileFlight(null);
      setPendingAiFlightSpecs([]);
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

  const createTileFlight = (color, count, from, to) => {
    const size = 34;
    const x = from.left + from.width / 2 - size / 2;
    const y = from.top + from.height / 2 - size / 2;
    const destinationX = to.left + to.width / 2 - size / 2;
    const destinationY = to.top + to.height / 2 - size / 2;
    const dx = destinationX - x;
    const dy = destinationY - y;
    const arc = Math.min(100, Math.max(42, Math.abs(dx) * 0.12 + 42));
    flightSequence.current += 1;
    return {
      sequence: flightSequence.current,
      color,
      count: count || 1,
      x,
      y,
      dx,
      dy,
      mx: dx / 2,
      my: dy / 2 - arc,
      startedAt: Date.now(),
    };
  };

  const startTileFlight = (action, context = {}) => {
    if (!motionEnabled || !action) return;
    const color = context.color || action.color || action.star;
    if (!COLORS.includes(color)) return;

    const isPlacement = action.kind.startsWith("place");
    const sourceElement = isPlacement
      ? document.querySelector(
          `.your-pavilion .inventory-shelf [data-tile-color="${color}"]`,
        )
      : context.sourceElement;
    const from = elementRect(sourceElement);

    let to = context.targetRect;
    if (!to && (action.kind === "draft" || action.kind === "bonus")) {
      const matchingInventory = document.querySelector(
        `.your-pavilion .inventory-shelf [data-tile-color="${color}"]`,
      );
      to = elementRect(
        matchingInventory ||
          document.querySelector(
            '.your-pavilion .inventory-shelf .inventory-tiles',
          ),
      );
    }
    if (!to && action.kind === "keep") {
      to = elementRect(
        document.querySelector('.your-pavilion .stored-row') ||
          document.querySelector('.your-pavilion .inventory-shelf'),
      );
    }
    if (!from || !to) return;

    if (flightTimer.current) window.clearTimeout(flightTimer.current);
    const flight = createTileFlight(color, context.count, from, to);
    activeFlight.current = flight;
    setTileFlight(flight);
  };

  useEffect(() => {
    if (!motionEnabled || !pendingAiFlightSpecs.length) return;

    const flights = pendingAiFlightSpecs.flatMap((spec) => {
      const targetElement = spec.targets
        .map((selector) => document.querySelector(selector))
        .find(Boolean);
      const to = elementRect(targetElement);
      return to
        ? [createTileFlight(spec.color, spec.count, spec.from, to)]
        : [];
    });
    setPendingAiFlightSpecs([]);
    if (!flights.length) return;

    const schedule = (callback, delay) => {
      const timer = window.setTimeout(callback, delay);
      aiFlightTimers.current.push(timer);
    };
    const playFlight = (index) => {
      const flight = { ...flights[index], startedAt: Date.now() };
      activeFlight.current = flight;
      setTileFlight(flight);
      schedule(() => {
        activeFlight.current = null;
        setTileFlight(null);
        if (index + 1 < flights.length) {
          schedule(() => playFlight(index + 1), 130);
        }
      }, 830);
    };

    const humanFlightDelay = activeFlight.current
      ? Math.max(180, 880 - (Date.now() - activeFlight.current.startedAt)) + 130
      : 90;
    schedule(() => playFlight(0), humanFlightDelay);
  }, [pendingAiFlightSpecs, motionEnabled]);

  const playAction = (actionId, context = {}) => {
    const action = game?.legal_actions?.find((item) => item.id === actionId);
    startTileFlight(action, context);
    loadGame("/api/action", { action: actionId }, true);
  };
  const changeMotionPreference = (enabled) => {
    document.documentElement.classList.toggle("motion-disabled", !enabled);
    setMotionEnabled(enabled);
    if (!enabled) {
      activeFlight.current = null;
      setTileFlight(null);
      setPendingAiFlightSpecs([]);
      aiFlightTimers.current.forEach((timer) => window.clearTimeout(timer));
      aiFlightTimers.current = [];
    }
    if (previewTimer.current) window.clearTimeout(previewTimer.current);
    setMotionPreview(enabled);
    if (enabled) {
      previewTimer.current = window.setTimeout(
        () => setMotionPreview(false),
        1400,
      );
    }
  };
  const choosePlacement = (choice) => {
    if (choice.actions.length === 1) {
      playAction(choice.actions[0].id, {
        color: choice.color,
        targetRect: choice.targetRect,
      });
    }
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
          <WildOrder round={game.round} />
          <MotionToggle
            enabled={motionEnabled}
            onChange={changeMotionPreference}
            previewing={motionPreview}
          />
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
      <main id="game-table" className={actionPhase}>
        <TileFlight flight={tileFlight} />
        {motionPreview && (
          <div className="motion-preview" role="status">
            <span aria-hidden="true">✦</span>
            <small>Animations are on</small>
          </div>
        )}
        {actionCue && (
          <div
            className="action-confirmation"
            role="status"
            key={actionCue.sequence}
          >
            <span aria-hidden="true">✦</span>
            <small>{actionCue.label}</small>
          </div>
        )}
        <StatusStrip game={game} />
        <RewardEvent events={game.reward_events} />
        <div className="game-layout">
          <div className="left-rail">
            <DraftSurface
              game={game}
              busy={busy}
              onAction={playAction}
              recentChanges={recentChanges}
            />
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
              recentChanges={recentChanges}
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
              recentChanges={recentChanges}
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
      {showThinking && (
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
