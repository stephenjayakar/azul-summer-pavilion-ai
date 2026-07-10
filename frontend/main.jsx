import { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "../src/azul/web_static/styles.css";

const COLORS = ["purple", "green", "orange", "yellow", "blue", "red"];

function Tile({ color, count, large = false }) {
  return <span className="tile-stack"><span className={`tile-dot ${color} ${large ? "large-tile" : ""}`} title={color}>{color[0].toUpperCase()}</span>{count !== undefined && <b>{count}</b>}</span>;
}

function TileRow({ counts, showZero = false, large = false }) {
  const colors = COLORS.filter(color => showZero || counts?.[color]);
  return <div className={`tile-row ${large ? "large" : ""}`}>{colors.length ? colors.map(color => <Tile key={color} color={color} count={counts[color] || 0} large={large} />) : <span className="player-subtitle">No tiles</span>}</div>;
}

function FeatureMarker({ feature, label, position }) {
  const almost = !feature.complete && feature.progress === feature.required - 1;
  return <span className={`board-feature-marker ${position} ${feature.complete ? "claimed" : almost ? "near" : ""}`} title={`${feature.name}: ${feature.requirement}. Reward: ${feature.reward} supply tiles. Progress ${feature.progress}/${feature.required}.`}><span>{label}</span><b>+{feature.reward}</b><small>{feature.complete ? "✓" : `${feature.progress}/${feature.required}`}</small></span>;
}

function PlayerBoard({ player, opponentName }) {
  const inventoryTotal = Object.values(player.inventory).reduce((sum, count) => sum + count, 0);
  return <aside className={`player-panel ${player.index ? "ai-panel" : "human-panel"}`} aria-label={`${player.name} player board`}>
    <div className="player-head"><div className="player-identity"><span className="avatar">{player.index ? "AI" : "YOU"}</span><div><p className="player-name">{player.name}</p><p className="player-subtitle">{player.index ? opponentName : "Player 1"}</p></div></div><div className="score-box"><strong>{player.score}</strong><span>POINTS</span></div></div>
    <div className="inventory-block"><div className="mini-title"><span>AVAILABLE TILES</span><span>{inventoryTotal} total</span></div><TileRow counts={player.inventory} /></div>
    <div className="stars-grid">{player.outer.map(star => {
      const features = player.architecture.filter(feature => feature.name.toLowerCase().startsWith(star.color));
      return <div className="star-card" key={star.color}><div className="star-label"><span>{star.color}</span><span>{star.slots.filter(Boolean).length}/6</span></div><div className="star-petals">{star.slots.map((occupied, index) => <span className={`petal slot-${index + 1} ${occupied ? "occupied" : ""}`} key={index}><span className="cost-number">{index + 1}</span>{occupied && <span className={`placed-tile ${star.color}`}>{star.color[0].toUpperCase()}</span>}</span>)}
        {features.find(f => f.kind === "statue") && <FeatureMarker feature={features.find(f => f.kind === "statue")} label="S" position="statue-marker" />}
        {features.find(f => f.kind === "pillar") && <FeatureMarker feature={features.find(f => f.kind === "pillar")} label="P" position="pillar-marker" />}
        {features.find(f => f.kind === "window") && <FeatureMarker feature={features.find(f => f.kind === "window")} label="W" position="window-marker" />}
      </div></div>;
    })}</div>
    <div className="center-star"><div className="star-label"><span>Center star</span><span>{player.center.filter(Boolean).length}/6</span></div><div className="center-slots">{player.center.map((color, index) => <span className={`center-slot ${color ? "filled" : ""}`} key={index} title={color ? `${color} tile on center cost ${index + 1}` : `Center cost ${index + 1}`}><span className="cost-number">{index + 1}</span>{color && <span className={`placed-tile ${color}`}>{color[0].toUpperCase()}</span>}</span>)}</div></div>
    <div className="architecture-progress"><div className="architecture-progress-title"><span>Architectural rewards</span><span>{player.architecture.filter(f => f.complete).length}/18 claimed</span></div><p className="architecture-inline-note">Markers show the patterns to complete: S = statue, P = pillar, W = window.</p></div>
  </aside>;
}

function actionLabel(action) {
  if (action.kind === "draft") return [`Take ${action.color}`, action.source === 9 ? "From center" : `From factory ${action.source + 1}`];
  if (action.kind === "place_outer") return [`${action.star} · cost ${action.cost}`, `${action.natural} natural · ${action.cost - action.natural} wild`];
  if (action.kind === "place_center") return [`Center ${action.color} · cost ${action.cost}`, `${action.natural} natural · ${action.cost - action.natural} wild`];
  if (action.kind === "bonus") return [`Take ${action.color}`, "Bonus tile from supply"];
  if (action.kind === "keep") return [`Keep ${action.color}`, "Carry into next round"];
  return [action.kind === "keep_finish" ? "Finish keeping" : "Pass", action.kind === "keep_finish" ? "Discard remaining tiles" : "Choose up to four tiles to keep"];
}

function ActionDock({ game, busy, onAction }) {
  const [selected, setSelected] = useState(null);
  useEffect(() => setSelected(null), [game.phase, game.pending_bonus, game.legal_actions]);
  const actions = game.legal_actions || [];
  const [title, help] = game.pending_bonus ? ["Choose your reward", `Take ${game.pending_bonus} more bonus tile${game.pending_bonus === 1 ? "" : "s"} from the supply.`] : game.phase === "draft" ? ["Choose a tile group", "Pick a color from a workshop or the center. Wild tiles join a non-wild draft when available."] : ["Build your pavilion", "Choose where to spend tiles, then pass when you are ready."];
  const selectedAction = actions.find(action => action.id === selected);
  return <section className="action-dock" aria-label="Legal moves"><div className="action-header"><div><p className="eyebrow">YOUR TURN</p><h2>{title}</h2></div><div className="action-count">{actions.length} legal move{actions.length === 1 ? "" : "s"}</div></div><div className="action-help">{help}</div>
    {!actions.length ? <div className="empty-actions">{game.done ? "Game complete — start a new match when you’re ready." : "Waiting for the AI…"}</div> : <><div className="actions">{actions.map(action => { const [main, meta] = actionLabel(action); return <button key={action.id} className={`action-button ${action.kind} ${selected === action.id ? "selected" : ""}`} onClick={() => setSelected(action.id)} disabled={busy} title={action.description}>{action.color ? <Tile color={action.color} /> : <span className="action-icon">{action.kind === "pass" ? "↷" : "✓"}</span>}<span><span className="action-main">{main}</span><span className="action-meta">{meta}</span></span></button>; })}</div>
    <div className="move-confirm"><span>{selectedAction ? selectedAction.description : "Select a move to review it before sending."}</span><button className="primary" disabled={!selectedAction || busy} onClick={() => onAction(selectedAction.id)}>{busy ? "Thinking…" : "Play selected move"}</button></div></>}</section>;
}

function App() {
  const [game, setGame] = useState(null); const [opponents, setOpponents] = useState([]); const [opponent, setOpponent] = useState("competitive_hybrid"); const [busy, setBusy] = useState(false); const [error, setError] = useState("");
  const selectedOpponent = useMemo(() => opponents.find(item => item.id === opponent), [opponents, opponent]);
  const request = async (path, options = {}) => { const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options }); const data = await response.json(); if (!response.ok) throw new Error(data.error || "Something went wrong"); return data; };
  const loadGame = async (path, body) => { setBusy(true); try { const next = await request(path, { method: "POST", body: JSON.stringify(body) }); setGame(next); setOpponents(next.opponents || []); } catch (e) { setError(e.message); } finally { setBusy(false); } };
  useEffect(() => { request("/api/state").then(initial => { setOpponents(initial.opponents || []); setOpponent(initial.opponent_id || "competitive_hybrid"); if (initial.started) setGame(initial); else loadGame("/api/new", { opponent: initial.opponent_id || "competitive_hybrid" }); }).catch(e => setError(e.message)); }, []);
  if (!game) return <div className="loading-screen"><span className="spinner" /><strong>Preparing the pavilion…</strong></div>;
  const status = game.done ? [game.winner === "human" ? "You win the pavilion!" : game.winner === "ai" ? `${game.opponent_name} wins.` : "The game ends in a tie.", `Final score: ${game.players[0].score}–${game.players[1].score}`] : game.pending_bonus && game.current_player === 0 ? ["Architectural reward — choose from the supply", `Take ${game.pending_bonus} bonus tile${game.pending_bonus === 1 ? "" : "s"}.`] : [game.current_player === 0 ? "Your move" : "AI is thinking", `${game.phase === "draft" ? "Draft from the workshops." : "Place tiles on your pavilion."} Playing ${game.opponent_name}.`];
  return <><header className="topbar"><div className="brand"><span className="brand-mark">✦</span><div><p className="eyebrow">AZUL · SUMMER PAVILION</p><h1>Play the AI</h1></div></div><div className="game-controls"><div className="opponent-field"><label htmlFor="opponent-select">Opponent {selectedOpponent?.recommended && <span className="best-label">★ BEST 1v1</span>}</label><select id="opponent-select" value={opponent} onChange={e => setOpponent(e.target.value)} disabled={busy}>{opponents.map(item => <option key={item.id} value={item.id} disabled={!item.available}>{item.name}{item.score_champion ? " — highest score" : ""}</option>)}</select><small>{selectedOpponent?.description}</small></div><button className="primary" disabled={busy} onClick={() => loadGame("/api/new", { opponent })}>New game</button></div></header>
  <main><section className="status-strip" aria-live="polite"><div className="round-medallion"><span>{game.round}</span><small>ROUND</small></div><div className="status-copy"><p id="status-line">{status[0]}</p><p id="status-detail">{status[1]}</p></div><div className="wild-badge"><Tile color={game.wild || "purple"} /><span>Wild: {game.wild || "—"}</span></div></section>
  {!!game.reward_events?.length && <section className="reward-event" aria-live="assertive">{game.reward_events.map(event => <div className="reward-event-card" key={`${event.player}-${event.kind}-${event.index}`}><span className="reward-event-icon">✦</span><span className="reward-event-copy"><strong>{event.player === 0 ? "You completed" : "AI completed"} {event.name}</strong><span>{event.requirement}</span></span><span className="reward-event-value">+{event.reward} supply tile{event.reward === 1 ? "" : "s"}</span></div>)}</section>}
  <section className="table-layout"><PlayerBoard player={game.players[1]} opponentName={game.opponent_name} /><section className="shared-board"><div className="section-heading"><div><p className="eyebrow">WORKSHOPS</p><h2>Factory displays</h2></div><span className="phase-pill">{game.phase} phase</span></div><div className="factories">{game.factories.map((factory, index) => <div className={`factory ${Object.values(factory).reduce((a,b) => a + b, 0) ? "" : "empty"}`} key={index}>{COLORS.flatMap(color => Array.from({ length: factory[color] || 0 }, (_, i) => <Tile color={color} key={`${color}-${i}`} />)) || "Empty"}</div>)}</div><div className="market-grid"><div className="market-card center-market"><div className="market-title"><span>Center</span><small>{game.next_start_player === null ? "First-player token available" : `${game.next_start_player === 0 ? "You" : "AI"} holds next-round start`}</small></div><TileRow counts={game.center_pool} large /></div><div className="market-card"><div className="market-title"><span>Supply</span><small>Architectural rewards</small></div><TileRow counts={game.supply} /></div></div><div className="architecture-legend"><div className="architecture-legend-title"><span>✦</span><div><strong>Architectural rewards</strong><small>Complete patterns to draft bonus tiles from the supply.</small></div></div>{[["+3", "Window", "Fill costs 5 + 6"], ["+2", "Statue", "Fill a star junction"], ["+1", "Pillar", "Fill the outer and center pair"]].map(([points, name, text]) => <div className="architecture-rule" key={name}><b>{points}</b><span><strong>{name}</strong><small>{text}</small></span></div>)}</div>{game.last_ai_actions?.length > 0 && <div className="ai-log"><span className="ai-log-icon">AI</span><div><strong>Last move</strong><p>{game.last_ai_actions.join(" · ")}</p></div></div>}</section><PlayerBoard player={game.players[0]} opponentName={game.opponent_name} /></section><ActionDock game={game} busy={busy} onAction={id => loadGame("/api/action", { action: id })} /></main>
  {busy && <div className="thinking"><div className="thinking-card"><span className="spinner" /><strong>AI is planning…</strong><small>Search opponents can take a little longer late in the game.</small></div></div>}{error && <div className="toast" role="alert">{error}<button onClick={() => setError("")}>×</button></div>}</>;
}

createRoot(document.getElementById("root")).render(<App />);
