const COLORS = ["purple", "green", "orange", "yellow", "blue", "red"];
const state = { game: null, busy: false };

const $ = (id) => document.getElementById(id);

function tile(color, count = null, large = false) {
  const stack = document.createElement("span");
  stack.className = "tile-stack";
  const dot = document.createElement("span");
  dot.className = `tile-dot ${color}`;
  dot.textContent = color[0].toUpperCase();
  dot.title = color;
  if (large) dot.style.width = "28px";
  stack.append(dot);
  if (count !== null) {
    const number = document.createElement("b");
    number.textContent = count;
    stack.append(number);
  }
  return stack;
}

function fillTileRow(element, counts, showZero = false) {
  element.replaceChildren();
  let shown = 0;
  for (const color of COLORS) {
    const count = counts[color] || 0;
    if (count || showZero) {
      element.append(tile(color, count));
      shown += count;
    }
  }
  if (!shown && !showZero) {
    const empty = document.createElement("span");
    empty.className = "player-subtitle";
    empty.textContent = "No tiles";
    element.append(empty);
  }
}

function renderPlayer(player, target, opponentName) {
  target.replaceChildren();
  const head = document.createElement("div");
  head.className = "player-head";
  head.innerHTML = `
    <div class="player-identity"><span class="avatar">${player.index === 0 ? "YOU" : "AI"}</span><div>
      <p class="player-name">${player.name}</p><p class="player-subtitle">${player.index === 0 ? "Player 1" : opponentName}</p>
    </div></div><div class="score-box"><strong>${player.score}</strong><span>POINTS</span></div>`;
  target.append(head);

  const inventory = document.createElement("div");
  inventory.className = "inventory-block";
  inventory.innerHTML = `<div class="mini-title"><span>AVAILABLE TILES</span><span>${Object.values(player.inventory).reduce((a,b)=>a+b,0)} total</span></div>`;
  const row = document.createElement("div"); row.className = "tile-row";
  fillTileRow(row, player.inventory); inventory.append(row); target.append(inventory);

  const stars = document.createElement("div"); stars.className = "stars-grid";
  for (const star of player.outer) {
    const card = document.createElement("div"); card.className = "star-card";
    const filled = star.slots.filter(Boolean).length;
    card.innerHTML = `<div class="star-label"><span>${star.color}</span><span>${filled}/6</span></div>`;
    const petals = document.createElement("div"); petals.className = "star-petals";
    star.slots.forEach((occupied, index) => {
      const petal = document.createElement("span");
      petal.className = `petal${occupied ? ` filled ${star.color}` : ""}`;
      petal.textContent = index + 1; petals.append(petal);
    });
    card.append(petals); stars.append(card);
  }
  target.append(stars);

  const center = document.createElement("div"); center.className = "center-star";
  center.innerHTML = `<div class="star-label"><span>Center star</span><span>${player.center.filter(Boolean).length}/6</span></div>`;
  const slots = document.createElement("div"); slots.className = "center-slots";
  player.center.forEach((color, index) => {
    const slot = document.createElement("span");
    slot.className = `center-slot${color ? ` filled ${color}` : ""}`;
    slot.textContent = index + 1; slot.title = color || `Cost ${index + 1}`; slots.append(slot);
  });
  center.append(slots); target.append(center);

  const features = document.createElement("div"); features.className = "feature-row";
  if (player.claimed.length) {
    for (const feature of player.claimed) {
      const badge = document.createElement("span"); badge.className = "feature-badge";
      badge.textContent = feature.kind; badge.title = `${feature.kind} ${feature.index + 1}`; features.append(badge);
    }
  } else {
    features.innerHTML = `<span class="player-subtitle">No architectural rewards yet</span>`;
  }
  target.append(features);
}

function renderFactories(factories) {
  const root = $("factories"); root.replaceChildren();
  factories.forEach((factory, index) => {
    const display = document.createElement("div");
    const total = Object.values(factory).reduce((a,b)=>a+b,0);
    display.className = `factory${total ? "" : " empty"}`;
    display.title = `Factory ${index + 1}`;
    for (const color of COLORS) {
      for (let i = 0; i < factory[color]; i++) display.append(tile(color));
    }
    if (!total) display.textContent = "Empty";
    root.append(display);
  });
}

function actionLabel(action) {
  if (action.kind === "draft") return { main: `Take ${action.color}`, meta: action.source === 9 ? "From center" : `From factory ${action.source + 1}` };
  if (action.kind === "place_outer") return { main: `${action.star} · cost ${action.cost}`, meta: `${action.natural} natural · ${action.cost - action.natural} wild` };
  if (action.kind === "place_center") return { main: `Center ${action.color} · cost ${action.cost}`, meta: `${action.natural} natural · ${action.cost - action.natural} wild` };
  if (action.kind === "bonus") return { main: `Take ${action.color}`, meta: "Bonus tile from supply" };
  if (action.kind === "keep") return { main: `Keep ${action.color}`, meta: "Carry into next round" };
  if (action.kind === "keep_finish") return { main: "Finish keeping", meta: "Discard all remaining tiles" };
  return { main: "Pass", meta: "Choose up to four tiles to keep" };
}

function renderActions(game) {
  const root = $("actions"); root.replaceChildren();
  const actions = game.legal_actions || [];
  $("action-count").textContent = `${actions.length} legal move${actions.length === 1 ? "" : "s"}`;
  const title = game.phase === "draft" ? "Choose a tile group" : "Build your pavilion";
  $("action-title").textContent = game.pending_bonus ? "Choose your reward" : title;
  $("action-help").textContent = game.phase === "draft"
    ? "Choose one color from a factory or the center. A non-wild choice also takes one wild tile when present."
    : "Place tiles, then pass when you are ready. Cost is the number printed on that petal.";

  if (!actions.length) {
    const empty = document.createElement("div"); empty.className = "empty-actions";
    empty.textContent = game.done ? "Game complete — start a new game for another match." : "Waiting for the AI…";
    root.append(empty); return;
  }
  for (const action of actions) {
    const button = document.createElement("button"); button.className = `action-button ${action.kind}`;
    const label = actionLabel(action);
    if (action.color) button.append(tile(action.color));
    else {
      const icon = document.createElement("span"); icon.className = "action-icon";
      icon.textContent = action.kind === "pass" ? "↷" : "✓"; button.append(icon); button.classList.add("utility");
    }
    const copy = document.createElement("span");
    copy.innerHTML = `<span class="action-main">${label.main}</span><span class="action-meta">${label.meta}</span>`;
    button.append(copy); button.title = action.description;
    button.addEventListener("click", () => playAction(action.id));
    root.append(button);
  }
}

function render(game) {
  state.game = game;
  if (!game.started) return;
  $("round-number").textContent = game.round;
  $("phase-pill").textContent = `${game.phase[0].toUpperCase()}${game.phase.slice(1)} phase`;
  const wild = $("wild-badge");
  wild.replaceChildren(tile(game.wild || "purple"), document.createTextNode(`Wild: ${game.wild || "—"}`));
  renderPlayer(game.players[1], $("ai-board"), game.opponent_name);
  renderPlayer(game.players[0], $("human-board"), game.opponent_name);
  renderFactories(game.factories);
  fillTileRow($("center-pool"), game.center_pool);
  fillTileRow($("supply"), game.supply);
  $("token-state").textContent = game.next_start_player === null
    ? "First-player token available"
    : `${game.next_start_player === 0 ? "You hold" : "AI holds"} next-round start`;

  const log = $("ai-log");
  if (game.last_ai_actions.length) {
    log.hidden = false; $("ai-log-text").textContent = game.last_ai_actions.join(" · ");
  } else log.hidden = true;

  if (game.done) {
    const result = game.winner === "human" ? "You win the pavilion!" : game.winner === "ai" ? `${game.opponent_name} wins.` : "The game ends in a tie.";
    $("status-line").textContent = result;
    $("status-detail").textContent = `Final score: ${game.players[0].score}–${game.players[1].score}`;
  } else {
    $("status-line").textContent = game.current_player === 0 ? "Your move" : "AI is thinking";
    $("status-detail").textContent = `You are playing ${game.opponent_name}. ${game.phase === "draft" ? "Draft from the workshops." : "Place tiles on your pavilion."}`;
  }
  renderActions(game);
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Something went wrong");
  return data;
}

function setBusy(busy) {
  state.busy = busy; $("thinking").hidden = !busy;
  $("new-game").disabled = busy;
  document.querySelectorAll(".action-button").forEach(button => button.disabled = busy);
}

function showError(error) {
  const toast = $("toast"); toast.textContent = error.message; toast.hidden = false;
  clearTimeout(showError.timer); showError.timer = setTimeout(() => toast.hidden = true, 5000);
}

async function newGame() {
  if (state.busy) return;
  setBusy(true);
  try {
    const game = await request("/api/new", { method: "POST", body: JSON.stringify({ opponent: $("opponent-select").value }) });
    render(game);
  } catch (error) { showError(error); }
  finally { setBusy(false); }
}

async function playAction(action) {
  if (state.busy) return;
  setBusy(true);
  try {
    render(await request("/api/action", { method: "POST", body: JSON.stringify({ action }) }));
  } catch (error) { showError(error); }
  finally { setBusy(false); }
}

async function init() {
  try {
    const game = await request("/api/state");
    const select = $("opponent-select");
    for (const opponent of game.opponents) {
      const option = document.createElement("option");
      option.value = opponent.id; option.textContent = opponent.name;
      option.disabled = !opponent.available; option.title = opponent.description;
      select.append(option);
    }
    select.value = game.opponent_id || "competitive_hybrid";
    $("new-game").addEventListener("click", newGame);
    if (game.started) render(game); else await newGame();
  } catch (error) { showError(error); }
}

init();
