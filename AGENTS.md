# AGENTS.md

## Purpose

This repository is a rules-faithful Azul: Summer Pavilion simulator, local human-vs-AI web app, and PyTorch self-play research project. Prefer targeted inspection and tests over rereading the whole repository.

## Start here

- `README.md` documents supported rules, current model claims, commands, and the project map.
- `src/azul/game.py` is the authoritative deterministic rules engine and action/observation encoding.
- `src/azul/agents.py` contains random, heuristic, neural, hybrid, and rollout agents.
- `src/azul/web.py` owns the local HTTP API, game sessions, opponent loading, and action logging.
- `frontend/main.jsx` and `frontend/styles.css` are the editable React UI sources.
- `src/azul/web_static/` is the built UI served by Python. Do not hand-edit generated `app.js` or `styles.css`; run `npm run build` after frontend changes.
- `tests/test_game.py`, `tests/test_web.py`, and `tests/test_training.py` are the fastest executable specifications for their areas.
- `reference/README.md` explains the supplied rules/board references. Do not commit copyrighted rulebook or artwork files.

## Environment and commands

Python 3.10+ is required. The package uses a `src/` layout.

```sh
python -m pip install -e '.[dev]'
pytest
python -m azul.web              # http://127.0.0.1:8000
npm install
npm run build                   # frontend/ -> src/azul/web_static/
```

Use focused checks while iterating:

```sh
pytest tests/test_game.py
pytest tests/test_web.py
pytest tests/test_training.py
pytest tests/test_game.py -k connected
```

For a frontend change, run `npm run build` and `pytest tests/test_web.py`. For engine changes, run the relevant focused tests and then full `pytest` when practical.

## Important invariants

- The rules engine supports 2–4 players and official factory counts; neural observations, training, evaluation, CLI play, and web play are intentionally two-player.
- Only the normal colored-star board is implemented.
- Legal moves must come from `AzulGame.legal_actions()`; keep action IDs compatible with `encode_action`/`decode_action` and the legal-action mask.
- Preserve deterministic seeded games, tile conservation, round termination, and the 100-action AI-loop guard.
- Placement payment choices are distinct actions: at least one natural tile is normally required, except the round's wild color on its matching star.
- Training's exact-score objective is intentional: per-player episode reward sums to `(final score - initial score) / 20` with `gamma=1` and `gae_lambda=1`.
- Checkpoint paths are resolved from the repository root. Missing optional checkpoints should make opponents unavailable, not break the opponent catalog.
- Web games log transitions under `log/`; logs and model `.pt` files are ignored. Keep log records replay/training-friendly even though replay is not implemented.

## Web architecture and debugging

The Python server serves `src/azul/web_static/` and exposes JSON endpoints from `src/azul/web.py`. The browser UI uses those endpoints; there is no separate production Node server.

If the page remains on “Preparing the pavilion”:

1. Confirm `python -m azul.web` starts without a checkpoint/import error.
2. Check that `src/azul/web_static/index.html`, `app.js`, and `styles.css` exist.
3. Rebuild with `npm run build` after React changes.
4. Inspect the first failed browser request and server traceback; avoid speculative engine changes.
5. Use the `random` or `heuristic` opponent to separate UI/API problems from checkpoint loading.

UI action animations must be cosmetic, must not delay or duplicate actions, and must respect the UI animation toggle and reduced-motion preferences.

## Change discipline

- The worktree may contain user changes from concurrent UI, logging, or engine work. Check `git status` first and do not revert unrelated modifications.
- Keep rules logic in `game.py`, presentation/API serialization in `web.py`, persistence in `game_log.py`, and UI behavior in `frontend/`.
- Add a regression test with bug fixes. Prefer seeded tests and existing public methods over brittle private-state setup.
- Do not casually regenerate or edit `*.egg-info`, checkpoints, metrics, frontier JSON, or training reports.
- Do not retrain models for ordinary code/UI fixes. Training is expensive and hardware-dependent; verify simulator behavior independently first.
- If changing reported scores, model rankings, or strategy claims, require reproducible evaluation evidence and update the relevant report/README together.

## Search shortcuts

Use `rg` before broad file reads:

```sh
rg -n "OPPONENTS|GameSession|do_GET|do_POST" src/azul/web.py
rg -n "legal_actions|apply_action|encode_action|decode_action" src/azul/game.py
rg -n "fetch\(|animation|reduced-motion" frontend
rg -n "def test_" tests
```
