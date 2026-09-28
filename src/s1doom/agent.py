"""Episode loop: perception -> state -> questions -> backend -> policy -> controller.

Two timing modes:
  turn      Mode.PLAYER: the game waits while the model thinks (deterministic evals).
  realtime  Mode.ASYNC_PLAYER @35 Hz: decisions run on a worker thread; the controller
            keeps executing the committed macro; answers older than max_staleness_ms drop.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from s1doom.backends.base import BackendError, DecisionBackend, DecisionResult
from s1doom.control.controller import Controller
from s1doom.decision.calibration import Calibrator
from s1doom.decision.policy import Policy, PolicyConfig
from s1doom.decision.questions import PromptSet, build_questions, legal_modes
from s1doom.decision.scheduler import QuestionScheduler
from s1doom.decision.state_builder import StateContext, build_state
from s1doom.decision.tokens import tokenizer_name
from s1doom.env.game import GameEnv, ScenarioConfig
from s1doom.mapmodel.runtime import LevelModel
from s1doom.perception import classes as C
from s1doom.perception.memory import Perception
from s1doom.telemetry.jsonl import JsonlLog
from s1doom.telemetry.metrics import DecisionStats, EpisodeMetrics

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
TRIGGER_EVENTS = {"new_monster", "damage", "health_threshold", "kill", "key_seen",
                  "target_lost", "macro_done", "stuck", "teleported"}
MIN_TRIGGER_GAP = 3


@dataclass
class RunConfig:
    scenario: str
    backend: str = "heuristic"
    mode: str = "turn"                 # turn | realtime
    episodes: int = 1
    seed: int = 1234
    omniscient: bool = False
    include_rules: bool = True
    decision_tics: int | None = None
    overlay: bool = False
    visible: bool = False
    save_frames_every: int = 0
    prompt_version: str = "v1"
    calibration: str | None = None
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    max_staleness_ms: float = 400.0
    max_calls: int | None = None        # budget cap per run
    run_id: str | None = None
    url: str | None = None
    skill: int | None = None
    record: bool = True
    replay_path: str | None = None
    max_tics: int | None = None
    opportunistic_fire: bool = True


class CircuitBreaker:
    def __init__(self, threshold: int = 5, cooldown_s: float = 30.0) -> None:
        self.threshold, self.cooldown_s = threshold, cooldown_s
        self.fails = 0
        self.open_until = 0.0

    def allow(self) -> bool:
        return time.monotonic() >= self.open_until

    def ok(self) -> None:
        self.fails = 0

    def fail(self) -> None:
        self.fails += 1
        if self.fails >= self.threshold:
            self.open_until = time.monotonic() + self.cooldown_s
            self.fails = 0


class _Worker:
    """Background asyncio loop for realtime mode."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.t = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.t.start()

    def submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)


class Agent:
    def __init__(self, cfg: RunConfig, backend: DecisionBackend) -> None:
        self.cfg = cfg
        self.backend = backend
        self.sc = ScenarioConfig.load(cfg.scenario)
        if cfg.skill is not None:
            self.sc.skill = cfg.skill
        self.decision_tics = cfg.decision_tics or self.sc.decision_tics
        self.run_id = cfg.run_id or time.strftime("%Y-%m-%d_%H%M%S") + f"_{self.sc.name}_{backend.name}_{uuid.uuid4().hex[:4]}"
        self.run_dir = RUNS_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.env = GameEnv(self.sc, realtime=cfg.mode == "realtime", visible=cfg.visible,
                           want_screen=cfg.overlay or cfg.save_frames_every > 0, seed=cfg.seed)
        self.level: LevelModel | None = None
        if self.sc.mapmodel and self.env.wad_path:
            self.level = LevelModel(self.env.wad_path, self.sc.map)
        self.perception = Perception(omniscient=cfg.omniscient)
        self.prompts = PromptSet.load(cfg.prompt_version)
        self.scheduler = QuestionScheduler()
        if backend.name == "random":
            cfg.policy.fallback_enabled = False  # sanity floor must not borrow heuristic answers
        self.policy = Policy(backend.name, cfg.policy, Calibrator.load(cfg.calibration))
        self.controller = Controller(self.sc.button_names, self.level, seed=cfg.seed,
                                     opportunistic_fire=cfg.opportunistic_fire)
        self.breaker = CircuitBreaker()
        self.calls_made = 0
        self.overlay = None
        if cfg.overlay or cfg.save_frames_every:
            from s1doom.telemetry.overlay import Overlay

            self.overlay = Overlay(show=cfg.overlay, save_dir=self.run_dir / "frames" if cfg.save_frames_every else None,
                                   save_every=cfg.save_frames_every, level=self.level)
        (self.run_dir / "config.json").write_text(json.dumps({
            "scenario": self.sc.model_dump(), "backend": backend.name, "mode": cfg.mode, "seed": cfg.seed,
            "fairness": "omniscient" if cfg.omniscient else "visible_only", "include_rules": cfg.include_rules,
            "decision_tics": self.decision_tics, "opportunistic_fire": cfg.opportunistic_fire,
            "prompt_version": self.prompts.version,
            "policy": cfg.policy.__dict__, "calibration": cfg.calibration, "tokenizer": tokenizer_name(),
        }, indent=2, default=str))

    # ------------------------------------------------------------------
    async def _call_backend(self, state: dict, questions: dict) -> tuple[DecisionResult | None, str | None]:
        if not questions:
            return None, None
        if not self.breaker.allow():
            return None, "circuit_open"
        if self.cfg.max_calls is not None and self.calls_made >= self.cfg.max_calls:
            return None, "budget_exhausted"
        try:
            self.calls_made += 1
            res = await self.backend.system_one(state, questions)
            self.breaker.ok()
            return res, None
        except (BackendError, ValueError) as e:
            self.breaker.fail()
            return None, str(e)[:300]

    def _prepare(self, p, ctx: StateContext):
        bs = build_state(p, ctx)
        modes = legal_modes(bs, self.sc.modes, self.level is not None)
        which = self.scheduler.pick(bs, self.controller.can_move)
        questions, forced = build_questions(self.prompts, bs, which, modes)
        return bs, modes, which, questions, forced

    def _context(self, recent: list[str], exit_len: float) -> StateContext:
        d = self.policy.current
        return StateContext(objective=self.sc.objective, level=self.level, mode=d.mode if d else None,
                            recent=recent, rules=self.prompts.rules, include_rules=self.cfg.include_rules,
                            exit_path_len=exit_len,
                            exclude={i for i, t in self.controller.blacklist.items() if t > self.env.tic})

    # ------------------------------------------------------------------
    async def run_episode(self, ep: int) -> EpisodeMetrics:
        cfg = self.cfg
        seed = cfg.seed + ep
        rec = str(self.run_dir / f"ep_{ep}.lmp") if cfg.record else None
        obs = self.env.reset(rec, seed=seed)
        self.perception.reset()
        self.policy.reset()
        self.scheduler.reset()
        if self.level is not None:
            self.level.reset()
        self.controller.reset(obs.v("ANGLE"))
        log = JsonlLog(self.run_dir / f"ep_{ep}" / "calls.jsonl")
        m = EpisodeMetrics(self.sc.name, self.backend.name, seed, ep)
        stats = DecisionStats()
        recent: list[str] = []
        next_decision = 0
        last_decision = -10 ** 9
        pending_row: dict | None = None
        dmg_since = 0.0
        prev_items = 0.0
        exit_len = float("inf")
        worker = _Worker() if cfg.mode == "realtime" else None
        inflight = None  # (future, submit_time, bs, modes, which, questions, forced, tic, triggered)
        ctrl_events: set[str] = set()
        mode_counts: dict[str, int] = {}
        teleported = False
        last_xy = None
        t_next_tic = time.perf_counter()

        def finish_decision(tic, bs, modes, which, questions, forced, result, err, triggered, staleness=None):
            nonlocal pending_row, dmg_since, last_decision
            d, calibrated, heur = self.policy.decide(tic, bs.state, questions, forced, result, triggered, modes)
            self.controller.commit(d, bs.refs, bs.weapons)
            last_decision = tic
            m.decisions += 1
            m.state_tokens_max = max(m.state_tokens_max, bs.tokens)
            if result is not None:
                m.model_calls += 1
                m.latency_ms.append(result.latency_ms)
                m.input_tokens += result.usage.get("input_tokens", 0)
                if self.overlay:
                    self.overlay.add_latency(result.latency_ms)
            if err:
                m.backend_errors += 1
            stats.add(d.sources, set(questions), result.answers if result else {}, heur, d.mode)
            mode_counts[d.mode] = mode_counts.get(d.mode, 0) + 1
            if pending_row is not None:
                pending_row["outcome"] = {"in_danger": dmg_since > 0}
                log.write(pending_row)
            dmg_since = 0.0
            pending_row = {
                "kind": "decision", "tic": tic, "episode": ep, "backend": self.backend.name,
                "pos": [round(self._p.x), round(self._p.y), round(self._p.angle)],
                "prompt_version": self.prompts.version,
                "fairness": "omniscient" if cfg.omniscient else "visible_only",
                "triggered": triggered, "state_tokens": bs.tokens, "state": bs.state, "questions": questions,
                "forced": forced,
                "raw_answers": {k: a.model_dump() for k, a in (result.answers.items() if result else [])},
                "calibrated_probs": {k: a.probabilities for k, a in calibrated.items()},
                "heuristic_answers": {k: a.model_dump() for k, a in heur.items()},
                "decision": {"mode": d.mode, "target": d.target, "pickup": d.pickup, "in_danger": d.in_danger,
                             "aggression": d.aggression, "weapon": d.weapon},
                "source": d.sources, "latency_ms": result.latency_ms if result else None,
                "usage": result.usage if result else None, "error": err, "staleness_ms": staleness,
            }

        try:
            while not self.env.finished():
                if cfg.max_tics and self.env.tic >= cfg.max_tics:
                    break
                t_tic = time.perf_counter()
                obs = self.env.observe()
                if obs is None:
                    break
                if self.level is not None:
                    self.level.update(obs.v("POSITION_X"), obs.v("POSITION_Y"), obs.sectors, obs.tic)
                p = self.perception.update(obs)
                self._p = p
                xy = (p.x, p.y)
                if last_xy is not None and abs(xy[0] - last_xy[0]) + abs(xy[1] - last_xy[1]) > 200:
                    p.events.add("teleported")
                    recent.append("teleported")
                last_xy = xy
                dmg_since += p.damage_taken
                if p.damage_taken > 0:
                    recent.append(f"took {int(p.damage_taken)} damage" + (" from unseen attacker" if p.unseen_attack else ""))
                if "kill" in p.events:
                    recent.append("killed a monster")
                items = p.vars.get("ITEMCOUNT", 0)
                if items > prev_items:
                    recent.append("picked up an item")
                prev_items = items
                recent = recent[-6:]
                events = (p.events | ctrl_events) & TRIGGER_EVENTS
                if "stuck" in ctrl_events:
                    m.stuck_events += 1
                triggered = bool(events) and self.env.tic - last_decision >= MIN_TRIGGER_GAP
                due = self.env.tic >= next_decision or triggered

                if self.level is not None and self.level.exit_known and (self.env.tic % 35 == 0 or due):
                    g = self.level.exit_goal()
                    exit_len = self.level.path_length(self.level.plan(p.x, p.y, g[0], g[1])) if g else float("inf")

                if worker is None:
                    if due:
                        ctx = self._context(recent, exit_len)
                        bs, modes, which, questions, forced = self._prepare(p, ctx)
                        result, err = await self._call_backend(bs.state, questions)
                        finish_decision(self.env.tic, bs, modes, which, questions, forced, result, err, triggered)
                        next_decision = self.env.tic + self.decision_tics
                else:
                    if inflight is not None and inflight[0].done():
                        fut, t_sub, bs, modes, which, questions, forced, tic0, trig0 = inflight
                        inflight = None
                        stale = (time.perf_counter() - t_sub) * 1000
                        m.staleness_ms.append(stale)
                        try:
                            result, err = fut.result()
                        except Exception as e:  # noqa: BLE001
                            result, err = None, repr(e)
                        if stale > cfg.max_staleness_ms:
                            m.dropped_stale += 1
                        else:
                            finish_decision(self.env.tic, bs, modes, which, questions, forced, result, err, trig0, stale)
                    if inflight is None and (due or self.policy.current is None):
                        ctx = self._context(recent, exit_len)
                        bs, modes, which, questions, forced = self._prepare(p, ctx)
                        if self.policy.current is None or not questions:
                            # first decision / nothing to ask: decide synchronously from forced/heuristic
                            finish_decision(self.env.tic, bs, modes, which, questions, forced, None, None, triggered)
                        else:
                            fut = worker.submit(self._call_backend(bs.state, questions))
                            inflight = (fut, time.perf_counter(), bs, modes, which, questions, forced,
                                        self.env.tic, triggered)
                        next_decision = self.env.tic + self.decision_tics

                act = self.controller.act(p)
                if worker is not None:
                    m.tic_work_ms.append((time.perf_counter() - t_tic) * 1000)
                ctrl_events = set(self.controller.events)
                if self.overlay is not None and obs.screen is not None:
                    path = self.controller.nav.path[self.controller.nav.idx:] if (self.controller.nav and self.controller.nav.path) else None
                    img = self.overlay.render(obs.screen, p, self.policy.current, self.backend.name,
                                              self.controller.target_id, path)
                    if not self.overlay.emit(img, self.env.tic):
                        break
                if worker is not None:
                    # ViZDoom's make_action blocks until the next tic while holding the GIL,
                    # starving the decision thread. Sleep (GIL released) until the tic is due.
                    now = time.perf_counter()
                    if t_next_tic > now:
                        time.sleep(max(0.0, t_next_tic - now - 0.002))
                self.env.step(act, 1)
                t_next_tic = max(time.perf_counter(), t_next_tic) + 1 / 35
        finally:
            if worker is not None:
                worker.stop()
            if pending_row is not None:
                pending_row["outcome"] = {"in_danger": dmg_since > 0}
                log.write(pending_row)

        v = self.env.last_vars()
        m.kills = int(v["KILLCOUNT"])
        m.damage_taken = float(v["DAMAGE_TAKEN"])
        m.hits = int(v["HITCOUNT"])
        m.survival_tics = int(max(self.env.episode_time(), self.env.tic))
        m.health_end = float(v["HEALTH"])
        m.armor_end = float(v["ARMOR"])
        m.items = int(v["ITEMCOUNT"])
        m.secrets = int(v["SECRETCOUNT"])
        m.dead = bool(self.env.game.is_player_dead())
        timeout = self.env.game.get_episode_timeout()
        m.timeout = bool(timeout and m.survival_tics >= timeout - 1) or bool(cfg.max_tics and self.env.tic >= cfg.max_tics)
        finished = self.env.finished()
        if self.sc.success == "survive":
            m.completed = not m.dead
        else:
            m.completed = finished and not m.dead and not m.timeout
        if m.completed and self.sc.success == "exit":
            m.time_to_exit_tics = m.survival_tics
        m.fallback_rate = stats.fallback_rate()
        m.agreement = stats.agreement()
        m.mode_switches_per_min = stats.switches_per_min(max(m.survival_tics, 1))
        m.mode_counts = mode_counts
        if self.level is not None:
            m.explored = round(self.level.explore.seen_fraction(), 3)
        log.write({"kind": "episode_end", **m.to_dict()})
        log.close()
        return m

    async def run(self, progress=None) -> list[EpisodeMetrics]:
        out = []
        for ep in range(self.cfg.episodes):
            m = await self.run_episode(ep)
            out.append(m)
            if progress:
                progress(m)
        from s1doom.telemetry.summary import write_summary

        write_summary(self.run_dir, out)
        if hasattr(self.backend, "aclose"):
            await self.backend.aclose()
        return out

    def close(self) -> None:
        self.env.close()
        if self.overlay:
            self.overlay.close()


def weapon_name(slot: int) -> str:
    return C.WEAPON_SLOTS.get(slot, "?")
