"""ViZDoom setup: scenario configs, button mapping, sync/async modes, demo recording."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import vizdoom as vzd
import yaml
from pydantic import BaseModel

from s1doom.env.wads import resolve_wad

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"

# Logical action name -> ViZDoom button. Verified against ViZDoom 1.3.1:
# TURN_LEFT_RIGHT_DELTA > 0 turns RIGHT (ANGLE decreases), units are degrees per tic.
BUTTONS: dict[str, vzd.Button] = {
    "turn": vzd.Button.TURN_LEFT_RIGHT_DELTA,
    "forward": vzd.Button.MOVE_FORWARD_BACKWARD_DELTA,
    "strafe": vzd.Button.MOVE_LEFT_RIGHT_DELTA,
    "attack": vzd.Button.ATTACK,
    "use": vzd.Button.USE,
    "speed": vzd.Button.SPEED,
    **{f"weapon{i}": getattr(vzd.Button, f"SELECT_WEAPON{i}") for i in range(1, 8)},
}

GAME_VARS = [
    "HEALTH", "ARMOR", "SELECTED_WEAPON", "SELECTED_WEAPON_AMMO",
    *[f"AMMO{i}" for i in range(10)], *[f"WEAPON{i}" for i in range(10)],
    "KILLCOUNT", "ITEMCOUNT", "SECRETCOUNT", "POSITION_X", "POSITION_Y", "POSITION_Z",
    "ANGLE", "DAMAGECOUNT", "DAMAGE_TAKEN", "HITCOUNT", "DEAD", "ON_GROUND",
]


class ScenarioConfig(BaseModel):
    name: str
    cfg: str | None = None            # ViZDoom scenario .cfg (relative to vizdoom.scenarios_path)
    wad: str | None = None            # "freedoom2", "doom1", or a path; None = taken from cfg
    map: str = "MAP01"
    skill: int | None = None
    buttons: list[str] | str = "all"
    modes: list[str] = ["fight", "explore"]
    objective: str = "survive"        # survive | reach_goal | reach_exit | gather
    success: str = "survive"          # survive | not_dead_before_timeout | exit
    episode_timeout: int | None = None
    mapmodel: bool = False
    decision_tics: int = 8
    notes: str = ""

    @classmethod
    def load(cls, name_or_path: str) -> "ScenarioConfig":
        p = Path(name_or_path)
        if not p.exists():
            p = CONFIG_DIR / "scenarios" / f"{name_or_path}.yaml"
        return cls(**yaml.safe_load(p.read_text()))

    @property
    def button_names(self) -> list[str]:
        return list(BUTTONS) if self.buttons == "all" else list(self.buttons)


@dataclass
class Obs:
    """One tic of engine state, flattened for perception."""
    tic: int
    vars: dict[str, float]
    labels: list[Any]
    objects: list[Any]
    sectors: list[Any]
    screen: np.ndarray | None
    labels_buf: np.ndarray | None
    automap: np.ndarray | None
    screen_w: int = 320
    screen_h: int = 240

    def v(self, name: str, default: float = 0.0) -> float:
        return self.vars.get(name, default)


@dataclass
class GameEnv:
    scenario: ScenarioConfig
    realtime: bool = False
    visible: bool = False
    want_screen: bool = False
    seed: int | None = None
    game: vzd.DoomGame = field(init=False)
    wad_path: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        sc = self.scenario
        g = vzd.DoomGame()
        if sc.cfg:
            g.load_config(os.path.join(vzd.scenarios_path, sc.cfg))
        if sc.wad:
            self.wad_path = resolve_wad(sc.wad)
            g.set_doom_game_path(self.wad_path)
            g.set_doom_map(sc.map)
        elif sc.cfg:
            # Scenario WAD is loaded as scenario path by the cfg.
            self.wad_path = os.path.join(vzd.scenarios_path, sc.cfg.replace(".cfg", ".wad"))
        if sc.skill is not None:
            g.set_doom_skill(sc.skill)
        if sc.episode_timeout is not None:
            g.set_episode_timeout(sc.episode_timeout)
        g.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        g.set_screen_format(vzd.ScreenFormat.BGR24)
        g.set_labels_buffer_enabled(True)
        g.set_objects_info_enabled(True)
        g.set_sectors_info_enabled(True)
        g.set_automap_buffer_enabled(self.want_screen)
        g.set_automap_mode(vzd.AutomapMode.OBJECTS_WITH_SIZE)
        g.set_automap_render_textures(False)
        g.set_render_hud(True)
        g.set_window_visible(self.visible)
        g.set_available_buttons([BUTTONS[b] for b in sc.button_names])
        for b in ("turn", "forward", "strafe"):
            if b in sc.button_names:
                g.set_button_max_value(BUTTONS[b], 0)  # 0 = unlimited
        g.set_available_game_variables([getattr(vzd.GameVariable, v) for v in GAME_VARS])
        g.set_mode(vzd.Mode.ASYNC_PLAYER if self.realtime else vzd.Mode.PLAYER)
        if self.realtime:
            g.set_ticrate(35)
        if self.seed is not None:
            g.set_seed(self.seed)
        g.init()
        self.game = g
        self._button_index = {b: i for i, b in enumerate(sc.button_names)}
        self.tic = 0

    # -- episode control -------------------------------------------------
    def reset(self, record_path: str | None = None, seed: int | None = None) -> Obs:
        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode(record_path or "")
        self.tic = 0
        return self.observe()

    def finished(self) -> bool:
        return self.game.is_episode_finished()

    def observe(self) -> Obs | None:
        s = self.game.get_state()
        if s is None:
            return None
        gv = dict(zip(GAME_VARS, (float(x) for x in s.game_variables)))
        return Obs(
            tic=self.tic, vars=gv, labels=list(s.labels), objects=list(s.objects),
            sectors=list(s.sectors), screen=s.screen_buffer if self.want_screen else None,
            labels_buf=s.labels_buffer if self.want_screen else None,
            automap=s.automap_buffer if self.want_screen else None,
        )

    def step(self, action: dict[str, float], tics: int = 1) -> float:
        vec = [0.0] * len(self._button_index)
        for k, v in action.items():
            i = self._button_index.get(k)
            if i is not None:
                vec[i] = float(v)
        r = self.game.make_action(vec, tics)
        self.tic += tics
        return r

    def last_vars(self) -> dict[str, float]:
        return {v: self.game.get_game_variable(getattr(vzd.GameVariable, v)) for v in GAME_VARS}

    def episode_time(self) -> int:
        return self.game.get_episode_time()

    def close(self) -> None:
        self.game.close()

    def has(self, button: str) -> bool:
        return button in self._button_index
