"""WAD parse + nav graph on Freedoom2 MAP01 (bundled with ViZDoom)."""
import pytest

from s1doom.env.wads import resolve_wad
from s1doom.mapmodel import specials as SP
from s1doom.mapmodel.runtime import LevelModel


@pytest.fixture(scope="module")
def lv():
    return LevelModel(resolve_wad("freedoom2"), "MAP01")


def test_specials():
    assert SP.classify(11) == "exit" and SP.classify(52) == "exit"
    assert SP.classify(124) == "secret_exit"
    assert SP.classify(1) == "door" and SP.door_key(26) == "blue"
    assert SP.classify(103) == "remote_door" and SP.classify(62) == "lift"
    assert SP.classify(97) == "teleport"
    assert SP.is_use(11) and not SP.is_use(52)


def test_map01_parsed(lv):
    assert len(lv.m.sectors) == 211 and len(lv.exits) == 1
    assert lv.m.player_start() is not None


def test_start_to_exit_path_exists(lv):
    st = lv.m.player_start()
    gx, gy, _ = lv.exit_goal()
    path = lv.plan(st.x, st.y, gx, gy)
    assert path is not None and 2000 < lv.path_length(path) < 8000


def test_edges_are_directed_for_drops(lv):
    """Some ledge must be droppable but not climbable."""
    nav = lv.nav
    one_way = 0
    for u in range(nav.n):
        for e in nav.adj[u]:
            if e.teleport < 0 and nav.edge(e.to, u) is None:
                one_way += 1
    assert one_way > 0


def test_thin_crack_is_walkable(lv):
    """8-unit sector with floor -480 between 143 and 144 (exit room): Doom's 32-unit
    player box spans it, so the graph must connect across."""
    a, b = lv.node(167, -728), lv.node(167, -680)
    assert lv.nav.astar(a, b) is not None


def test_nodes_not_on_linedefs(lv):
    import numpy as np
    assert not np.any(np.isclose(lv.nav.xs % 8, 0))


def test_remote_door_trigger_known(lv):
    # line 839 (S1 open door, tag 4) opens sector 208 in front of the exit switch
    assert 839 in lv.triggers.get(208, [])
