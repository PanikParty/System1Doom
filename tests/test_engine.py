"""Engine smoke tests (boot ViZDoom headless)."""
import asyncio
import json

import pytest

pytestmark = pytest.mark.engine


def test_turn_delta_positive_turns_right():
    from s1doom.env.game import GameEnv, ScenarioConfig

    env = GameEnv(ScenarioConfig.load("defend_the_center"))
    try:
        env.reset()
        a0 = env.observe().v("ANGLE")
        env.step({"turn": 10}, 1)
        a1 = env.observe().v("ANGLE")
        assert (a0 - a1) % 360 == pytest.approx(10, abs=0.1)  # angle DEcreased -> right turn
    finally:
        env.close()


def test_random_episode_writes_jsonl_and_lmp(tmp_path):
    from s1doom.agent import Agent, RunConfig
    from s1doom.backends.random_ import RandomBackend

    agent = Agent(RunConfig(scenario="defend_the_center", backend="random", run_id=f"test_{tmp_path.name}"),
                  RandomBackend(1))
    try:
        eps = asyncio.run(agent.run())
    finally:
        agent.close()
    assert eps[0].survival_tics > 0 and eps[0].decisions > 0
    d = agent.run_dir  # demo is finalized on close()
    assert (d / "ep_0.lmp").exists() and (d / "summary.json").exists()
    rows = [json.loads(l) for l in (d / "ep_0" / "calls.jsonl").read_text().splitlines()]
    dec = [r for r in rows if r["kind"] == "decision"]
    assert dec and all({"tic", "state", "questions", "source", "fairness", "prompt_version"} <= set(r) for r in dec)
    assert all(r["fairness"] == "visible_only" for r in dec)
    assert rows[-1]["kind"] == "episode_end"
    import shutil
    shutil.rmtree(d)


def test_replay_reproduces_episode():
    from s1doom.agent import Agent, RunConfig
    from s1doom.backends.heuristic import HeuristicBackend
    from s1doom.backends.replay import ReplayBackend

    a = Agent(RunConfig(scenario="defend_the_center", run_id="test_replay_src", record=False), HeuristicBackend())
    try:
        e1 = asyncio.run(a.run())[0]
    finally:
        a.close()
    calls = a.run_dir / "ep_0" / "calls.jsonl"
    b = Agent(RunConfig(scenario="defend_the_center", run_id="test_replay_dst", record=False),
              ReplayBackend(str(calls)))
    try:
        e2 = asyncio.run(b.run())[0]
    finally:
        b.close()
    assert (e1.kills, e1.survival_tics) == (e2.kills, e2.survival_tics)
    import shutil
    shutil.rmtree(a.run_dir)
    shutil.rmtree(b.run_dir)
