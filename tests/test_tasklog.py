"""Task log: what the robot is doing, shown live and read by the agents."""
import asyncio
from types import SimpleNamespace

from cobot.agents.vision_agent import sim
from cobot.tasklog import TaskLog, log_event


def test_begin_add_end_and_snapshot():
    log = TaskLog()
    seen = []
    log.listeners.append(seen.append)
    log.begin("sort the parts")
    assert log.running
    log.add("result", "part 2: FAIL, crack", part="part 2", result="fail")
    snap = log.snapshot()
    assert snap["running_task"] == "sort the parts"
    assert snap["events"][-1]["data"]["result"] == "fail"
    log.end("finished")
    assert not log.running and log.last_task == "sort the parts"
    assert [e.kind for e in seen] == ["task", "result", "task"]


def test_listener_errors_do_not_break_the_task():
    log = TaskLog()
    log.listeners.append(lambda ev: 1 / 0)
    log.add("step", "picked part 1")          # must not raise
    assert len(log.events) == 1


def test_log_event_without_a_log_is_a_no_op():
    log_event(SimpleNamespace(), "step", "nothing to log to")


def test_inspection_is_logged_with_result_and_defect():
    class Scene:
        async def call_tool(self, tool, args):
            return {"ok": True, "name": "part 2", "inspectable": True, "result": "fail",
                    "defect": "crack across the top face"}
    log = TaskLog()
    deps = SimpleNamespace(sim_scene=Scene(), task_log=log)
    asyncio.run(sim.sim_inspect_object(deps, "part 2"))
    ev = log.events[-1]
    assert ev.kind == "result" and ev.data == {"part": "part 2", "result": "fail",
                                               "defect": "crack across the top face"}
    assert "crack" in ev.text
