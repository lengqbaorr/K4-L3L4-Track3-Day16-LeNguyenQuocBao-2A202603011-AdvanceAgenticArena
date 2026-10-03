import json

from arena.model import ModelResponse
from arena.tools import Tools
from arena.trace import Trace
from harness.agent import ReActAgent
from tests.fixtures_briefs import BRIEF_SLA, CORPUS


class ScriptedModel:
    def __init__(self, *turns):
        self.turns = list(turns)
        self.index = 0

    def complete(self, messages, **kwargs):
        text = self.turns[min(self.index, len(self.turns) - 1)]
        self.index += 1
        return ModelResponse(text=text, prompt_tokens=1, completion_tokens=1)


def test_flat_action_fields_are_dispatched_as_tool_args():
    trace = Trace(run_id="flat-action", seed=12)
    tools = Tools(CORPUS, trace, seed=12, flaky=False)
    agent = ReActAgent(
        ScriptedModel(
            'THOUGHT: tìm\nACTION: {"k": 5, "query": "SLA", "tool": "search"}',
            'THOUGHT: xong\nFINAL: {"answer":"x","claims":[],"citations":[],"abstain":true}',
        ),
        tools,
        trace,
        corpus=CORPUS,
    )

    agent.run(BRIEF_SLA)

    searches = [
        json.loads(line)
        for line in trace.to_jsonl().splitlines()
        if json.loads(line).get("event") == "tool_call"
        and json.loads(line).get("name") == "search"
    ]
    assert len(searches) == 1
    assert searches[0]["query"] == "SLA"
    assert searches[0]["n_results"] > 0


def test_empty_search_query_is_rejected_without_tool_call():
    trace = Trace(run_id="empty-search", seed=13)
    tools = Tools(CORPUS, trace, seed=13, flaky=False)
    agent = ReActAgent(
        ScriptedModel(
            'THOUGHT: tìm\nACTION: {"tool": "search"}',
            'THOUGHT: xong\nFINAL: {"answer":"x","claims":[],"citations":[],"abstain":true}',
        ),
        tools,
        trace,
        corpus=CORPUS,
    )

    agent.run(BRIEF_SLA)

    events = [json.loads(line) for line in trace.to_jsonl().splitlines()]
    assert not any(
        event.get("event") == "tool_call" and event.get("name") == "search"
        for event in events
    )
    assert tools.calls == 1  # submit only
