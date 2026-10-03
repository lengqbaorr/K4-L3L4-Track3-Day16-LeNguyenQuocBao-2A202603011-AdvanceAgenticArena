import json

from arena.model import ModelResponse
from arena.tools import Tools
from arena.trace import Trace
from harness.agent import ReActAgent
from tests.fixtures_briefs import BRIEF_SLA, CORPUS


class RealLikeScript:
    def __init__(self, *turns):
        self.turns = list(turns)
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append(messages)
        text = self.turns[min(len(self.calls) - 1, len(self.turns) - 1)]
        return ModelResponse(text=text, prompt_tokens=10, completion_tokens=5)


def test_truncated_observed_quote_gets_one_full_line_reask():
    doc = next(doc for doc in CORPUS.docs if any(len(line) > 50 for line in doc.body.splitlines()))
    full_line = next(line for line in doc.body.splitlines() if len(line) > 50)
    partial = full_line[:30]
    partial_final = json.dumps(
        {"answer": "answer", "abstain": False, "citations": [doc.doc_id],
         "claims": [{"text": partial, "doc_id": doc.doc_id}]},
        ensure_ascii=False,
    )
    full_final = json.dumps(
        {"answer": "answer", "abstain": False, "citations": [doc.doc_id],
         "claims": [{"text": full_line, "doc_id": doc.doc_id}]},
        ensure_ascii=False,
    )
    model = RealLikeScript(
        f'THOUGHT: read\nACTION: {{"tool":"fetch_doc","args":{{"doc_id":"{doc.doc_id}"}}}}',
        f"THOUGHT: done\nFINAL: {partial_final}",
        f"THOUGHT: done\nFINAL: {full_final}",
    )
    trace = Trace(run_id="quote-completion", seed=101)
    tools = Tools(CORPUS, trace, seed=101, flaky=False)
    agent = ReActAgent(model, tools, trace, corpus=CORPUS)

    report = agent.run(BRIEF_SLA)

    assert report["claims"][0]["text"] == full_line
    assert len(model.calls) == 3
    assert full_line in model.calls[2][-1]["content"]
