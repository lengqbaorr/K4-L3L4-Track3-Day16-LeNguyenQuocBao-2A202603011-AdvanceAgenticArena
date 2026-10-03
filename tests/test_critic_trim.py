from types import SimpleNamespace

from harness.layers.critic import Critic


def make_corpus(entries):
    docs = [SimpleNamespace(doc_id=doc_id, body=body) for doc_id, body in entries]
    return SimpleNamespace(
        docs=docs,
        get=lambda doc_id: next((doc for doc in docs if doc.doc_id == doc_id), None),
    )


def run_critic(observed, claims, corpus=None):
    ctx = SimpleNamespace(observed_text=observed, corpus=corpus)
    report = {"claims": claims, "citations": []}
    return Critic().after_agent(ctx, report)


def test_trailing_period_is_trimmed_and_kept():
    sentence = "The observed sentence contains enough useful detail"
    corpus = make_corpus([("doc-1", sentence)])
    report = run_critic(sentence, [{"text": sentence + ".", "doc_id": "doc-1"}], corpus)
    assert report["claims"][0]["text"] == sentence


def test_bold_wrapper_is_trimmed_and_kept():
    sentence = "A bold wrapped sentence remains verbatim inside"
    corpus = make_corpus([("doc-1", sentence)])
    report = run_critic(sentence, [{"text": f"**{sentence}**", "doc_id": "doc-1"}], corpus)
    assert report["claims"][0]["text"] == sentence


def test_paraphrase_is_dropped():
    report = run_critic(
        "The observed source says the shipment arrives tomorrow",
        [{"text": "The delivery is expected to arrive the following day", "doc_id": "doc-1"}],
    )
    assert report["claims"] == []
    assert report["abstain"] is True


def test_exact_duplicate_claims_are_removed():
    claim = {"text": "This exact observed quotation is long enough", "doc_id": "doc-1"}
    report = run_critic(claim["text"], [claim.copy(), claim.copy()])
    assert report["claims"] == [claim]


def test_trimmed_claim_is_reattributed_to_observed_source():
    sentence = "This quotation belongs to the observed source"
    corpus = make_corpus([("wrong-doc", "An unrelated line"), ("source-doc", sentence)])
    report = run_critic(
        sentence,
        [{"text": f"**{sentence}**", "doc_id": "wrong-doc"}],
        corpus,
    )
    assert report["claims"] == [{"text": sentence, "doc_id": "source-doc"}]
