from types import SimpleNamespace

from harness.layers.budget_policy import BudgetPolicy


def test_model_token_cost_triggers_budget_policy():
    ctx = SimpleNamespace(
        max_tool_calls=None,
        tools=SimpleNamespace(calls=0),
        budget={"max_tokens": 100},
        state={},
    )
    policy = BudgetPolicy()
    policy.before_agent(ctx)
    response = SimpleNamespace(prompt_tokens=40, completion_tokens=11)

    assert not policy._spent(ctx)
    returned = policy.wrap_model_call(ctx, lambda messages: response, [{"role": "user"}])

    assert returned is response
    assert ctx.state["budget_tokens_used"] == 51
    assert ctx.state["budget_last_turn_cost"] == 51
    assert policy._spent(ctx)
