from src.main.semantic_query.planner import plan_query


def test_planner_no_deterministic_time_extraction():
    q = "show events 09:15-09:45 with people"
    plan = plan_query(q, interpreter_constraints=None)
    # Without interpreter, no time windows extracted
    assert plan.time_windows == []
    # Primary text retains original including time expression
    assert "09:15-09:45" in plan.primary_text


def test_planner_fallback_primary():
    q = "09:00-10:00"
    plan = plan_query(q, None)
    # If only the time range present, primary fallback is original
    assert plan.primary_text == q
