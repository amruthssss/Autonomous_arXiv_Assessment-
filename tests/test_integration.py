import os

import pytest


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY") or os.getenv("RUN_NETWORK_TESTS") != "1",
    reason="set GEMINI_API_KEY and RUN_NETWORK_TESTS=1",
)
def test_live_pipeline():
    from app.graph import build_graph

    result = build_graph().invoke({"query": "briefing: retrieval augmented generation", "max_results": 1})
    assert result.get("briefing") or not result.get("errors")
