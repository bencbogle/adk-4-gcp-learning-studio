import json
from pathlib import Path


def test_retrieval_eval_set_has_unique_grounded_cases() -> None:
    """Verify retrieval cases are unique and use official documentation URLs."""
    cases_path = Path(__file__).parents[1] / "evals" / "retrieval_cases.json"
    cases = json.loads(cases_path.read_text())

    assert len(cases) == 5
    assert len({case["id"] for case in cases}) == len(cases)

    for case in cases:
        assert case["query"]
        assert case["product"]
        assert case["expected_source_urls"]
        assert all(url.startswith("https://docs.cloud.google.com/") for url in case["expected_source_urls"])
