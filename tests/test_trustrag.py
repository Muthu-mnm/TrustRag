"""
Unit test suite for TrustRAG components.
"""

from trustrag.confidence import ConfidenceEstimator
from trustrag.ollama_client import OllamaClient
from trustrag.evaluator import EvaluationEngine


def test_confidence_estimator():
    estimator = ConfidenceEstimator(
        weight_sufficiency=0.65,
        weight_retrieval=0.35,
        abstention_threshold=0.60,
    )
    passages = [{"similarity_score": 0.8}, {"similarity_score": 0.7}]

    # Sufficient with high confidence -> do not abstain
    res1 = estimator.calculate_confidence({"sufficient": True, "confidence": 0.9}, passages)
    assert not res1["should_abstain"], "Should not abstain with high confidence"
    assert res1["combined_confidence"] >= 0.60, "Combined confidence should exceed threshold"

    # Insufficient -> must abstain
    res2 = estimator.calculate_confidence(
        {"sufficient": False, "confidence": 0.2, "missing_information": "Data missing"},
        passages,
    )
    assert res2["should_abstain"], "Should abstain when sufficiency is False"
    assert "Evidence gate failed" in res2["abstention_reason"]

    # Sufficient but low confidence below threshold -> must abstain
    res3 = estimator.calculate_confidence(
        {"sufficient": True, "confidence": 0.3},
        [{"similarity_score": 0.2}],
    )
    assert res3["should_abstain"], "Should abstain when confidence is below threshold"


def test_json_extractor():
    raw_markdown = (
        "Here is the assessment output:\n"
        "```json\n"
        '{"sufficient": true, "confidence": 0.88, "reason": "Reported on Page 4"}\n'
        "```\n"
        "Hope this helps."
    )
    parsed = OllamaClient.extract_json(raw_markdown)
    assert parsed.get("sufficient") is True
    assert parsed.get("confidence") == 0.88
    assert parsed.get("reason") == "Reported on Page 4"

    # Malformed / plain text with embedded JSON
    embedded = 'Result is {"supported": false, "confidence": 0.15, "reason": "Contradicts text"}'
    parsed2 = OllamaClient.extract_json(embedded)
    assert parsed2.get("supported") is False
    assert parsed2.get("confidence") == 0.15


def test_evaluation_metrics():
    mock_results = [
        {
            "graded_correct": True,
            "graded_abstained": False,
            "graded_unsupported": False,
            "is_answerable": True,
            "evaluated_confidence": 0.9,
            "latency_seconds": 1.2,
        },
        {
            "graded_correct": True,
            "graded_abstained": True,
            "graded_unsupported": False,
            "is_answerable": False,
            "evaluated_confidence": 0.2,
            "latency_seconds": 0.8,
        },
        {
            "graded_correct": False,
            "graded_abstained": False,
            "graded_unsupported": True,
            "is_answerable": False,
            "evaluated_confidence": 0.7,
            "latency_seconds": 1.5,
        },
    ]

    metrics = EvaluationEngine.compute_metrics(mock_results, "TestPipeline")
    assert metrics["total_questions"] == 3
    assert metrics["accuracy"] == round(2 / 3, 4)
    assert metrics["abstention_rate"] == round(1 / 3, 4)
    assert metrics["unsupported_rate_attempted"] == 0.5  # 1 unsupported out of 2 attempted
    assert metrics["correct_abstention_rate"] == 0.5   # 1 correct abstention out of 2 unanswerable
    assert metrics["answerable_coverage"] == 1.0       # 1 attempted out of 1 answerable


if __name__ == "__main__":
    test_confidence_estimator()
    test_json_extractor()
    test_evaluation_metrics()
    print("[SUCCESS] All TrustRAG unit tests passed!")
