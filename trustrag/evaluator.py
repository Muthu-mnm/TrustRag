"""
Evaluation framework for TrustRAG and Baseline RAG.
Computes Accuracy, Hallucination/Unsupported rate, Abstention rate, Coverage,
Abstention Precision, Expected Calibration Error (ECE), and Risk-Coverage curves.
"""

import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


class EvaluationEngine:
    """Runs batch evaluation of Baseline RAG vs TrustRAG on labelled benchmark CSVs."""

    def __init__(self, rag_pipeline):
        self.pipeline = rag_pipeline

    @staticmethod
    def parse_boolean(val: Any) -> bool:
        """Helper to parse boolean values from CSV columns."""
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        val_str = str(val).strip().lower()
        return val_str in ["true", "1", "yes", "y", "t"]

    def evaluate_dataset(
        self,
        df: pd.DataFrame,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Run both Baseline RAG and TrustRAG over the benchmark dataset.
        Expected CSV columns:
            question, expected_answer, answerable, question_type
        Optional columns:
            relevant_pages, relevant_evidence, notes
        """
        # Validate columns
        required_cols = ["question", "expected_answer", "answerable"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required CSV column: '{col}'")

        records = df.to_dict(orient="records")
        total_samples = len(records)

        baseline_results = []
        trustrag_results = []

        for idx, row in enumerate(records):
            q = str(row["question"]).strip()
            exp_ans = str(row["expected_answer"]).strip()
            is_answerable = self.parse_boolean(row["answerable"])
            q_type = str(row.get("question_type", "Answerable" if is_answerable else "Unanswerable")).strip()

            # 1. Run Baseline RAG
            base_out = self.pipeline.run_baseline_rag(q)
            base_eval = self._grade_response(
                response_text=base_out["answer"],
                abstained=base_out["abstained"],
                expected_answer=exp_ans,
                is_answerable=is_answerable,
                question_type=q_type,
                confidence=0.85,  # Baseline has uncalibrated static confidence
            )
            base_out.update(base_eval)
            base_out["expected_answer"] = exp_ans
            base_out["is_answerable"] = is_answerable
            base_out["question_type"] = q_type
            baseline_results.append(base_out)

            # 2. Run TrustRAG
            trust_out = self.pipeline.run_trust_rag(q)
            confidence_score = (
                trust_out["confidence_details"]["combined_confidence"]
                if trust_out.get("confidence_details")
                else 0.0
            )
            trust_eval = self._grade_response(
                response_text=trust_out["answer"],
                abstained=trust_out["abstained"],
                expected_answer=exp_ans,
                is_answerable=is_answerable,
                question_type=q_type,
                confidence=confidence_score,
            )
            trust_out.update(trust_eval)
            trust_out["expected_answer"] = exp_ans
            trust_out["is_answerable"] = is_answerable
            trust_out["question_type"] = q_type
            trustrag_results.append(trust_out)

            if progress_callback:
                progress_callback(idx + 1, total_samples)

        # Compute summary metrics for both systems
        baseline_metrics = self.compute_metrics(baseline_results, mode_name="Baseline RAG")
        trustrag_metrics = self.compute_metrics(trustrag_results, mode_name="TrustRAG")

        return {
            "baseline_results": baseline_results,
            "trustrag_results": trustrag_results,
            "baseline_metrics": baseline_metrics,
            "trustrag_metrics": trustrag_metrics,
        }

    def _grade_response(
        self,
        response_text: str,
        abstained: bool,
        expected_answer: str,
        is_answerable: bool,
        question_type: str,
        confidence: float,
    ) -> Dict[str, Any]:
        """
        Grade a response based on ground-truth answerability and expected answer.
        Policy:
        - If Answerable:
            - Abstained: Incorrect (Unnecessary abstention / coverage loss)
            - Answered: Evaluated for correctness against expected answer.
        - If Unanswerable:
            - Abstained: Correct (Appropriate abstention)
            - Answered: Unsupported / Hallucination
        - If Misleading:
            - Abstained or identified false premise: Correct
            - Blindly Answered: Unsupported / Hallucination
        """
        response_lower = response_text.lower().strip()
        expected_lower = expected_answer.lower().strip()

        # Check for abstention keywords in response even if not strictly flagged
        is_refusal = (
            abstained
            or "insufficient" in response_lower
            or "cannot provide" in response_lower
            or "not mentioned" in response_lower
            or "i cannot" in response_lower
        )

        is_correct = False
        is_unsupported = False

        if is_answerable:
            if is_refusal:
                # Unnecessary refusal on answerable question
                is_correct = False
                is_unsupported = False
            else:
                # Basic semantic/keyword matching check for automatic benchmarking
                keywords = [w for w in expected_lower.split() if len(w) > 3]
                match_count = sum(1 for kw in keywords if kw in response_lower)
                overlap_ratio = match_count / len(keywords) if keywords else 0.5

                if overlap_ratio >= 0.5 or expected_lower in response_lower:
                    is_correct = True
                    is_unsupported = False
                else:
                    is_correct = False
                    is_unsupported = True
        else:
            # Unanswerable or Misleading question
            if is_refusal:
                is_correct = True
                is_unsupported = False
            else:
                # Attempted to answer an unanswerable/misleading question -> Unsupported/Hallucination
                is_correct = False
                is_unsupported = True

        return {
            "graded_correct": is_correct,
            "graded_unsupported": is_unsupported,
            "graded_abstained": is_refusal,
            "evaluated_confidence": confidence,
        }

    @staticmethod
    def compute_metrics(results: List[Dict[str, Any]], mode_name: str) -> Dict[str, Any]:
        """
        Compute standard benchmark metrics with explicit denominators.
        """
        n = len(results)
        if n == 0:
            return {}

        total_correct = sum(1 for r in results if r["graded_correct"])
        total_abstained = sum(1 for r in results if r["graded_abstained"])
        total_attempted = n - total_abstained
        total_unsupported = sum(1 for r in results if r["graded_unsupported"])

        # Answerable subset
        answerable_results = [r for r in results if r["is_answerable"]]
        n_answerable = len(answerable_results)
        answerable_attempted = sum(1 for r in answerable_results if not r["graded_abstained"])
        answerable_correct = sum(1 for r in answerable_results if r["graded_correct"])

        # Unanswerable subset
        unanswerable_results = [r for r in results if not r["is_answerable"]]
        n_unanswerable = len(unanswerable_results)
        correct_abstentions = sum(1 for r in unanswerable_results if r["graded_abstained"])

        # Metric A: Answer Accuracy
        accuracy = round(total_correct / n, 4)

        # Metric B: Hallucination / Unsupported-Answer Rate
        # Denominator 1: attempted answers
        unsupported_rate_attempted = (
            round(total_unsupported / total_attempted, 4) if total_attempted > 0 else 0.0
        )
        # Denominator 2: all questions
        unsupported_rate_total = round(total_unsupported / n, 4)

        # Metric C: Abstention Rate
        abstention_rate = round(total_abstained / n, 4)

        # Metric D: Answerable Coverage
        coverage = (
            round(answerable_attempted / n_answerable, 4) if n_answerable > 0 else 0.0
        )

        # Metric E: Correct Abstention Rate (Specificity on unanswerable)
        correct_abstention_rate = (
            round(correct_abstentions / n_unanswerable, 4) if n_unanswerable > 0 else 0.0
        )

        # Metric F: Abstention Precision (Proportion of abstentions that were on unanswerable questions)
        abstention_precision = (
            round(correct_abstentions / total_abstained, 4) if total_abstained > 0 else 0.0
        )

        # Metric G: Expected Calibration Error (ECE)
        ece, calibration_bins = EvaluationEngine.compute_ece(results, num_bins=5)

        # Metric H: Risk-Coverage Curve points
        risk_coverage = EvaluationEngine.compute_risk_coverage(results)

        # Metric I: Latency
        latencies = [r.get("latency_seconds", 0.0) for r in results]
        avg_latency = round(float(np.mean(latencies)), 2) if latencies else 0.0

        return {
            "mode": mode_name,
            "total_questions": n,
            "total_correct": total_correct,
            "total_attempted": total_attempted,
            "total_abstained": total_abstained,
            "total_unsupported": total_unsupported,
            "accuracy": accuracy,
            "unsupported_rate_attempted": unsupported_rate_attempted,
            "unsupported_rate_total": unsupported_rate_total,
            "abstention_rate": abstention_rate,
            "answerable_coverage": coverage,
            "correct_abstention_rate": correct_abstention_rate,
            "abstention_precision": abstention_precision,
            "ece": ece,
            "calibration_bins": calibration_bins,
            "risk_coverage": risk_coverage,
            "avg_latency": avg_latency,
        }

    @staticmethod
    def compute_ece(
        results: List[Dict[str, Any]], num_bins: int = 5
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Compute Expected Calibration Error (ECE) and reliability bin statistics.
        ECE = sum_m ( |B_m| / N ) * |acc(B_m) - conf(B_m)|
        """
        confidences = np.array([r.get("evaluated_confidence", 0.5) for r in results])
        accuracies = np.array([1.0 if r.get("graded_correct") else 0.0 for r in results])
        n = len(confidences)

        if n == 0:
            return 0.0, []

        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        bin_stats = []

        for i in range(num_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            if i == num_bins - 1:
                in_bin = (confidences >= low) & (confidences <= high)
            else:
                in_bin = (confidences >= low) & (confidences < high)

            count = int(np.sum(in_bin))
            if count > 0:
                bin_acc = float(np.mean(accuracies[in_bin]))
                bin_conf = float(np.mean(confidences[in_bin]))
                ece += (count / n) * abs(bin_acc - bin_conf)
                bin_stats.append({
                    "bin_range": f"{low:.1f}-{high:.1f}",
                    "count": count,
                    "accuracy": round(bin_acc, 4),
                    "confidence": round(bin_conf, 4),
                    "error": round(abs(bin_acc - bin_conf), 4),
                })
            else:
                bin_stats.append({
                    "bin_range": f"{low:.1f}-{high:.1f}",
                    "count": 0,
                    "accuracy": 0.0,
                    "confidence": (low + high) / 2.0,
                    "error": 0.0,
                })

        return round(float(ece), 4), bin_stats

    @staticmethod
    def compute_risk_coverage(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compute empirical Risk-Coverage tradeoff across threshold sweeps in [0.0, 1.0].
        Coverage: Attempted / N
        Risk: Incorrect Attempted / Total Attempted
        """
        n = len(results)
        if n == 0:
            return []

        thresholds = np.linspace(0.2, 0.9, 8)
        points = []

        for tau in thresholds:
            # Under threshold tau, answer is attempted only if confidence >= tau
            attempted = [
                r for r in results if r.get("evaluated_confidence", 0.5) >= tau
            ]
            num_att = len(attempted)
            coverage = num_att / n

            if num_att > 0:
                incorrect = sum(1 for r in attempted if not r.get("graded_correct"))
                risk = incorrect / num_att
            else:
                risk = 0.0

            points.append({
                "threshold": round(float(tau), 2),
                "coverage": round(float(coverage), 4),
                "risk": round(float(risk), 4),
            })

        return points
