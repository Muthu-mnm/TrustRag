"""
Confidence estimation and abstention policy manager for TrustRAG.
Provides transparent heuristic scoring combining retrieval signals and LLM self-reported estimates.
"""

from typing import Dict, Any, List
from trustrag.config import (
    DEFAULT_SUFFICIENCY_WEIGHT,
    DEFAULT_RETRIEVAL_WEIGHT,
    DEFAULT_ABSTENTION_THRESHOLD,
)


class ConfidenceEstimator:
    """
    Computes a transparent composite confidence score and enforces the abstention policy.
    
    Formula:
        Confidence = (w_suff * S_suff) + (w_ret * R_ret)
        
    Where:
        - S_suff in [0, 1]: Self-reported confidence from LLM Evidence Sufficiency Auditor.
        - R_ret in [0, 1]: Average normalized retrieval similarity of top retrieved passages.
        - w_suff, w_ret: Heuristic weights summing to 1.0 (default: 0.65 and 0.35).
        
    Scientific Note:
        The retrieval score is a normalized distance heuristic and the LLM score is a
        self-reported estimate. Neither is a calibrated posterior probability without
        empirical validation against a held-out dataset.
    """

    def __init__(
        self,
        weight_sufficiency: float = DEFAULT_SUFFICIENCY_WEIGHT,
        weight_retrieval: float = DEFAULT_RETRIEVAL_WEIGHT,
        abstention_threshold: float = DEFAULT_ABSTENTION_THRESHOLD,
    ):
        # Normalize weights so they sum to 1.0
        total = weight_sufficiency + weight_retrieval
        self.w_suff = weight_sufficiency / total if total > 0 else 0.5
        self.w_ret = weight_retrieval / total if total > 0 else 0.5
        self.abstention_threshold = abstention_threshold

    def calculate_confidence(
        self,
        sufficiency_assessment: Dict[str, Any],
        retrieved_passages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculate the combined confidence score and check abstention criteria.
        """
        # Signal 1: LLM self-reported sufficiency confidence
        s_suff = float(sufficiency_assessment.get("confidence", 0.0))

        # Signal 2: Retrieval similarity score (mean of top retrieved passages)
        if retrieved_passages:
            scores = [p.get("similarity_score", 0.5) for p in retrieved_passages]
            r_ret = sum(scores) / len(scores)
        else:
            r_ret = 0.0

        # Composite score
        combined_score = round((self.w_suff * s_suff) + (self.w_ret * r_ret), 4)

        # Abstention Decision Logic:
        # Abstain if:
        # 1. No passages retrieved
        # 2. Sufficiency gate evaluated as False
        # 3. Combined score falls below configurable threshold
        is_sufficient = sufficiency_assessment.get("sufficient", False)
        
        abstain = False
        abstention_reasons = []

        if not retrieved_passages:
            abstain = True
            abstention_reasons.append("No document passages could be retrieved for this query.")

        if not is_sufficient:
            abstain = True
            missing = sufficiency_assessment.get("missing_information", "")
            reason_text = sufficiency_assessment.get("reason", "Evidence does not support question.")
            abstention_reasons.append(f"Evidence gate failed: {reason_text}")
            if missing:
                abstention_reasons.append(f"Missing information: {missing}")

        if combined_score < self.abstention_threshold:
            abstain = True
            abstention_reasons.append(
                f"Combined confidence ({combined_score:.2f}) is below abstention threshold ({self.abstention_threshold:.2f})."
            )

        return {
            "combined_confidence": combined_score,
            "sufficiency_score": round(s_suff, 4),
            "retrieval_score": round(r_ret, 4),
            "weights": {"w_sufficiency": round(self.w_suff, 3), "w_retrieval": round(self.w_ret, 3)},
            "abstention_threshold": self.abstention_threshold,
            "should_abstain": abstain,
            "abstention_reason": " | ".join(abstention_reasons) if abstention_reasons else None,
        }
