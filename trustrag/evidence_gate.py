"""
Evidence Sufficiency Gate component for TrustRAG.
Evaluates whether retrieved passages contain sufficient factual evidence to answer a question.
"""

from typing import Dict, Any, List
from trustrag.ollama_client import OllamaClient
from trustrag.config import EVIDENCE_SUFFICIENCY_PROMPT


class EvidenceSufficiencyGate:
    """Evaluates factual sufficiency of retrieved passages before answer generation."""

    def __init__(self, ollama_client: OllamaClient, model_name: str = "gemma4:26b"):
        self.client = ollama_client
        self.model_name = model_name

    def assess_sufficiency(
        self,
        question: str,
        formatted_context: str,
        retrieved_passages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Assess evidence sufficiency using the LLM auditor prompt.
        Safely validates and clamps fields into:
        {
            "sufficient": bool,
            "confidence": float (clamped 0.0 - 1.0),
            "sufficiency_level": str ("sufficient", "partially_sufficient", "insufficient"),
            "reason": str,
            "missing_information": str,
            "raw_output": str
        }
        """
        # Edge case: No passages retrieved at all
        if not retrieved_passages:
            return {
                "sufficient": False,
                "confidence": 0.0,
                "sufficiency_level": "insufficient",
                "reason": "No evidence passages were retrieved for the question.",
                "missing_information": "Entire factual context is missing.",
                "raw_output": "[]",
            }

        prompt = EVIDENCE_SUFFICIENCY_PROMPT.format(
            context=formatted_context,
            question=question,
        )

        try:
            raw_response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                temperature=0.0,
            )
            parsed = self.client.extract_json(raw_response)

            # Extract and validate fields
            sufficient_raw = parsed.get("sufficient")
            if isinstance(sufficient_raw, bool):
                sufficient = sufficient_raw
            elif isinstance(sufficient_raw, str):
                sufficient = sufficient_raw.strip().lower() == "true"
            else:
                sufficient = False

            # Self-reported confidence score: validate and clamp between 0.0 and 1.0
            confidence_raw = parsed.get("confidence", 0.5)
            try:
                confidence = float(confidence_raw)
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 0.5

            # Reason and missing information
            reason = str(parsed.get("reason", "No reason provided by model.")).strip()
            missing_info = str(parsed.get("missing_information", "")).strip()

            # Determine granular sufficiency level
            sufficiency_level = str(parsed.get("sufficiency_level", "")).lower()
            if sufficiency_level not in ["sufficient", "partially_sufficient", "insufficient"]:
                if sufficient and confidence >= 0.7:
                    sufficiency_level = "sufficient"
                elif sufficient and confidence < 0.7:
                    sufficiency_level = "partially_sufficient"
                else:
                    sufficiency_level = "insufficient"

            # If marked partially sufficient, we treat as insufficient under strict abstention
            if sufficiency_level == "partially_sufficient":
                sufficient = False

            return {
                "sufficient": sufficient,
                "confidence": round(confidence, 4),
                "sufficiency_level": sufficiency_level,
                "reason": reason,
                "missing_information": missing_info,
                "raw_output": raw_response,
            }

        except Exception as e:
            # Fallback on network or parsing failure: abstain conservatively
            return {
                "sufficient": False,
                "confidence": 0.0,
                "sufficiency_level": "insufficient",
                "reason": f"Evidence sufficiency assessment failed: {str(e)}",
                "missing_information": "Assessment error prevented verification.",
                "raw_output": str(e),
            }
