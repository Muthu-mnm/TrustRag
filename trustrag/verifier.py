"""
Answer-support verification step for TrustRAG.
Verifies whether factual claims in the candidate answer are grounded in the retrieved passages.
"""

from typing import Dict, Any, List
from trustrag.ollama_client import OllamaClient
from trustrag.config import ANSWER_VERIFICATION_PROMPT


class AnswerVerifier:
    """Verifies that generated candidate answers do not contain unsupported claims or hallucinations."""

    def __init__(self, ollama_client: OllamaClient, model_name: str = "gemma4:26b"):
        self.client = ollama_client
        self.model_name = model_name

    def verify_answer(
        self,
        question: str,
        candidate_answer: str,
        formatted_context: str,
    ) -> Dict[str, Any]:
        """
        Verify candidate answer against retrieved evidence.
        Returns:
        {
            "supported": bool,
            "confidence": float (clamped 0.0 - 1.0),
            "reason": str,
            "raw_output": str
        }
        """
        prompt = ANSWER_VERIFICATION_PROMPT.format(
            context=formatted_context,
            question=question,
            candidate_answer=candidate_answer,
        )

        try:
            raw_response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                temperature=0.0,
            )
            parsed = self.client.extract_json(raw_response)

            # Validate supported boolean
            supported_raw = parsed.get("supported")
            if isinstance(supported_raw, bool):
                supported = supported_raw
            elif isinstance(supported_raw, str):
                supported = supported_raw.strip().lower() == "true"
            else:
                supported = False

            # Validate confidence
            conf_raw = parsed.get("confidence", 0.5)
            try:
                confidence = float(conf_raw)
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 0.5

            reason = str(parsed.get("reason", "No reason provided by verifier.")).strip()

            return {
                "supported": supported,
                "confidence": round(confidence, 4),
                "reason": reason,
                "raw_output": raw_response,
            }

        except Exception as e:
            return {
                "supported": False,
                "confidence": 0.0,
                "reason": f"Verification error: {str(e)}",
                "raw_output": str(e),
            }
