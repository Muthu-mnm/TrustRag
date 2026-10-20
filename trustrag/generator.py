"""
Answer generation workflows for Baseline RAG and TrustRAG.
Handles both execution modes with page-level citations, bounded verification retry, and transparent traces.
"""

import time
from typing import Dict, Any, List, Optional
from trustrag.config import (
    BASELINE_RAG_PROMPT,
    GROUNDED_ANSWER_PROMPT,
    ABSTENTION_MESSAGE,
)
from trustrag.ollama_client import OllamaClient
from trustrag.vector_store import VectorStoreManager
from trustrag.evidence_gate import EvidenceSufficiencyGate
from trustrag.confidence import ConfidenceEstimator
from trustrag.verifier import AnswerVerifier


class RAGPipeline:
    """Orchestrates Baseline RAG and TrustRAG execution modes."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        vector_store_manager: VectorStoreManager,
        model_name: str = "gemma4:26b",
        abstention_threshold: float = 0.60,
    ):
        self.client = ollama_client
        self.vector_store = vector_store_manager
        self.model_name = model_name
        self.evidence_gate = EvidenceSufficiencyGate(self.client, model_name=model_name)
        self.confidence_estimator = ConfidenceEstimator(abstention_threshold=abstention_threshold)
        self.verifier = AnswerVerifier(self.client, model_name=model_name)

    def run_baseline_rag(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """
        Baseline RAG Mode:
        Retrieves passages and generates an answer without evidence sufficiency gating,
        confidence calibration, or post-generation verification.
        """
        start_time = time.time()
        passages = self.vector_store.retrieve(query, top_k=top_k)
        formatted_context = self.vector_store.format_passages_for_prompt(passages)

        prompt = BASELINE_RAG_PROMPT.format(context=formatted_context, question=query)
        try:
            answer = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                temperature=0.2,
            )
        except Exception as e:
            answer = f"Error generating baseline response: {str(e)}"

        elapsed = round(time.time() - start_time, 2)

        return {
            "mode": "Baseline RAG",
            "query": query,
            "answer": answer,
            "abstained": False,
            "abstention_reason": None,
            "retrieved_passages": passages,
            "latency_seconds": elapsed,
            "confidence_details": None,
            "sufficiency_assessment": None,
            "verification": None,
        }

    def run_trust_rag(
        self,
        query: str,
        top_k: int = 4,
        max_verification_retries: int = 1,
    ) -> Dict[str, Any]:
        """
        TrustRAG Mode:
        1. Retrieval of top-k passages with similarity scores.
        2. Evidence sufficiency assessment via Gemma.
        3. Transparent confidence estimation & abstention policy.
        4. Grounded answer generation with page citations.
        5. Answer-support verification step (with bounded retry).
        """
        start_time = time.time()

        # Step 1: Retrieval
        passages = self.vector_store.retrieve(query, top_k=top_k)
        formatted_context = self.vector_store.format_passages_for_prompt(passages)

        # Step 2: Evidence Sufficiency Assessment
        sufficiency = self.evidence_gate.assess_sufficiency(
            question=query,
            formatted_context=formatted_context,
            retrieved_passages=passages,
        )

        # Step 3: Confidence Estimation & Abstention Gate
        conf_eval = self.confidence_estimator.calculate_confidence(
            sufficiency_assessment=sufficiency,
            retrieved_passages=passages,
        )

        if conf_eval["should_abstain"]:
            elapsed = round(time.time() - start_time, 2)
            return {
                "mode": "TrustRAG",
                "query": query,
                "answer": ABSTENTION_MESSAGE,
                "abstained": True,
                "abstention_reason": conf_eval["abstention_reason"],
                "retrieved_passages": passages,
                "latency_seconds": elapsed,
                "confidence_details": conf_eval,
                "sufficiency_assessment": sufficiency,
                "verification": None,
            }

        # Step 4: Grounded Answer Generation
        gen_prompt = GROUNDED_ANSWER_PROMPT.format(
            context=formatted_context,
            question=query,
        )

        candidate_answer = self.client.generate(
            model=self.model_name,
            prompt=gen_prompt,
            temperature=0.1,
        )

        # Step 5: Answer-Support Verification Step
        verification = self.verifier.verify_answer(
            question=query,
            candidate_answer=candidate_answer,
            formatted_context=formatted_context,
        )

        # Bounded retry if verification fails
        retries_used = 0
        if not verification.get("supported", False) and max_verification_retries > 0:
            retry_prompt = (
                f"{gen_prompt}\n\n"
                f"[Feedback: Your previous answer was flagged as unsupported: {verification.get('reason')}. "
                f"Please produce a strictly grounded answer citing only proven facts, or state what cannot be answered.]\n"
                f"Revised Answer:"
            )
            candidate_answer = self.client.generate(
                model=self.model_name,
                prompt=retry_prompt,
                temperature=0.0,
            )
            retries_used += 1
            verification = self.verifier.verify_answer(
                question=query,
                candidate_answer=candidate_answer,
                formatted_context=formatted_context,
            )

        # If answer is still unverified after bounded retry, enforce abstention
        abstained = False
        abstention_reason = None
        final_answer = candidate_answer

        if not verification.get("supported", False):
            abstained = True
            abstention_reason = (
                f"Answer verification failed: {verification.get('reason', 'Unsupported claims detected.')}"
            )
            final_answer = ABSTENTION_MESSAGE

        elapsed = round(time.time() - start_time, 2)

        return {
            "mode": "TrustRAG",
            "query": query,
            "answer": final_answer,
            "abstained": abstained,
            "abstention_reason": abstention_reason,
            "retrieved_passages": passages,
            "latency_seconds": elapsed,
            "confidence_details": conf_eval,
            "sufficiency_assessment": sufficiency,
            "verification": verification,
            "verification_retries": retries_used,
        }
