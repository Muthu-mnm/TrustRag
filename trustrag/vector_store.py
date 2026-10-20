"""
Vector store management using FAISS or ChromaDB.
Handles embedding generation with nomic-embed-text and top-k retrieval with normalized similarity scores.
"""

from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


class VectorStoreManager:
    """Manages document embedding and vector retrieval using FAISS."""

    def __init__(
        self,
        embed_model: str = "nomic-embed-text",
        embed_base_url: str = "http://localhost:11434",
    ):
        self.embed_model = embed_model
        self.embed_base_url = embed_base_url.rstrip("/")
        self.embeddings = OllamaEmbeddings(
            model=self.embed_model,
            base_url=self.embed_base_url,
            client_kwargs={"headers": {"ngrok-skip-browser-warning": "1"}},
        )
        self.vector_store: Optional[FAISS] = None

    def build_index(self, documents: List[Document]) -> int:
        """
        Build a FAISS vector index from chunked documents.
        Returns the number of documents indexed.
        """
        if not documents:
            raise ValueError("No documents provided for indexing.")

        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        return len(documents)

    def retrieve(
        self, query: str, top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant chunks for a given query with similarity scores.
        
        FAISS computes L2 distance: d in [0, inf).
        We transform distance into a normalized heuristic similarity score in (0, 1]:
            similarity_score = 1.0 / (1.0 + distance)
        
        Note: This is a monotonic ranking heuristic, NOT a calibrated probability.
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store is not initialized. Upload a PDF first.")

        docs_and_scores = self.vector_store.similarity_search_with_score(query, k=top_k)

        results = []
        for doc, distance in docs_and_scores:
            # Monotonic distance to similarity transform
            # Guard against negative or NaN distances
            dist = max(0.0, float(distance))
            similarity_score = round(1.0 / (1.0 + dist), 4)

            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "page": doc.metadata.get("page", 1),
                "source": doc.metadata.get("source", "document"),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "distance": round(dist, 4),
                "similarity_score": similarity_score,
            })

        return results

    @staticmethod
    def format_passages_for_prompt(retrieved_passages: List[Dict[str, Any]]) -> str:
        """Format retrieved passages into a structured text block for the LLM prompt."""
        if not retrieved_passages:
            return "No relevant passages found."

        formatted_blocks = []
        for idx, p in enumerate(retrieved_passages, 1):
            header = f"[Passage {idx} | Source: {p['source']} | Page: {p['page']} | Relevance: {p['similarity_score']}]"
            formatted_blocks.append(f"{header}\n{p['content']}")

        return "\n\n".join(formatted_blocks)
