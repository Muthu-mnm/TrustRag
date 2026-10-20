"""
Robust HTTP client for communicating with local or remote Ollama instances.
Handles ngrok bypass headers, custom timeouts, error handling, and robust JSON parsing.
"""

import json
import re
import requests
from typing import Dict, Any, Optional, List


class OllamaClient:
    """Wrapper around Ollama HTTP API with ngrok support and robust JSON extraction."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "1",
            "User-Agent": "TrustRAG-Client/1.0",
        }

    def check_connection(self) -> Dict[str, Any]:
        """Check if Ollama server is reachable and list available models."""
        url = f"{self.base_url}/api/tags"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name") for m in data.get("models", [])]
                return {"connected": True, "models": models, "error": None}
            return {
                "connected": False,
                "models": [],
                "error": f"HTTP {resp.status_code}: {resp.text[:100]}",
            }
        except Exception as e:
            return {"connected": False, "models": [], "error": str(e)}

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
    ) -> str:
        """Call Ollama generate endpoint synchronously."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if system:
            payload["system"] = system

        try:
            resp = requests.post(url, json=payload, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Ollama request to '{model}' timed out after {self.timeout}s at {self.base_url}."
            )
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error calling Ollama at {self.base_url}: {e}")

    def embed(self, model: str, prompt: str) -> List[float]:
        """Generate embedding vector for a given text."""
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": model, "prompt": prompt}
        try:
            resp = requests.post(url, json=payload, headers=self.headers, timeout=45)
            resp.raise_for_status()
            return resp.json().get("embedding", [])
        except Exception as e:
            raise ConnectionError(f"Error getting embeddings from {self.base_url}: {e}")

    @staticmethod
    def extract_json(raw_text: str) -> Dict[str, Any]:
        """
        Safely extract and parse JSON from LLM output, handling markdown fences,
        trailing commas, or leading/trailing commentary.
        """
        if not raw_text:
            return {}

        # 1. Try direct parsing
        try:
            return json.loads(raw_text)
        except Exception:
            pass

        # 2. Extract content from markdown code fences ```json ... ```
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except Exception:
                pass

        # 3. Find the first '{' and the last '}'
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_candidate = raw_text[start : end + 1]
            try:
                return json.loads(json_candidate)
            except Exception:
                # Remove common LLM formatting issues like trailing commas
                cleaned = re.sub(r",\s*([\]}])", r"\1", json_candidate)
                try:
                    return json.loads(cleaned)
                except Exception:
                    pass

        # 4. Fallback: heuristic key-value extraction
        result = {}
        for key in ["sufficient", "supported"]:
            match = re.search(rf'"{key}"\s*:\s*(true|false)', raw_text, re.IGNORECASE)
            if match:
                result[key] = match.group(1).lower() == "true"

        conf_match = re.search(r'"confidence"\s*:\s*([0-9\.]+)', raw_text)
        if conf_match:
            try:
                result["confidence"] = float(conf_match.group(1))
            except ValueError:
                pass

        reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', raw_text)
        if reason_match:
            result["reason"] = reason_match.group(1)

        return result
