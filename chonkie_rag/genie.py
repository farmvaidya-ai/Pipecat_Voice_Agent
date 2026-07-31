"""
VertexGeminiGenie — a chonkie BaseGenie backed by Vertex AI (project/location
auth), for use with SlumberChunker during KB ingestion.
"""

from google import genai

from chonkie.genie import BaseGenie
from .config import PROJECT_ID, LOCATION, MODEL_NAME


class VertexGeminiGenie(BaseGenie):
    """Gemini genie authenticated via Vertex AI, not a GEMINI_API_KEY."""

    def __init__(self, model: str = MODEL_NAME) -> None:
        super().__init__()
        self.model = model
        self.client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    def generate(self, prompt: str) -> str:
        """Generate a text response based on the given prompt."""
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return response.text or ""

    def __repr__(self) -> str:
        return f"VertexGeminiGenie(model={self.model})"
