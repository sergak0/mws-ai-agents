import json
import os
import re
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class OpenSourceLLM:
    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model

    def is_available(self) -> bool:
        if self.provider == "openrouter":
            return bool(os.getenv("OPENROUTER_API_KEY"))
        if self.provider == "ollama":
            return True
        return False

    def generate_structured(
        self,
        schema: type[SchemaT],
        system_prompt: str,
        user_prompt: str,
    ) -> SchemaT:
        client = self._build_client()
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        payload = _extract_json(content)
        return schema.model_validate(payload)

    def _build_client(self) -> OpenAI:
        if self.provider == "openrouter":
            return OpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
            )
        if self.provider == "ollama":
            return OpenAI(
                api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            )
        raise ValueError(f"Unsupported provider: {self.provider}")


def _extract_json(content: str) -> dict:
    fenced_match = re.search(r"```json\s*(\{[\s\S]*\})\s*```", content)
    if fenced_match:
        return json.loads(fenced_match.group(1))
    raw_match = re.search(r"\{[\s\S]*\}", content)
    if raw_match:
        return json.loads(raw_match.group(0))
    raise ValueError("No JSON object found in model response")
