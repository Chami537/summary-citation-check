"""The model interface article_summary.py is written against, copied verbatim from the service's
providers module (LlmReply and LlmProvider only). Anything with an async complete_json(system=, user=,
schema=) that returns an LlmReply will do: the tests use scripted fakes, and a real client is not part of
this export.
"""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class LlmReply:
    content: dict[str, Any]
    input_tokens: int | None = None
    output_tokens: int | None = None


class LlmProvider(Protocol):
    name: str
    model: str
    region: str

    async def complete_json(self, *, system: str, user: str, schema: dict[str, Any]) -> LlmReply:
        """Return a JSON object that the provider was asked to shape after `schema`."""
        ...
