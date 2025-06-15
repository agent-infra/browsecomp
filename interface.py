from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

# Types definitions
Message = dict[str, Any]  # keys role, content
MessageList = list[Message]


@dataclass
class SamplerResponse:
    """Response from a sampler."""
    response_text: str
    actual_queried_message_list: MessageList
    response_metadata: dict[str, Any]


class SamplerBase:
    """Base class for defining a sampling model."""

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        raise NotImplementedError

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}


@dataclass
class SingleEvalResult:
    """Result of evaluating a single sample"""
    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    html: str | None = None
    convo: MessageList | None = None
    example_level_metadata: dict[str, Any] | None = None


@dataclass
class EvalResult:
    """Result of running an evaluation (usually consisting of many samples)"""
    score: float | None
    metrics: dict[str, float] | None
    htmls: list[str]
    convos: list[MessageList]
    metadata: dict[str, Any] | None


class Eval:
    """Base class for defining an evaluation."""

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        raise NotImplementedError
