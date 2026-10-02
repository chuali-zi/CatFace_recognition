"""The public boundary shared with consumers of recognition results."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Protocol


@dataclass(frozen=True)
class RecognitionResult:
    status: Literal["known", "unknown", "no_cat", "unusable"]
    cat_id: str | None = None
    score: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"known", "unknown", "no_cat", "unusable"}:
            raise ValueError("Invalid recognition status")
        if self.status == "known":
            if not isinstance(self.cat_id, str) or not self.cat_id.strip():
                raise ValueError("A known result needs a cat_id")
            if self.score is None:
                raise ValueError("A known result needs a score")
        elif self.cat_id is not None:
            raise ValueError("Only known results carry a cat_id")
        if self.score is not None and not isfinite(self.score):
            raise ValueError("score must be finite")
        if self.status in {"no_cat", "unusable"} and self.score is not None:
            raise ValueError("No identity score exists for this status")
        if self.status == "unusable" and not self.reason:
            raise ValueError("An unusable result needs a reason")


class Recognizer(Protocol):
    def recognize(self, image: bytes) -> RecognitionResult:
        """Recognize an encoded image using an already configured identity gallery.

        Raise on model/runtime failure. Never invent an identity or silently
        substitute a demonstration backend.
        """
        ...
