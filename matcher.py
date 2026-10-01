"""Wake-word matching utilities.

This module deliberately has no AstrBot dependency so the matching rules can be
tested without starting an AstrBot instance.
"""

from __future__ import annotations

from collections.abc import Iterable


class WakeWordMatcher:
    """Match configured wake words anywhere in a message."""

    def __init__(
        self,
        wake_words: Iterable[object],
        *,
        case_sensitive: bool = False,
    ) -> None:
        self.case_sensitive = case_sensitive
        self.wake_words = self._normalize_words(wake_words)

    @staticmethod
    def _normalize_words(wake_words: Iterable[object]) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in wake_words:
            if not isinstance(value, str):
                continue
            word = value.strip()
            if not word or word in seen:
                continue
            normalized.append(word)
            seen.add(word)
        return tuple(normalized)

    def find(self, message: str) -> str | None:
        """Return the first configured wake word found in ``message``."""

        if not isinstance(message, str) or not message or not self.wake_words:
            return None

        if self.case_sensitive:
            return next((word for word in self.wake_words if word in message), None)

        folded_message = message.casefold()
        return next(
            (word for word in self.wake_words if word.casefold() in folded_message),
            None,
        )

    def matches(self, message: str) -> bool:
        """Return whether ``message`` contains a configured wake word."""

        return self.find(message) is not None

    def find_at_start(self, message: str) -> str | None:
        """Return the first configured word at the start of ``message``."""

        if not isinstance(message, str) or not message or not self.wake_words:
            return None

        message = message.lstrip()
        if self.case_sensitive:
            return next(
                (word for word in self.wake_words if message.startswith(word)),
                None,
            )

        folded_message = message.casefold()
        return next(
            (
                word
                for word in self.wake_words
                if folded_message.startswith(word.casefold())
            ),
            None,
        )
