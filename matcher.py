"""Wake-word matching utilities.

This module deliberately has no AstrBot dependency so the matching rules can be
tested without starting an AstrBot instance.
"""

from __future__ import annotations

from collections.abc import Iterable


def message_text_without_at_mentions(
    message: str,
    segments: Iterable[object] | None,
) -> str:
    """Return text for matching without rendered names from @ segments.

    Some adapters include a mentioned user's display name in ``message`` even
    though that name is metadata on an ``At`` segment, not text the sender
    typed. When an @ segment is present, rebuild the match text from plain
    text segments so the target's nickname cannot accidentally wake the bot.
    """

    if not isinstance(message, str):
        return ""
    if segments is None:
        return message

    try:
        components = tuple(segments)
    except TypeError:
        return message

    mention_types = {"at", "atall"}
    if not any(
        _component_type_name(component) in mention_types for component in components
    ):
        return message

    text_parts: list[str] = []
    for component in components:
        component_type = _component_type_name(component)
        if component_type in mention_types:
            # Keep a non-whitespace boundary so text on either side of an @
            # cannot combine into a wake word, and text after a leading @ does
            # not become a false sentence-initial (direct) wake word.
            text_parts.append("\u200b")
            continue
        if component_type not in {"plain", "text"}:
            continue
        text = getattr(component, "text", None)
        if isinstance(text, str):
            text_parts.append(text)
    return "".join(text_parts)


def _component_type_name(component: object) -> str:
    component_type = getattr(component, "type", None)
    component_type = getattr(component_type, "value", component_type)
    return component_type.casefold() if isinstance(component_type, str) else ""


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
