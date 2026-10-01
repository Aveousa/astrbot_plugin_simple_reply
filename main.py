"""AstrBot loose wake-word plugin entrypoint."""

from __future__ import annotations

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import CustomFilter
from astrbot.api.star import Context, Star

from .matcher import WakeWordMatcher

_MATCHED_WORD_EXTRA_KEY = "astrbot_plugin_simple_reply.matched_wake_word"


class LooseWakeWordFilter(CustomFilter):
    """Pass only when the message contains a configured wake word."""

    _matcher = WakeWordMatcher(())

    @classmethod
    def configure(
        cls,
        wake_words: list[object],
        *,
        case_sensitive: bool,
    ) -> None:
        cls._matcher = WakeWordMatcher(
            wake_words,
            case_sensitive=case_sensitive,
        )

    def filter(self, event: AstrMessageEvent, cfg: AstrBotConfig) -> bool:
        del cfg  # Matching is controlled by this plugin's own configuration.
        matched_word = self._matcher.find(event.get_message_str())
        if matched_word is None:
            return False
        event.set_extra(_MATCHED_WORD_EXTRA_KEY, matched_word)
        return True


class SimpleReplyPlugin(Star):
    """Wake AstrBot when a configured word occurs anywhere in a message."""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context, config)
        self.config = config

        configured_words = config.get("wake_words", ["AstrBot"])
        if not isinstance(configured_words, list):
            configured_words = [configured_words]

        LooseWakeWordFilter.configure(
            configured_words,
            case_sensitive=bool(config.get("case_sensitive", False)),
        )

    @filter.custom_filter(LooseWakeWordFilter)
    async def loose_wake(self, event: AstrMessageEvent) -> None:
        """Mark an in-sentence wake-word match for AstrBot's normal LLM flow."""

        # WakingCheckStage already sets is_wake when this filter passes. Setting
        # both flags explicitly also documents the contract and keeps the
        # handler robust if the internal stage behavior changes.
        event.is_wake = True
        event.is_at_or_wake_command = True
        matched_word = event.get_extra(_MATCHED_WORD_EXTRA_KEY, "")
        logger.debug("宽松唤醒词命中：%s", matched_word)

    async def terminate(self) -> None:
        """Reset matcher state when the plugin is unloaded."""

        LooseWakeWordFilter.configure([], case_sensitive=False)
