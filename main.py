"""AstrBot loose wake-word plugin entrypoint."""

from __future__ import annotations

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import CustomFilter
from astrbot.api.star import Context, Star
from astrbot.core.platform.message_type import MessageType

from .intent_judge import MentionIntentJudge, contains_explicit_no_reply
from .matcher import WakeWordMatcher

_MATCHED_WORD_EXTRA_KEY = "astrbot_plugin_simple_reply.matched_wake_word"
_TRIGGER_TYPE_EXTRA_KEY = "astrbot_plugin_simple_reply.trigger_type"
_TRIGGER_WAKE_WORD = "wake_word"
_TRIGGER_MENTION_KEYWORD = "mention_keyword"
_REACTION_PENDING_EXTRA_KEY = "astrbot_plugin_simple_reply.reply_emoji_pending"


class LooseWakeWordFilter(CustomFilter):
    """Pass when a wake word or group-chat mention keyword is found."""

    _wake_matcher = WakeWordMatcher(())
    _mention_matcher = WakeWordMatcher(())
    _intent_judge_enabled = False

    @classmethod
    def configure(
        cls,
        wake_words: list[object],
        mention_keywords: list[object],
        *,
        case_sensitive: bool,
        intent_judge_enabled: bool,
    ) -> None:
        cls._wake_matcher = WakeWordMatcher(
            wake_words,
            case_sensitive=case_sensitive,
        )
        cls._mention_matcher = WakeWordMatcher(
            mention_keywords,
            case_sensitive=case_sensitive,
        )
        cls._intent_judge_enabled = intent_judge_enabled

    def filter(self, event: AstrMessageEvent, cfg: AstrBotConfig) -> bool:
        del cfg  # Matching is controlled by this plugin's own configuration.
        message = event.get_message_str()
        is_group = event.get_message_type() == MessageType.GROUP_MESSAGE
        wake_word = self._wake_matcher.find(message)
        mention_keyword = (
            self._mention_matcher.find(message)
            if self._intent_judge_enabled and is_group
            else None
        )
        leading_wake_word = self._wake_matcher.find_at_start(message)
        leading_mention_keyword = (
            self._mention_matcher.find_at_start(message) if mention_keyword else None
        )

        # A leading configured word is already an explicit request. It should
        # not incur the extra classifier call, even when the same word also
        # appears in mention_keywords.
        leading_word = leading_wake_word or leading_mention_keyword
        if wake_word is not None:
            logger.info(
                "检测到唤醒词：%s（位置：%s）",
                wake_word,
                "句首" if leading_wake_word is not None else "句中",
            )
        if mention_keyword is not None:
            logger.info(
                "检测到群聊提及关键词：%s（位置：%s）",
                mention_keyword,
                "句首" if leading_mention_keyword is not None else "句中",
            )

        if leading_word is not None:
            logger.info("句首命中，直接进入 AstrBot 原生唤醒处理链：%s", leading_word)
            event.set_extra(_TRIGGER_TYPE_EXTRA_KEY, _TRIGGER_WAKE_WORD)
            event.set_extra(_MATCHED_WORD_EXTRA_KEY, leading_word)
            return True

        matched_word = mention_keyword or wake_word
        if matched_word is None:
            return False

        # Native @/wake-prefix handling, private chats, and compatibility mode
        # without intent judging remain direct triggers. Only a non-leading
        # match in a group can enter the classifier path.
        if (
            event.is_at_or_wake_command
            or not self._intent_judge_enabled
            or not is_group
        ):
            if event.is_at_or_wake_command:
                direct_reason = "原生 @/唤醒前缀"
            elif not self._intent_judge_enabled:
                direct_reason = "未启用意图判断"
            else:
                direct_reason = "非群聊"
            logger.info(
                "句中命中后直接进入 AstrBot 原生唤醒处理链：%s（原因：%s）",
                matched_word,
                direct_reason,
            )
            event.set_extra(_TRIGGER_TYPE_EXTRA_KEY, _TRIGGER_WAKE_WORD)
            event.set_extra(_MATCHED_WORD_EXTRA_KEY, matched_word)
            return True

        logger.info("群聊句中命中，进入意图判断：%s", matched_word)
        event.set_extra(_TRIGGER_TYPE_EXTRA_KEY, _TRIGGER_MENTION_KEYWORD)
        event.set_extra(_MATCHED_WORD_EXTRA_KEY, matched_word)
        return True


class SimpleReplyPlugin(Star):
    """Wake AstrBot when a configured word occurs anywhere in a message."""

    def __init__(
        self,
        context: Context,
        config: AstrBotConfig | None = None,
    ) -> None:
        # The loader normally supplies AstrBotConfig from _conf_schema.json.
        # Keep a plain dict fallback so direct tests do not initialize a global
        # AstrBot configuration file just by constructing the plugin.
        config = config if config is not None else {}
        super().__init__(context, config)
        self.config = config

        configured_words = config.get("wake_words", ["AstrBot"])
        if not isinstance(configured_words, list):
            configured_words = [configured_words]
        mention_keywords = config.get("mention_keywords", [])
        if not isinstance(mention_keywords, list):
            mention_keywords = [mention_keywords]

        intent_judge_enabled = bool(config.get("enable_intent_judge", False))

        LooseWakeWordFilter.configure(
            configured_words,
            mention_keywords,
            case_sensitive=bool(config.get("case_sensitive", False)),
            intent_judge_enabled=intent_judge_enabled,
        )
        self.intent_judge = MentionIntentJudge(
            context,
            provider_id=self._get_provider_id(config),
            timeout_seconds=self._get_timeout(config),
        )
        self.reply_emoji_enabled = bool(config.get("enable_reply_emoji", False))
        self.reply_emoji_id = self._get_reply_emoji_id(config)

    @staticmethod
    def _get_provider_id(config: AstrBotConfig) -> str:
        provider_id = config.get("intent_judge_provider", "")
        return provider_id.strip() if isinstance(provider_id, str) else ""

    @staticmethod
    def _get_timeout(config: AstrBotConfig) -> float:
        timeout = config.get("intent_judge_timeout", 3)
        try:
            return float(timeout)
        except (TypeError, ValueError):
            return 3.0

    @staticmethod
    def _get_reply_emoji_id(config: AstrBotConfig) -> int:
        emoji_id = config.get("reply_emoji_id", 307)
        try:
            emoji_id = int(emoji_id)
        except (TypeError, ValueError):
            return 307
        return emoji_id if emoji_id > 0 else 307

    @filter.custom_filter(LooseWakeWordFilter)
    async def loose_wake(self, event: AstrMessageEvent) -> None:
        """Mark an in-sentence wake-word match for AstrBot's normal LLM flow."""

        trigger_type = event.get_extra(_TRIGGER_TYPE_EXTRA_KEY, "")
        matched_word = event.get_extra(_MATCHED_WORD_EXTRA_KEY, "")

        # An existing platform @, the native wake prefix, or a private message
        # already has the normal wake flag and should never be delayed by the
        # group-chat classifier.
        if (
            trigger_type == _TRIGGER_MENTION_KEYWORD
            and not event.is_at_or_wake_command
        ):
            provider_name = (
                self.intent_judge.provider_id
                or "当前会话对话模型"
            )
            explicit_no_reply = contains_explicit_no_reply(event.get_message_str())
            if explicit_no_reply:
                logger.info(
                    "检测到明确无需回复表达，跳过意图判断模型：%s",
                    matched_word,
                )
            else:
                logger.info(
                    "开始群聊提及意图判断：关键词=%s，模型=%s，超时=%.1fs",
                    matched_word,
                    provider_name,
                    self.intent_judge.timeout_seconds,
                )
            try:
                should_reply = await self.intent_judge.should_reply(
                    event=event,
                    message=event.get_message_str(),
                    matched_keyword=matched_word,
                )
            except TimeoutError:
                logger.warning("群聊提及意图判断超时，已忽略本条消息")
                return
            except Exception as exc:
                logger.warning("群聊提及意图判断失败，已忽略本条消息：%s", exc)
                return

            if not should_reply:
                logger.info("意图判断结果：无需回复：%s", matched_word)
                return

            logger.info("意图判断结果：需要回复，进入主 LLM：%s", matched_word)

        # WakingCheckStage already sets is_wake when this filter passes. Setting
        # both flags explicitly also documents the contract and keeps the
        # handler robust if the internal stage behavior changes.
        if self.reply_emoji_enabled:
            # The decorating-result hook runs only when AstrBot has a
            # non-empty response result. Keeping this marker on the event
            # avoids reacting to unrelated messages handled by other plugins.
            event.set_extra(_REACTION_PENDING_EXTRA_KEY, True)
        event.is_wake = True
        event.is_at_or_wake_command = True
        logger.debug("宽松唤醒词命中：%s", matched_word)

    @filter.on_decorating_result()
    async def react_before_replied_message(self, event: AstrMessageEvent) -> None:
        """Add a native OneBot emoji reaction immediately before the reply."""

        if not self.reply_emoji_enabled or not event.get_extra(
            _REACTION_PENDING_EXTRA_KEY,
            False,
        ):
            return

        if event.get_platform_name() != "aiocqhttp":
            logger.info(
                "当前平台不是 aiocqhttp，跳过回复前原生表情回应：%s",
                event.get_platform_name(),
            )
            return

        bot = getattr(event, "bot", None)
        message_id = getattr(event.message_obj, "message_id", None)
        try:
            message_id = int(message_id)
        except (TypeError, ValueError):
            message_id = None

        if bot is None or message_id is None:
            logger.warning("无法取得 aiocqhttp Bot 或原消息 ID，跳过回复前原生表情回应")
            return

        try:
            await bot.call_action(
                "set_msg_emoji_like",
                message_id=message_id,
                emoji_id=str(self.reply_emoji_id),
                set=True,
            )
        except Exception as exc:
            # Reaction support is an optional adapter extension; it must not
            # turn a successfully delivered LLM reply into a failed event.
            logger.warning(
                "回复前原生表情回应发送失败（message_id=%s, emoji_id=%s）：%s",
                message_id,
                self.reply_emoji_id,
                exc,
            )
            return

        logger.info(
            "已在发送回复正文前为原消息添加原生表情回应（message_id=%s, emoji_id=%s）",
            message_id,
            self.reply_emoji_id,
        )

    async def terminate(self) -> None:
        """Reset matcher state when the plugin is unloaded."""

        LooseWakeWordFilter.configure(
            [],
            [],
            case_sensitive=False,
            intent_judge_enabled=False,
        )
