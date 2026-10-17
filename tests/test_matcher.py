import unittest

from matcher import WakeWordMatcher, message_text_without_at_mentions


class FakeSegment:
    def __init__(self, segment_type: str, **values) -> None:
        self.type = segment_type
        for key, value in values.items():
            setattr(self, key, value)


class WakeWordMatcherTests(unittest.TestCase):
    def test_excludes_at_target_nickname_from_match_text(self) -> None:
        segments = [
            FakeSegment("Plain", text="你们聊一下"),
            FakeSegment("At", qq=123, name="小助手"),
            FakeSegment("Plain", text="吧"),
        ]

        match_text = message_text_without_at_mentions(
            "你们聊一下 @小助手(123) 吧",
            segments,
        )

        self.assertEqual(match_text, "你们聊一下\u200b吧")
        self.assertFalse(WakeWordMatcher(["小助手"]).matches(match_text))

    def test_keeps_user_typed_wake_word_around_at_segment(self) -> None:
        segments = [
            FakeSegment("Plain", text="小助手，问问"),
            FakeSegment("At", qq=123, name="其他人"),
            FakeSegment("Plain", text="这个"),
        ]

        match_text = message_text_without_at_mentions(
            "小助手，问问 @其他人(123) 这个",
            segments,
        )

        self.assertTrue(WakeWordMatcher(["小助手"]).matches(match_text))

    def test_at_target_before_wake_word_does_not_make_it_leading(self) -> None:
        segments = [
            FakeSegment("At", qq=123, name="其他人"),
            FakeSegment("Plain", text="小助手，问问"),
        ]

        match_text = message_text_without_at_mentions(
            "@其他人(123) 小助手，问问",
            segments,
        )

        matcher = WakeWordMatcher(["小助手"])
        self.assertTrue(matcher.matches(match_text))
        self.assertIsNone(matcher.find_at_start(match_text))

    def test_returns_original_message_without_at_segments(self) -> None:
        message = "请让小助手总结这段内容"

        self.assertEqual(
            message_text_without_at_mentions(
                message,
                [FakeSegment("Plain", text=message)],
            ),
            message,
        )

    def test_matches_wake_word_anywhere_in_message(self) -> None:
        matcher = WakeWordMatcher(["小助手"])

        self.assertTrue(matcher.matches("请让小助手总结这段话"))

    def test_matches_case_insensitively_by_default(self) -> None:
        matcher = WakeWordMatcher(["AstrBot"])

        self.assertEqual(matcher.find("你好，astrbot！"), "AstrBot")

    def test_can_match_case_sensitively(self) -> None:
        matcher = WakeWordMatcher(["AstrBot"], case_sensitive=True)

        self.assertFalse(matcher.matches("你好，astrbot！"))
        self.assertTrue(matcher.matches("你好，AstrBot！"))

    def test_ignores_empty_duplicate_and_non_string_values(self) -> None:
        matcher = WakeWordMatcher(["  ", "小助手", "小助手", None, 123])

        self.assertEqual(matcher.wake_words, ("小助手",))

    def test_empty_configuration_matches_nothing(self) -> None:
        matcher = WakeWordMatcher([])

        self.assertFalse(matcher.matches("任意消息"))

    def test_finds_word_at_start_after_whitespace(self) -> None:
        matcher = WakeWordMatcher(["AstrBot"])

        self.assertEqual(matcher.find_at_start("  astrbot，请回答"), "AstrBot")
        self.assertIsNone(matcher.find_at_start("请问 astrbot 能回答吗"))


if __name__ == "__main__":
    unittest.main()
