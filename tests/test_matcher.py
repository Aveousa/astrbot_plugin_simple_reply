import unittest

from matcher import WakeWordMatcher


class WakeWordMatcherTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
