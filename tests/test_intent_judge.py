import asyncio
import unittest

from intent_judge import (
    MentionIntentJudge,
    contains_explicit_no_reply,
    parse_reply_decision,
)


class IntentJudgeParsingTests(unittest.TestCase):
    def test_parses_json_decisions(self) -> None:
        self.assertTrue(parse_reply_decision('{"reply": true}'))
        self.assertFalse(parse_reply_decision('{"reply": false}'))

    def test_parses_fenced_or_embedded_json(self) -> None:
        self.assertTrue(parse_reply_decision("```json\n{\"reply\": true}\n```"))
        self.assertFalse(parse_reply_decision('结果：{"should_reply": false}'))

    def test_ambiguous_output_is_safe(self) -> None:
        self.assertFalse(parse_reply_decision("我认为这条消息可能需要回复"))

    def test_explicit_no_reply_is_detected(self) -> None:
        self.assertTrue(contains_explicit_no_reply("大家讨论就好，不用回复"))
        self.assertFalse(contains_explicit_no_reply("姬姬，帮我查一下天气"))


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.completion_text = text


class FakeEvent:
    unified_msg_origin = "test:GroupMessage:1"


class FakeContext:
    def __init__(self, response: str = '{"reply": true}') -> None:
        self.response = response
        self.calls = 0
        self.last_kwargs = None

    async def get_current_chat_provider_id(self, umo: str) -> str:
        return "current-provider"

    async def llm_generate(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return FakeResponse(self.response)


class IntentJudgeCallTests(unittest.TestCase):
    def test_explicit_no_reply_skips_provider_call(self) -> None:
        context = FakeContext()
        judge = MentionIntentJudge(context)

        result = asyncio.run(
            judge.should_reply(
                event=FakeEvent(),
                message="不用回复，只是讨论一下",
                matched_keyword="姬姬",
            ),
        )

        self.assertFalse(result)
        self.assertEqual(context.calls, 0)

    def test_calls_selected_provider_and_accepts_reply(self) -> None:
        context = FakeContext('{"reply": true}')
        judge = MentionIntentJudge(context, provider_id="small-model")

        result = asyncio.run(
            judge.should_reply(
                event=FakeEvent(),
                message="姬姬，帮我总结一下",
                matched_keyword="姬姬",
            ),
        )

        self.assertTrue(result)
        self.assertEqual(context.calls, 1)

    def test_prompt_accepts_direct_online_check_without_task_word(self) -> None:
        context = FakeContext('{"reply": true}')
        judge = MentionIntentJudge(context, provider_id="small-model")

        result = asyncio.run(
            judge.should_reply(
                event=FakeEvent(),
                message="在吗姬姬？",
                matched_keyword="姬姬",
            ),
        )

        self.assertTrue(result)
        self.assertIsNotNone(context.last_kwargs)
        system_prompt = context.last_kwargs["system_prompt"]
        self.assertIn("在吗{keyword}", system_prompt)
        self.assertIn("不要求用户一定提出具体任务", system_prompt)
