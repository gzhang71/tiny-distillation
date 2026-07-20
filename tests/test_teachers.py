import unittest
from types import SimpleNamespace

from tiny_distillation.core import TrainingExample
from tiny_distillation.teachers import (
    ChatGPTTeacher,
    ClaudeTeacher,
    DeepSeekTeacher,
    LlamaTeacher,
    Qwen35Teacher,
    T5Teacher,
    Teacher,
)


RESPONSE_JSON = (
    '{"answer": "yes", "reasoning": "The evidence supports yes.", '
    '"confidence": 0.8}'
)


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.request = None
        self.responses = self

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(id="openai-response", output_text=RESPONSE_JSON)


class FakeClaudeClient:
    def __init__(self) -> None:
        self.request = None
        self.messages = self

    def create(self, **kwargs):
        self.request = kwargs
        block = SimpleNamespace(type="text", text=f"```json\n{RESPONSE_JSON}\n```")
        return SimpleNamespace(id="claude-message", content=[block])


class FakeDeepSeekCompletions:
    def __init__(self) -> None:
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        message = SimpleNamespace(content=RESPONSE_JSON, reasoning_content=None)
        return SimpleNamespace(id="deepseek-completion", choices=[SimpleNamespace(message=message)])


class FakeDeepSeekClient:
    def __init__(self) -> None:
        self.completions = FakeDeepSeekCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class TeacherAdaptersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.example = TrainingExample(id="decision", prompt="Is the statement true?")

    def test_teacher_is_abstract(self) -> None:
        self.assertTrue(Teacher.__abstractmethods__)
        for implementation in (
            ChatGPTTeacher,
            ClaudeTeacher,
            DeepSeekTeacher,
            LlamaTeacher,
            T5Teacher,
            Qwen35Teacher,
        ):
            self.assertTrue(issubclass(implementation, Teacher))

    def test_chatgpt_responses_adapter(self) -> None:
        client = FakeOpenAIClient()
        teacher = ChatGPTTeacher(["no", "yes"], client=client)

        prediction = teacher.generate(self.example)

        self.assertEqual(prediction.answer, "yes")
        self.assertEqual(prediction.metadata["provider"], "openai")
        self.assertAlmostEqual(prediction.confidence, 0.8, places=5)
        self.assertIn("instructions", client.request)
        self.assertIn('"yes"', client.request["input"])

    def test_claude_messages_adapter(self) -> None:
        client = FakeClaudeClient()
        teacher = ClaudeTeacher(["no", "yes"], client=client)

        prediction = teacher.generate(self.example)

        self.assertEqual(prediction.reasoning, "The evidence supports yes.")
        self.assertEqual(prediction.metadata["response_id"], "claude-message")
        self.assertEqual(client.request["messages"][0]["role"], "user")

    def test_deepseek_chat_completions_adapter(self) -> None:
        client = FakeDeepSeekClient()
        teacher = DeepSeekTeacher(["no", "yes"], client=client, thinking=True)

        prediction = teacher.generate(self.example, include_reasoning=False)

        self.assertEqual(prediction.reasoning, "")
        self.assertEqual(prediction.metadata["provider"], "deepseek")
        self.assertEqual(
            client.completions.request["extra_body"],
            {"thinking": {"type": "enabled"}},
        )

    def test_unmapped_answer_fails_instead_of_creating_bad_labels(self) -> None:
        client = FakeOpenAIClient()
        client.create = lambda **kwargs: SimpleNamespace(
            output_text='{"answer": "maybe", "reasoning": "", "confidence": 0.5}'
        )
        teacher = ChatGPTTeacher(["no", "yes"], client=client)

        with self.assertRaisesRegex(ValueError, "does not match"):
            teacher.generate(self.example)

    def test_local_teachers_do_not_load_models_during_construction(self) -> None:
        tokenizer = object()
        model = object()
        teachers = [
            LlamaTeacher(["no", "yes"], tokenizer=tokenizer, model_instance=model),
            T5Teacher(["no", "yes"], tokenizer=tokenizer, model_instance=model),
            Qwen35Teacher(["no", "yes"], tokenizer=tokenizer, model_instance=model),
        ]

        self.assertEqual(
            [teacher.model for teacher in teachers],
            [
                "meta-llama/Llama-3.2-3B-Instruct",
                "google/flan-t5-base",
                "Qwen/Qwen3.5-2B",
            ],
        )


if __name__ == "__main__":
    unittest.main()

