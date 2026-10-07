import unittest

from shopping_grpo.training.grpo.adapter.tool_parser import ToolCallParseError, parse_tool_calls
from shopping_grpo.training.grpo.adapter.format_prompt import (
    NO_ARGUMENT_TOOLS, QWEN_TOOL_FORMAT_PROMPT, parse_error_feedback, qwen_tool_messages,
)


class ShoppingToolParserTest(unittest.TestCase):
    def test_literal_string_parameters_and_parameterless_calls(self):
        for value in ["null", "A&B < 100", "蓝色", "001234"]:
            text = f"<tool_call>\n<function=search_products>\n<parameter=query>{value}</parameter>\n</function>\n</tool_call><|im_end|>"
            self.assertEqual(parse_tool_calls(text), [("search_products", {"query": value})])
        self.assertEqual(parse_tool_calls("<tool_call><function=buy_now></function></tool_call>"), [("buy_now", {})])

    def test_incomplete_or_ambiguous_calls_never_execute(self):
        invalid = [
            "<tool_call><function=search_products><parameter=query",
            "<tool_call><function=search_products><parameter=query>shoe</function></tool_call>",
            "<tool_call><function=buy_now><parameter=</function></tool_call>",
            "<tool_call><function=search_products></function></tool_call>",
            "<tool_call><function=buy_now><parameter=extra>x</parameter></function></tool_call>",
            "<tool_call><function=unknown></function></tool_call>",
            "<tool_call><function=search_products><parameter=query>x</parameter><parameter=query>y</parameter></function></tool_call>",
            "<tool_call><function=buy_now></function></tool_call><tool_call>",
            "<function=buy_now></function>",
        ]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ToolCallParseError):
                parse_tool_calls(text)

    def test_all_calls_are_preserved_for_parallel_call_guard(self):
        text = "<tool_call><function=buy_now></function></tool_call>"
        self.assertEqual(len(parse_tool_calls(text + text)), 2)

    def test_plain_final_answer_has_no_calls(self):
        self.assertEqual(parse_tool_calls("Finished."), [])

    def test_actual_empty_object_error_gets_a_tool_specific_valid_example(self):
        for name in NO_ARGUMENT_TOOLS:
            text = f"<tool_call>\n<function={name}>\n<parameter={{}}\n</parameter>\n</function>\n</tool_call><|im_end|>"
            with self.subTest(name=name):
                with self.assertRaises(ToolCallParseError) as raised:
                    parse_tool_calls(text)
                error = raised.exception
                self.assertEqual(str(error), "unexpected_parameter_for_noarg_tool")
                self.assertEqual(error.tool_name, name)
                feedback = parse_error_feedback(error)
                self.assertNotIn("<parameter", feedback)
                self.assertEqual(parse_tool_calls(feedback), [(name, {})])

    def test_prompt_adapts_cached_json_wording_without_mutating_the_dataset(self):
        messages = [{"role": "system", "content": "无参数工具必须传严格的 `{}`。"},
                    {"role": "user", "content": "task"}]
        adapted = qwen_tool_messages(messages)
        self.assertIn("必须传严格的 `{}`", messages[0]["content"])
        self.assertNotIn("必须传严格的 `{}`", adapted[0]["content"])
        self.assertIn(QWEN_TOOL_FORMAT_PROMPT, adapted[0]["content"])
        self.assertEqual(qwen_tool_messages(adapted), adapted)
        self.assertEqual(adapted[1], messages[1])

    def test_missing_system_message_is_added_before_the_task(self):
        messages = [{"role": "user", "content": "task"}]
        adapted = qwen_tool_messages(messages)
        self.assertEqual(adapted[0]["role"], "system")
        self.assertEqual(adapted[1:], messages)
