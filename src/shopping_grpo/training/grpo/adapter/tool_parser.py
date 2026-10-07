"""Parse complete Qwen tool calls without accepting truncated actions."""

from __future__ import annotations

import json
import re

from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS


class ToolCallParseError(ValueError):
    """The actor emitted an incomplete or invalid tool call."""

    def __init__(self, reason, *, tool_name=None):
        super().__init__(reason)
        self.tool_name = tool_name


def parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Validate the entire call before any environment action can execute.

    Shop tool schema v2 has only string parameters. Preserve their literal
    values, including ampersands, angle brackets and the string ``null``.
    """
    if not any(marker in text for marker in ("<tool_call", "<function=", "<parameter=")):
        return []
    start = text.find("<tool_call>")
    if start < 0:
        raise ToolCallParseError("missing_tool_call_wrapper")
    body = text[start:]
    schemas = {s["function"]["name"]: s["function"]["parameters"] for s in SHOP_TOOL_SCHEMAS}
    calls = []
    while body.startswith("<tool_call>"):
        match = re.match(r"<tool_call>\s*<function=([A-Za-z_][A-Za-z_0-9]*)>(.*?)</function>\s*</tool_call>", body, re.S)
        if match is None:
            raise ToolCallParseError("incomplete_tool_call")
        name, parameters = match.groups()
        if name not in schemas:
            raise ToolCallParseError("unknown_tool")
        arguments = {}
        parameters = parameters.strip()
        if not schemas[name]["properties"] and parameters:
            raise ToolCallParseError("unexpected_parameter_for_noarg_tool", tool_name=name)
        while parameters:
            param = re.match(r"<parameter=([A-Za-z_][A-Za-z_0-9]*)>(.*?)</parameter>", parameters, re.S)
            if param is None:
                raise ToolCallParseError("incomplete_parameter")
            key, value = param.groups()
            if key in arguments or key not in schemas[name]["properties"]:
                raise ToolCallParseError("duplicate_or_unknown_parameter")
            if any(tag in value for tag in ("<parameter=", "<function=", "<tool_call>")):
                raise ToolCallParseError("nested_tool_markup")
            value = value.removeprefix("\n").removesuffix("\n")
            allowed = schemas[name]["properties"][key].get("enum")
            if allowed is not None and value not in allowed:
                raise ToolCallParseError("invalid_parameter_enum")
            arguments[key] = value
            parameters = parameters[param.end():].strip()
        if set(schemas[name]["required"]) - arguments.keys():
            raise ToolCallParseError("missing_required_parameter")
        calls.append((name, arguments))
        body = body[match.end():].strip()
    if body not in {"", "<|im_end|>", "<|endoftext|>"}:
        raise ToolCallParseError("unexpected_tool_call_suffix")
    return calls


class ShoppingToolParser:
    """veRL parser interface, scoped to the repository's Qwen tool protocol."""

    def __init__(self, tokenizer, delegate):
        self.tokenizer = tokenizer
        self.stop_token_ids = delegate.stop_token_ids

    async def extract_tool_calls(self, responses_ids, tools=None):
        from verl.experimental.agent_loop.tool_parser import FunctionCall

        text = self.tokenizer.decode(responses_ids)
        calls = parse_tool_calls(text)
        return text, [FunctionCall(name=name, arguments=json.dumps(args, ensure_ascii=False))
                      for name, args in calls]
