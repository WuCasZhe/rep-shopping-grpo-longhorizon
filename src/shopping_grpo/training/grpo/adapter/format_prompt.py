"""Qwen XML formatting instructions at the GRPO rendering boundary."""

from copy import deepcopy

from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS


NO_ARGUMENT_TOOLS = tuple(
    schema["function"]["name"] for schema in SHOP_TOOL_SCHEMAS
    if not schema["function"]["parameters"]["properties"]
)

QWEN_TOOL_FORMAT_PROMPT = """工具调用输出格式（Qwen XML）：
每回合只输出一个完整的 tool_call，function 名必须来自工具列表。
无参数表示 function 内没有任何 parameter 标签，也不写空对象或占位符。
以下工具均无参数：{names}。
例如调用 prev_page 的完整输出是：
<tool_call>
<function=prev_page>
</function>
</tool_call>
有参数工具才写具名 parameter。例如：
<tool_call>
<function=select_option>
<parameter=value>蓝色</parameter>
</function>
</tool_call>
以上只是格式示例；实际工具和参数必须依据最新 observation 选择。""".format(
    names=", ".join(NO_ARGUMENT_TOOLS)
)


def qwen_tool_messages(messages):
    """Adapt cached prompts without rewriting the training dataset."""
    result = deepcopy(list(messages))
    for message in result:
        if message.get("role") == "system" and isinstance(message.get("content"), str):
            content = message["content"].replace(
                "必须传严格的 `{}`", "不得提供任何参数"
            )
            if QWEN_TOOL_FORMAT_PROMPT not in content:
                content += "\n\n" + QWEN_TOOL_FORMAT_PROMPT
            message["content"] = content
            return result
    result.insert(0, {"role": "system", "content": QWEN_TOOL_FORMAT_PROMPT})
    return result


def parse_error_feedback(error):
    name = getattr(error, "tool_name", None)
    if name in NO_ARGUMENT_TOOLS:
        return (
            f"工具调用未执行：{name} 是无参数工具。删除整个 parameter 段，"
            "不要把空对象写成参数。若仍要执行该动作，请按以下完整格式重试：\n"
            f"<tool_call>\n<function={name}>\n</function>\n</tool_call>"
        )
    return "工具调用格式错误，未执行任何动作。请按以下规则重新生成：\n" + QWEN_TOOL_FORMAT_PROMPT
