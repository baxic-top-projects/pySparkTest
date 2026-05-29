"""Alpaca prompt helpers for local serve (без PySpark)."""
from __future__ import annotations

import os
import re

# Как в Stanford Alpaca / колонке `text` в Hive (короткий preamble).
ALPACA_PREAMBLE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
)

# System для safe chat (train DeepSeek + fallback serve).
DEEPSEEK_ALPACA_SYSTEM = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request. Answer helpfully; do not refuse "
    "general knowledge or lifestyle questions."
)

# Без дефолтного system «only computer science» (см. старый chat_template.jinja DeepSeek).
SAFE_DEEPSEEK_CHAT_TEMPLATE = (
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}"
    "{% endif %}{{ bos_token }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}{{ message['content'] }}\n\n"
    "{%- elif message['role'] == 'user' %}### Instruction:\n{{ message['content'] }}\n\n"
    "{%- elif message['role'] == 'assistant' %}### Response:\n{{ message['content'] }}\n"
    "<|EOT|>\n"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}### Response:\n\n{%- endif %}"
)

REFUSAL_PATTERNS = (
    r"i['\u2019]?m sorry",
    r"i cannot\b",
    r"i can['\u2019]t\b",
    r"as an ai\b",
    r"programming-related",
    r"only answer questions related to computer science",
    r"unable to provide advice",
    r"outside of my area",
)

_INSTRUCTION_RE = re.compile(
    r"### Instruction:\s*\n(.*?)(?=\n### Input:|\n### Response:|\Z)",
    re.DOTALL,
)
_INPUT_RE = re.compile(r"### Input:\s*\n(.*?)(?=\n### Response:|\Z)", re.DOTALL)


def is_deepseek_model(model_name: str) -> bool:
    name = (model_name or "").lower()
    return "deepseek" in name or "alpaca-deepseek" in name


def deepseek_serve_use_plain() -> bool:
    """По умолчанию plain Alpaca — стабильно после restart (disk chat_template может быть старым)."""
    return os.environ.get("DEEPSEEK_SERVE_PLAIN", "1").lower() not in (
        "0",
        "false",
        "no",
    )


def is_refusal_response(text: str) -> bool:
    lowered = (text or "").lower()
    return any(re.search(pat, lowered) for pat in REFUSAL_PATTERNS)


def parse_alpaca_prompt(prompt: str) -> tuple[str, str]:
    """Разобрать instruction/input из строки Alpaca (с preamble или без)."""
    body = prompt
    if "Below is an instruction" in prompt and "### Instruction:" in prompt:
        body = "### Instruction:" + prompt.split("### Instruction:", 1)[1]

    instruction = ""
    input_text = ""
    m_inst = _INSTRUCTION_RE.search(body)
    if m_inst:
        instruction = m_inst.group(1).strip()
    m_in = _INPUT_RE.search(body)
    if m_in:
        input_text = m_in.group(1).strip()
    if not instruction and "### Instruction:" not in body:
        instruction = prompt.strip()
    return instruction, input_text


def build_plain_alpaca_prompt(instruction: str, input_text: str = "") -> str:
    """Полная строка Alpaca для plain tokenize (как в датасете)."""
    parts = [ALPACA_PREAMBLE, f"### Instruction:\n{instruction.strip()}\n\n"]
    if input_text.strip():
        parts.extend([f"### Input:\n{input_text.strip()}\n\n"])
    parts.append("### Response:\n")
    return "".join(parts)


def is_complete_alpaca_prompt(prompt: str) -> bool:
    p = (prompt or "").strip()
    return "### Instruction:" in p and "### Response:" in p


def apply_safe_deepseek_chat_template(tokenizer) -> None:
    if hasattr(tokenizer, "chat_template"):
        tokenizer.chat_template = SAFE_DEEPSEEK_CHAT_TEMPLATE


def deepseek_alpaca_messages(instruction: str, input_text: str = "") -> list[dict[str, str]]:
    user = instruction.strip()
    if input_text.strip():
        user += f"\n\n### Input:\n{input_text.strip()}"
    return [
        {"role": "system", "content": DEEPSEEK_ALPACA_SYSTEM},
        {"role": "user", "content": user},
    ]


def resolve_plain_generation_prompt(prompt: str) -> str:
    """Промпт для plain tokenize: не переписывать уже полный Alpaca из датасета."""
    if is_complete_alpaca_prompt(prompt):
        text = prompt.strip()
        if not text.endswith("### Response:\n"):
            if "### Response:\n" in text:
                text = text.split("### Response:\n", 1)[0] + "### Response:\n"
            else:
                text = text + "\n### Response:\n"
        return text
    instruction, input_text = parse_alpaca_prompt(prompt)
    return build_plain_alpaca_prompt(instruction, input_text)


def tokenize_generation_prompt(tokenizer, prompt: str, *, base_model_name: str = ""):
    """Токенизация для generate; DeepSeek: plain Alpaca (default) или safe chat.

    Returns:
        (inputs, mode, encoded_prompt) — encoded_prompt для корректного decode.
    """
    is_ds = is_deepseek_model(base_model_name)
    use_plain = deepseek_serve_use_plain() if is_ds else False

    if is_ds and use_plain:
        plain = resolve_plain_generation_prompt(prompt)
        return tokenizer(plain, return_tensors="pt"), "plain", plain

    if is_ds and hasattr(tokenizer, "apply_chat_template"):
        apply_safe_deepseek_chat_template(tokenizer)
        instruction, input_text = parse_alpaca_prompt(prompt)
        messages = deepseek_alpaca_messages(instruction, input_text)
        enc = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        return enc, "chat", prompt

    plain = prompt
    if is_ds:
        plain = resolve_plain_generation_prompt(prompt)
    return tokenizer(plain, return_tensors="pt"), "plain", plain


def decode_new_tokens(
    tokenizer, output_ids, inputs, *, mode: str, encoded_prompt: str
) -> str:
    if mode == "chat":
        in_len = inputs["input_ids"].shape[-1]
        new_ids = output_ids[in_len:]
        return tokenizer.decode(new_ids, skip_special_tokens=True).lstrip()
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    plain = encoded_prompt
    if text.startswith(plain):
        return text[len(plain) :].lstrip()
    return text


def trim_alpaca_completion(text: str) -> str:
    for stop in ("\n\n###", "\n### Instruction", "\n### Input"):
        if stop in text:
            text = text.split(stop, 1)[0].rstrip()
    return text
