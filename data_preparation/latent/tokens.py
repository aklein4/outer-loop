

from collections.abc import Callable
from typing import Any

import torch
from gigatoken import Tokenizer as GigaTokenizer
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from transformers import AutoTokenizer
from transformers.utils.chat_template_utils import render_jinja_template


# meta-llama/Llama-3.2-1B-Instruct
ASSISTANT_MASK_CHAT_TEMPLATE = r"""{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "26 Jul 2024" %}
    {%- endif %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
        {%- if message['role'] == 'assistant' %}
            {%- generation %}{{- message['content'] | trim + '<|eot_id|>' }}{%- endgeneration %}
        {%- else %}
            {{- message['content'] | trim + '<|eot_id|>' }}
        {%- endif %}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
        {%- generation %}
            {{- '{"name": "' + tool_call.name + '", ' }}
            {{- '"parameters": ' }}
            {{- tool_call.arguments | tojson }}
            {{- "}" }}
            {{- "<|eot_id|>" }}
        {%- endgeneration %}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""


class GigaChat:
    """Render Hugging Face chat templates and tokenize them with Gigatoken."""

    def __init__(
        self,
        tokenizer_url: str,
        max_length: int | None = None,
        chat_template: str = ASSISTANT_MASK_CHAT_TEMPLATE,
        tokenizer_path: str | None = None,
    ) -> None:
        self.tokenizer_url = tokenizer_url
        self.max_length = max_length
        self.chat_template = chat_template
        if tokenizer_path is None:
            try:
                tokenizer_path = snapshot_download(
                    tokenizer_url,
                    local_files_only=True,
                )
            except LocalEntryNotFoundError:
                # Resolve/download once in the parent. Pickled worker copies
                # receive this local path and never repeat Hub API requests.
                tokenizer_path = snapshot_download(tokenizer_url)
        self.tokenizer_path = tokenizer_path
        self.hf_tokenizer = None
        self.gigatokenizer = None
        self._giga_hf = None

    def _initialize(self) -> None:
        if self.gigatokenizer is not None:
            return

        self.hf_tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            local_files_only=True,
        )
        self.hf_tokenizer.padding_side = "right"
        self.hf_tokenizer.chat_template = self.chat_template
        if self.hf_tokenizer.pad_token_id is None:
            self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token

        self.gigatokenizer = GigaTokenizer(self.hf_tokenizer)
        self._giga_hf = self.gigatokenizer.as_hf()
        self._giga_hf.padding_side = self.hf_tokenizer.padding_side
        self._giga_hf.pad_token = self.hf_tokenizer.pad_token

    def __getstate__(self) -> dict[str, Any]:
        # The native Gigatoken BPETokenizer cannot be pickled. Persist only the
        # constructor inputs and rebuild local tokenizer state in each worker.
        return {
            "tokenizer_url": self.tokenizer_url,
            "max_length": self.max_length,
            "chat_template": self.chat_template,
            "tokenizer_path": self.tokenizer_path,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(**state)

    @staticmethod
    def _as_batch(messages: list[dict[str, Any]] | list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
        if not isinstance(messages, (list, tuple)) or not messages:
            raise ValueError("messages must be a non-empty conversation or batch of conversations")
        if isinstance(messages[0], dict):
            return [messages]
        return list(messages)

    def __call__(
        self,
        messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
        *,
        tools: list[dict[str, Any] | Callable] | None = None,
        max_length: int | None = None,
        padding: bool | str | None = None,
        truncation: bool | None = None,
        add_generation_prompt: bool = False,
        documents: list[dict[str, str]] | None = None,
        **template_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Return right-padded ``input_ids`` and boolean assistant/attention masks.

        ``messages`` accepts either one conversation or a batch. ``tools`` may
        contain Hugging Face-style JSON schemas or typed Python callables.
        Additional keyword arguments are made available to the Jinja template.
        """
        self._initialize()
        conversations = self._as_batch(messages)
        effective_max_length = max_length if max_length is not None else self.max_length
        if padding is None:
            padding = "max_length" if effective_max_length is not None else "longest"
        if truncation is None:
            truncation = effective_max_length is not None
        if padding == "max_length" and effective_max_length is None:
            raise ValueError('padding="max_length" requires max_length')
        if truncation and effective_max_length is None:
            raise ValueError("truncation requires max_length")
        encoding_max_length = (
            effective_max_length
            if truncation or padding == "max_length"
            else None
        )

        render_kwargs = {**self.hf_tokenizer.special_tokens_map, **template_kwargs}
        rendered, assistant_spans = render_jinja_template(
            conversations=conversations,
            tools=tools,
            documents=documents,
            chat_template=self.chat_template,
            return_assistant_tokens_mask=True,
            continue_final_message=False,
            add_generation_prompt=add_generation_prompt,
            **render_kwargs,
        )

        encoded = self._giga_hf(
            rendered,
            add_special_tokens=False,
            padding=padding,
            truncation=truncation,
            max_length=encoding_max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = encoded["input_ids"].long()
        attention_mask = encoded["attention_mask"].bool()
        assistant_mask = torch.zeros_like(attention_mask)

        # The template renderer reports assistant spans as character offsets.
        # Encoding every span prefix maps those offsets to the identical token
        # boundaries without requiring Gigatoken to expose offset mappings.
        prefixes: list[str] = []
        locations: list[tuple[int, bool]] = []
        for row, (text, spans) in enumerate(zip(rendered, assistant_spans)):
            for start, end in spans:
                prefixes.extend((text[:start], text[:end]))
                locations.extend(((row, True), (row, False)))

        if prefixes:
            prefix_ids = self.gigatokenizer.encode_batch(prefixes)
            boundaries: dict[int, list[list[int]]] = {}
            for ids, (row, is_start) in zip(prefix_ids, locations):
                row_boundaries = boundaries.setdefault(row, [[], []])
                row_boundaries[0 if is_start else 1].append(len(ids))
            width = input_ids.shape[1]
            for row, (starts, ends) in boundaries.items():
                for start, end in zip(starts, ends):
                    if start >= width:
                        break
                    assistant_mask[row, start : min(end, width)] = True

        return {
            "input_ids": input_ids,
            "assistant_mask": assistant_mask,
            "attention_mask": attention_mask,
        }
