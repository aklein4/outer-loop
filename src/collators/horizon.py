from transformers import AutoTokenizer


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


class HorizonCollator:

    def __init__(
        self,
        tokenizer_url: str,
        max_length: int,
        cluster_length: int,
        episode_prefix: str = "episode_",
        chat_template: str = ASSISTANT_MASK_CHAT_TEMPLATE,
    ):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_url)
        self.tokenizer.padding_side = "right"
        self.tokenizer.chat_template = chat_template
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.max_length = max_length
        self.cluster_length = cluster_length
        self.episode_keys = [
            f"{episode_prefix}{i + 1}"
            for i in range(cluster_length)
        ]

        self.step = 0
        self.skip_steps = None


    def __call__(
        self,
        raw
    ):
        if self.skip_steps is not None and self.step < self.skip_steps:
            self.step += 1
            return {}
        
        missing = [
            key
            for key in self.episode_keys
            if key not in raw[0]
        ]
        if missing:
            raise ValueError(f"HorizonCollator requires episode columns, missing {missing}")

        messages = []
        for row in raw:
            for key in self.episode_keys:
                messages.append(row[key])

        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_dict=True,
            return_assistant_tokens_mask=True,
            return_attention_mask=True,
        )

        assistant_mask = encoded.pop("assistant_masks", None)
        if assistant_mask is None:
            assistant_mask = encoded.pop("assistant_tokens_mask", None)
        if assistant_mask is None:
            raise RuntimeError("tokenizer did not return assistant token masks")

        input_ids = encoded["input_ids"].long()
        assistant_mask = assistant_mask.bool()
        attention_mask = encoded["attention_mask"].bool()

        batch_size = len(raw)
        input_ids = input_ids.reshape(batch_size, self.cluster_length, self.max_length)
        assistant_mask = assistant_mask.reshape(batch_size, self.cluster_length, self.max_length)
        attention_mask = attention_mask.reshape(batch_size, self.cluster_length, self.max_length)

        out = {
            "input_ids": input_ids,
            "assistant_mask": assistant_mask,
            "attention_mask": attention_mask,
        }

        return out
