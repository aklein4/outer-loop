
import ast
import gzip
import hashlib
import html
import io
import json
import re
import zipfile
from pathlib import Path

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import semchunk
from huggingface_hub import hf_hub_download, hf_hub_url, list_repo_files

from base_handler import BaseHandler
from utils import html_to_markdown, simple_format


_HANDLER_REGISTRY: dict[str, type[BaseHandler]] = {}
_FIRST_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _is_english_language(value):
    language = (value or "").strip().lower().replace("_", "-")
    return language == "english" or language == "en" or language.startswith("en-")


def _safe_html_to_markdown(content):
    """Convert HTML while tolerating pathologically deep markup trees."""
    try:
        return html_to_markdown(content)
    except RecursionError:
        # Some feed descriptions contain thousands of nested tags. Beautiful
        # Soup/markdownify recursively walks that tree and can exceed Python's
        # recursion limit. This fallback is deliberately non-recursive: retain
        # readable text and line boundaries without hiding other exceptions.
        text = re.sub(r"(?i)<br\s*/?>|</p\s*>|</div\s*>|</li\s*>", "\n", content or "")
        text = re.sub(r"<[^>]*>", "", text)
        text = html.unescape(text)
        return "\n".join(
            line.strip() for line in text.splitlines() if line.strip()
        )


def _split_lead(text):
    """Split prose after its first paragraph, or first sentence as fallback."""
    text = (text or "").strip()
    paragraphs = re.split(r"\n\s*\n", text, maxsplit=1)
    if len(paragraphs) == 2:
        return paragraphs[0].strip(), paragraphs[1].strip()

    boundary = _FIRST_SENTENCE_END.search(text)
    if boundary is None:
        return "", ""
    return text[:boundary.start()].strip(), text[boundary.end():].strip()


def _parse_jsonish(value):
    """Parse JSON and the Python-literal tool calls used by Toucan SFT."""
    if not isinstance(value, str):
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value)
        except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            pass
    return None


def _json_compatible(value):
    """Normalize Python-literal values into deterministic JSON values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_json_compatible(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ),
        )
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _canonical_tools_hash(tools):
    if isinstance(tools, str):
        tools = _parse_jsonish(tools)
    if not isinstance(tools, list) or not tools:
        return None

    canonical_tools = sorted(
        json.dumps(
            _json_compatible(tool),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        for tool in tools
    )
    canonical = "[" + ",".join(canonical_tools) + "]"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _message_content(value):
    """Represent structured tool results as deterministic message text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(
        _json_compatible(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _normalize_tool_call(raw_call):
    """Convert legacy and OpenAI-style calls to the output tool schema."""
    raw_call = _parse_jsonish(raw_call)
    if not isinstance(raw_call, dict):
        return None

    function = _parse_jsonish(raw_call.get("function"))
    if not isinstance(function, dict):
        function = raw_call
    name = function.get("name")
    if not name:
        return None

    arguments = _parse_jsonish(function.get("arguments", {}))
    if not isinstance(arguments, dict):
        return None
    return {
        "type": "function",
        "function": {
            "name": str(name),
            # OpenAI/Llama tool-call interchange represents arguments as a
            # JSON-encoded string. This also prevents Arrow from trying to
            # unify unrelated argument structs from different tools.
            "arguments": json.dumps(
                _json_compatible(arguments),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        },
    }


def _toucan_tool_messages(raw_messages, available_tools):
    """Convert Toucan provider formats without changing legacy behavior."""
    messages = _parse_jsonish(raw_messages)
    tools = _parse_jsonish(available_tools)
    if not isinstance(messages, list) or not isinstance(tools, list) or not tools:
        return None

    tool_instructions = (
        "You have access to the following functions. To call a function, "
        "respond with a function call using its name and arguments.\n\n"
        + "\n".join(
            json.dumps(
                _json_compatible(tool), sort_keys=True, ensure_ascii=False
            )
            for tool in tools
        )
    )
    converted = [{"role": "system", "content": tool_instructions}]

    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")

        if role == "system":
            continue

        function_call = message.get("function_call")
        if function_call is None and role == "tool_call":
            function_call = _parse_jsonish(content)
        if function_call is not None:
            function_call = _parse_jsonish(function_call)
            if not isinstance(function_call, dict) or not function_call.get("name"):
                return None
            arguments = _parse_jsonish(function_call.get("arguments", {}))
            if not isinstance(arguments, dict):
                return None
            converted.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": function_call["name"],
                        "arguments": json.dumps(
                            _json_compatible(arguments),
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ),
                    },
                }],
            })
            continue

        if role in {"function", "tool_response", "tool"}:
            converted.append({
                "role": "tool",
                "content": "" if content is None else str(content),
            })
            continue

        if role not in {"user", "assistant"}:
            return None
        if content is None or not str(content).strip():
            continue
        converted.append({"role": role, "content": str(content)})

    return converted


def _interleave_parallel_tool_calls(messages):
    """Pair parallel assistant calls with their following tool responses."""
    interleaved = []
    message_index = 0
    while message_index < len(messages):
        message = messages[message_index]
        if not isinstance(message, dict):
            interleaved.append(message)
            message_index += 1
            continue

        tool_calls = _parse_jsonish(message.get("tool_calls"))
        if not isinstance(tool_calls, list) or len(tool_calls) <= 1:
            interleaved.append(message)
            message_index += 1
            continue

        response_end = message_index + 1
        responses = []
        while response_end < len(messages):
            response = messages[response_end]
            if not isinstance(response, dict) or response.get("role") not in {
                "function", "tool_response", "tool",
            }:
                break
            responses.append(response)
            response_end += 1

        unused_response_indices = set(range(len(responses)))
        for call_index, tool_call in enumerate(tool_calls):
            split_message = dict(message)
            split_message["content"] = (
                message.get("content") if call_index == 0 else ""
            )
            split_message["tool_calls"] = [tool_call]
            interleaved.append(split_message)

            call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
            response_index = next((
                index
                for index in unused_response_indices
                if call_id is not None
                and responses[index].get("tool_call_id") == call_id
            ), None)
            if response_index is None and unused_response_indices:
                response_index = min(unused_response_indices)
            if response_index is not None:
                interleaved.append(responses[response_index])
                unused_response_indices.remove(response_index)

        # Preserve surplus responses instead of silently dropping malformed
        # source data when the number of results does not match the calls.
        interleaved.extend(
            responses[index] for index in sorted(unused_response_indices)
        )
        message_index = response_end

    return interleaved


def _llama_tool_messages(
    raw_messages,
    available_tools,
    preserve_system=False,
):
    """Convert provider tool formats to Llama function-call messages."""
    messages = _parse_jsonish(raw_messages)
    tools = _parse_jsonish(available_tools)
    if not isinstance(messages, list) or not isinstance(tools, list) or not tools:
        return None
    messages = _interleave_parallel_tool_calls(messages)

    tool_instructions = (
        "You have access to the following functions. To call a function, "
        "respond with a function call using its name and arguments.\n\n"
        + "\n".join(
            json.dumps(
                _json_compatible(tool), sort_keys=True, ensure_ascii=False
            )
            for tool in tools
        )
    )
    converted = []
    system_contents = []

    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")

        # Original system prompts contain model-specific tool syntax. The
        # canonical Llama-compatible declaration above replaces them.
        if role == "system":
            if preserve_system and content is not None and str(content).strip():
                system_contents.append(str(content).strip())
            continue

        function_call = message.get("function_call")
        if function_call is None and role == "tool_call":
            function_call = _parse_jsonish(content)
        if function_call is not None:
            function_call = _normalize_tool_call(function_call)
            if function_call is None:
                return None
            converted.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [function_call],
            })
            continue

        tool_calls = _parse_jsonish(message.get("tool_calls"))
        if tool_calls:
            if not isinstance(tool_calls, list):
                return None
            normalized_calls = [_normalize_tool_call(call) for call in tool_calls]
            if any(call is None for call in normalized_calls):
                return None
            # The interleaving pass normally leaves exactly one call here.
            # Retain this loop as a schema-safe fallback for malformed rows
            # where calls could not be expanded before conversion.
            for call_index, normalized_call in enumerate(normalized_calls):
                converted.append({
                    "role": "assistant",
                    "content": (
                        _message_content(content) if call_index == 0 else ""
                    ),
                    "tool_calls": [normalized_call],
                })
            continue

        if role in {"function", "tool_response", "tool"}:
            converted.append({
                "role": "tool",
                "content": _message_content(content),
            })
            continue

        if role not in {"user", "assistant"}:
            return None
        if content is None or not str(content).strip():
            # Some OSS rows contain empty assistant messages whose useful
            # payload is provider-only hidden reasoning.
            continue
        converted.append({"role": role, "content": _message_content(content)})

    system_content = tool_instructions
    if system_contents:
        system_content += "\n\n" + "\n\n".join(system_contents)
    return [{"role": "system", "content": system_content}, *converted]


def _render_quora_span(span):
    modifiers = span.get("modifiers") or {}
    text = span.get("text") or ""

    image_url = modifiers.get("image")
    if image_url:
        return f"![]({image_url})"

    embed = modifiers.get("embed") or {}
    if embed.get("url"):
        title = embed.get("title") or embed["url"]
        return f"[{title}]({embed['url']})"

    citation = modifiers.get("citation") or {}
    citation_target = citation.get("target") or {}
    if citation_target.get("url") and not text.strip():
        title = citation.get("title") or citation_target["url"]
        return f"[{title}]({citation_target['url']})"

    if modifiers.get("math") and text:
        text = f"${text}$"
    if modifiers.get("bold") and text:
        text = f"**{text}**"
    if modifiers.get("italic") and text:
        text = f"*{text}*"

    link = modifiers.get("link") or {}
    if link.get("url") and text:
        text = f"[{text}]({link['url']})"
    return text


def _render_quora_content(content):
    if not content:
        return ""
    try:
        document = json.loads(content) if isinstance(content, str) else content
    except (TypeError, json.JSONDecodeError):
        return ""

    blocks = []
    ordered_index = 0
    previous_type = None
    for section in document.get("sections", []):
        section_type = section.get("type", "plain")
        text = "".join(_render_quora_span(span) for span in section.get("spans", []))
        text = text.strip()

        if section_type == "horizontal-rule":
            block = "---"
        elif section_type == "ordered-list":
            ordered_index = ordered_index + 1 if previous_type == section_type else 1
            block = f"{ordered_index}. {text}"
        elif section_type == "unordered-list":
            block = f"- {text}"
        elif section_type == "code":
            block = f"```\n{text}\n```"
        else:
            ordered_index = 0
            block = text

        if section.get("quoted") and block:
            block = "\n".join(f"> {line}" for line in block.splitlines())
        if block:
            blocks.append(block)
        previous_type = section_type

    return "\n\n".join(blocks).strip()


def register_handler(handler_type: type[BaseHandler]) -> type[BaseHandler]:
    """Register a concrete, zero-argument handler by its source name."""
    if not issubclass(handler_type, BaseHandler):
        raise TypeError(f"{handler_type!r} must inherit from BaseHandler.")

    try:
        handler = handler_type()
    except TypeError as error:
        raise TypeError(
            f"Registered handler {handler_type.__name__} must be constructible "
            "without arguments."
        ) from error

    source = handler.source()
    if not isinstance(source, str) or not source.strip():
        raise ValueError(
            f"Registered handler {handler_type.__name__} must define a non-empty source."
        )

    if source in _HANDLER_REGISTRY:
        registered_type = _HANDLER_REGISTRY[source]
        raise ValueError(
            f"Duplicate handler source {source!r}: "
            f"{registered_type.__name__} and {handler_type.__name__}."
        )

    _HANDLER_REGISTRY[source] = handler_type
    return handler_type


def get_handlers(names: list[str] | str | None = None) -> tuple[type[BaseHandler], ...]:
    """Return registered handlers, optionally selecting sources in the given order."""
    if names is None:
        return tuple(_HANDLER_REGISTRY.values())

    names = (names,) if isinstance(names, str) else tuple(names)
    missing = [name for name in names if name not in _HANDLER_REGISTRY]
    if missing:
        raise ValueError(f"Dataset names not found in handlers: {missing}")

    return tuple(_HANDLER_REGISTRY[name] for name in names)


"""
All handlers must define:

def map_fn(self, example):
    ...
    return conversation, latent, keep

Handlers that emit multiple rows may instead return parallel lists:

    return conversations, latents, keeps
"""


@register_handler
class FandomHandler(BaseHandler):

    url = "RyokoAI/Fandom23K"
    subset = None
    split = ["train"]

    kind = "wiki"

    def map_fn(self, example):

        latent = example["tag"]
        if latent.startswith("fandom."):
            latent = latent[len("fandom."):].strip()

        text = example["text"]
        title = example["title"]
        check = f"# {title.lower()}"
        if text.lower().startswith(check):
            text = text[len(check):].strip()

        conversation = simple_format(title, text)
        return conversation, latent, True


@register_handler
class WebFaqHandler(BaseHandler): 

    url = "PaDaS-Lab/webfaq-v2"
    subset = "eng"
    split = "default"

    kind = "qa"

    def map_fn(self, example):
        conversation = simple_format(example["question"], example["answer"])
        return conversation, example["origin"], True


@register_handler
class BlogAuthorshipHandler(BaseHandler):

    url = "barilan/blog_authorship_corpus"
    subset = None
    split = None

    kind = "blog"

    @staticmethod
    def _generate_examples(archive_path, max_count):
        count = 0
        with zipfile.ZipFile(archive_path) as archive:
            paths = sorted(
                path for path in archive.namelist()
                if path.startswith("blogs/") and path.endswith(".xml")
            )
            for path in paths:
                author = path.rsplit("/", maxsplit=1)[-1].split(".", maxsplit=1)[0]
                date = ""
                with archive.open(path) as raw_file:
                    for raw_line in raw_file:
                        line = raw_line.decode("latin_1").strip()
                        if "<date>" in line:
                            date = line.split("<date>", maxsplit=1)[-1].split(
                                "</date>", maxsplit=1
                            )[0]
                        elif line and not line.startswith("<"):
                            yield {"author": author, "date": date, "text": line}
                            count += 1
                            if max_count is not None and count >= max_count:
                                return

    def load_dataset(self, max_count=None):
        archive_path = hf_hub_download(
            self.url,
            "data/blogs.zip",
            repo_type="dataset",
        )
        features = datasets.Features({
            "author": datasets.Value("string"),
            "date": datasets.Value("string"),
            "text": datasets.Value("string"),
        })
        return datasets.Dataset.from_generator(
            self._generate_examples,
            gen_kwargs={"archive_path": archive_path, "max_count": max_count},
            features=features,
            cache_dir=self.generator_cache_dir(),
        )

    def map_fn(self, example):
        text = example["text"].strip()
        boundary = _FIRST_SENTENCE_END.search(text)
        if boundary is None:
            return None, None, False

        prompt = text[:boundary.start()].strip()
        response = text[boundary.end():].strip()
        keep = bool(prompt and response and example["author"])
        return simple_format(prompt, response), example["author"], keep


@register_handler
class EnronEmailsHandler(BaseHandler):

    url = "corbt/enron-emails"
    subset = None
    split = "train"

    kind = "email"

    def map_fn(self, example):
        headers = [
            f"To: {', '.join(address for address in example['to'] if address)}",
            f"Cc: {', '.join(address for address in example['cc'] if address)}",
            f"Bcc: {', '.join(address for address in example['bcc'] if address)}",
            f"Subject: {example['subject']}",
        ]
        conversation = simple_format("\n".join(headers), example["body"])
        return conversation, example["from"], True


@register_handler
class CodeSearchNetHandler(BaseHandler):

    url = "code-search-net/code_search_net"
    subset = "all"
    split = ["train", "validation", "test"]

    kind = "code"

    def map_fn(self, example):
        conversation = simple_format(
            example["func_documentation_string"],
            example["func_code_string"],
        )
        return conversation, example["repository_name"], True


@register_handler
class StackExchangeHandler(BaseHandler):

    url = "HuggingFaceTB/stackexchange_2025_md"
    subset = None
    split = "train"

    kind = "qa"

    # The full 31M-row source peaked above 400 GiB when mapped as one table.
    # Keep the caller's configured parallelism inside bounded source slices.
    max_num_proc = 16
    source_batch_size = 1_000_000

    def _config_names(self):
        configs = datasets.get_dataset_config_names(self.url)
        if configs != ["default"]:
            return configs

        # During Hub rate limiting, ``datasets`` can silently return the
        # fallback config ``default`` even though this dataset has only named
        # community configs. Recover already-discovered configs for a safe
        # offline/resume path.
        cache_root = (
            Path(datasets.config.HF_DATASETS_CACHE)
            / self.url.replace("/", "___")
        )
        cached_configs = sorted(
            path.name for path in cache_root.iterdir() if path.is_dir()
        ) if cache_root.is_dir() else []
        return cached_configs or configs

    def process(
        self,
        tokenizer=None,
        max_count=None,
        num_proc=1,
        batch_size=1000,
        intermediate_cache_dir=None,
    ):
        # Mapping one 31M-row concatenation causes each worker to fault large
        # portions of every source table into memory. Process configurations
        # in bounded contiguous slices, then concatenate the uniform mapped
        # outputs (messages/latent/source/kind) in their original order.
        num_proc = min(num_proc, self.max_num_proc)
        parts = []
        remaining = max_count
        for subset in self._config_names():
            if remaining == 0:
                break

            ds = self._load_dataset_part(subset, self.split, remaining)
            source_count = len(ds)
            if source_count == 0:
                continue

            for source_start in range(0, source_count, self.source_batch_size):
                source_end = min(
                    source_start + self.source_batch_size,
                    source_count,
                )
                source_part = (
                    ds
                    if source_start == 0 and source_end == source_count
                    else ds.select(range(source_start, source_end))
                )
                source_part = source_part.add_column(
                    "subset",
                    [subset] * len(source_part),
                )
                mapped = self.process_dataset(
                    source_part,
                    tokenizer=tokenizer,
                    num_proc=num_proc,
                    batch_size=batch_size,
                    # Each source slice has a stable fingerprint. This makes
                    # an interrupted run resumable without redoing every
                    # already completed map and tokenizer filter.
                    load_from_cache_file=True,
                    intermediate_cache_dir=intermediate_cache_dir,
                )
                if len(mapped):
                    parts.append(mapped)

            if remaining is not None:
                remaining -= source_count

        if not parts:
            return datasets.Dataset.from_dict({
                "messages": [],
                "latent": [],
                "source": [],
                "kind": [],
            })
        return datasets.concatenate_datasets(parts)

    def load_dataset(self, max_count=None):
        parts = []
        remaining = max_count
        for subset in self._config_names():
            if remaining == 0:
                break
            ds = self._load_dataset_part(subset, self.split, remaining)
            if len(ds) == 0:
                continue
            parts.append(ds.add_column("subset", [subset] * len(ds)))
            if remaining is not None:
                remaining -= len(ds)

        common_columns = set.intersection(*[set(ds.column_names) for ds in parts])
        parts = [
            ds.remove_columns([
                column for column in ds.column_names
                if column not in common_columns
            ])
            for ds in parts
        ]
        return datasets.concatenate_datasets(parts)

    def map_fn(self, example):
        title = (example["Title"] or "").strip()
        question_body = html_to_markdown(example["Body"])
        question = f"# {title}\n\n{question_body}".strip()

        conversations = []
        latents = []
        keeps = []
        for answer in example["Answers"]:
            author_id = answer.get("OwnerUserId")
            answer_body = html_to_markdown(answer.get("Body"))
            if not question or not answer_body or not author_id or author_id < 0:
                continue

            conversations.append(simple_format(question, answer_body))
            latents.append(f"{example['subset']}:{author_id}")
            keeps.append(True)

        return conversations, latents, keeps


@register_handler
class QuoraQaHandler(BaseHandler):

    url = "LxYxvv/quora_qa_raw"
    subset = None
    split = "train"

    kind = "qa"

    def map_fn(self, example):
        question = _render_quora_content(example["title"])
        try:
            answers = json.loads(example["answers"])
        except (TypeError, json.JSONDecodeError):
            return [], [], []

        conversations = []
        latents = []
        keeps = []
        for answer in answers:
            author = answer.get("author") or {}
            author_id = author.get("uid")
            answer_body = _render_quora_content(answer.get("content"))
            if not question or not answer_body or author_id is None:
                continue

            conversations.append(simple_format(question, answer_body))
            latents.append(author_id)
            keeps.append(True)

        return conversations, latents, keeps


@register_handler
class SummScreenHandler(BaseHandler):

    url = "YuanPJ/summ_screen"
    subset = "all"
    split = ["train", "validation", "test"]

    kind = "screenplay"

    @staticmethod
    def _generate_examples(paths, max_count):
        count = 0
        for path in paths:
            with open(path, encoding="utf-8") as source:
                for example in json.load(source):
                    yield {
                        "recap": "\n".join(example.get("Recap") or []),
                        "transcript": "\n".join(example.get("Transcript") or []),
                        "show_title": example.get("Show Title") or "",
                    }
                    count += 1
                    if max_count is not None and count >= max_count:
                        return

    def load_dataset(self, max_count=None):
        paths = [
            hf_hub_download(
                self.url,
                f"data/{subset}_{split}.json",
                repo_type="dataset",
            )
            for subset in ("fd", "tms")
            for split in ("train", "dev", "test")
        ]
        features = datasets.Features({
            "recap": datasets.Value("string"),
            "transcript": datasets.Value("string"),
            "show_title": datasets.Value("string"),
        })
        return datasets.Dataset.from_generator(
            self._generate_examples,
            gen_kwargs={"paths": paths, "max_count": max_count},
            features=features,
            cache_dir=self.generator_cache_dir(),
        )

    def map_fn(self, example):
        recap = example["recap"].strip()
        transcript = example["transcript"].strip()
        show_title = example["show_title"].strip()
        keep = bool(recap and transcript and show_title)
        return simple_format(recap, transcript), show_title, keep


@register_handler
class Ao3Handler(BaseHandler):

    url = "ray0rf1re/AO3-2020"
    subset = "full"
    split = "train"

    kind = "fiction"

    def map_fn(self, example):
        prompt, response = _split_lead(example["text"])
        story_id = example["storyId"]
        keep = bool(prompt and response and story_id is not None)
        return simple_format(prompt, response), story_id, keep


@register_handler
class StarcoderIssuesHandler(BaseHandler):

    url = "bigcode/starcoder2data-extras"
    subset = "issues"
    split = "train"

    kind = "code"

    def map_fn(self, example):
        content = (example["content"] or "").strip()
        content = re.sub(r"^<issue_start>\s*", "", content)
        first_line, separator, rest = content.partition("\n")
        title = re.sub(r"^Title:\s*", "", first_line).strip()
        repo_name = (example["repo_name"] or "").strip()
        keep = bool(separator and title and rest.strip() and repo_name)
        return simple_format(title, rest.strip()), repo_name, keep


@register_handler
class HiCupidEvaluationHandler(BaseHandler):

    url = "arranonymsub/HiCUPID"
    subset = "evaluation"
    split = ["test_1", "test_2"]

    kind = "qa"

    def map_fn(self, example):
        question = (example["question"] or "").strip()
        answer = (example["personalized_answer"] or "").strip()
        user_id = example["user_id"]
        keep = bool(question and answer and user_id is not None)
        return simple_format(question, answer), user_id, keep


@register_handler
class Tldr17Handler(BaseHandler):

    url = "webis/tldr-17"
    subset = None
    split = "train"

    kind = "summary"

    @staticmethod
    def _generate_examples(archive_path, max_count):
        count = 0
        with zipfile.ZipFile(archive_path) as archive:
            member = next(
                name for name in archive.namelist()
                if name.endswith("corpus-webis-tldr-17.json")
            )
            with archive.open(member) as source:
                for line in source:
                    example = json.loads(line)
                    if "summary" not in example or "content" not in example:
                        continue
                    yield {
                        "summary": example.get("summary") or "",
                        "content": example.get("content") or "",
                        "author": example.get("author") or "",
                    }
                    count += 1
                    if max_count is not None and count >= max_count:
                        return

    def load_dataset(self, max_count=None):
        archive_path = hf_hub_download(
            self.url,
            "data/corpus-webis-tldr-17.zip",
            repo_type="dataset",
        )
        features = datasets.Features({
            "summary": datasets.Value("string"),
            "content": datasets.Value("string"),
            "author": datasets.Value("string"),
        })
        return datasets.Dataset.from_generator(
            self._generate_examples,
            gen_kwargs={"archive_path": archive_path, "max_count": max_count},
            features=features,
            cache_dir=self.generator_cache_dir(),
        )

    def map_fn(self, example):
        summary = example["summary"].strip()
        content = example["content"].strip()
        author = example["author"].strip()
        keep = bool(
            summary
            and content
            and author
            and author.casefold() != "[deleted]"
        )
        return simple_format(summary, content), author, keep


@register_handler
class ToucanHandler(BaseHandler):

    url = "Agent-Ark/Toucan-1.5M"
    subset = ["Kimi-K2", "OSS", "Qwen3", "SFT"]
    split = "train"

    kind = "tool"

    output_features = datasets.Features({
        "messages": datasets.List({
            "role": datasets.Value("string"),
            "content": datasets.Value("string"),
            "tool_calls": datasets.List({
                "type": datasets.Value("string"),
                "function": {
                    "name": datasets.Value("string"),
                    "arguments": datasets.Value("string"),
                },
            }),
        }),
        "latent": datasets.Value("string"),
        "keep": datasets.Value("bool"),
    })

    def load_dataset(self, max_count=None):
        parts = []
        subset_count = len(self.subset)
        for index, subset in enumerate(self.subset):
            subset_max = None
            if max_count is not None:
                subset_max = max_count // subset_count + (
                    index < max_count % subset_count
                )
                if subset_max == 0:
                    continue

            ds = self._load_dataset_part(subset, self.split, subset_max)
            tools_column = "tools" if subset == "SFT" else "available_tools"
            if tools_column != "available_tools":
                ds = ds.rename_column(tools_column, "available_tools")
            if subset == "SFT":
                ds = ds.add_column(
                    "question_quality_assessment",
                    [""] * len(ds),
                )
                ds = ds.add_column(
                    "response_quality_assessment",
                    [""] * len(ds),
                )
            normalized = ds.add_column("config", [subset] * len(ds))
            normalized = normalized.select_columns([
                "config",
                "messages",
                "available_tools",
                "question_quality_assessment",
                "response_quality_assessment",
            ])
            parts.append(normalized)
        return datasets.concatenate_datasets(parts)

    @staticmethod
    def _is_correct(example):
        if example["config"] == "SFT":
            return True
        assessment = _parse_jsonish(example["response_quality_assessment"])
        return bool(
            isinstance(assessment, dict)
            and assessment.get("desired_tools_used_percentage") == 1.0
            and assessment.get("order_correctness") is True
        )

    def map_fn(self, example):
        latent = _canonical_tools_hash(example["available_tools"])
        conversation = _toucan_tool_messages(
            example["messages"],
            example["available_tools"],
        )
        keep = bool(latent and conversation and self._is_correct(example))
        return conversation, latent, keep


class _NemotronAgenticHandler(BaseHandler):

    subset = "default"
    split = ["interactive_agent", "tool_calling"]

    kind = "tool"

    output_features = ToucanHandler.output_features

    @staticmethod
    def _generate_examples(repo_id, splits, max_counts, schema_version):
        del schema_version  # Included in gen_kwargs to invalidate old caches.
        for split, max_count in zip(splits, max_counts):
            if max_count == 0:
                continue

            filename = f"data/{split}.jsonl"
            response = None
            if max_count is None:
                path = hf_hub_download(repo_id, filename, repo_type="dataset")
                source = open(path, "rb")
            else:
                # The upstream JSONL has heterogeneous nested tool schemas,
                # which Arrow cannot infer. Stream bounded runs line-by-line
                # so a smoke test does not download the full 5-15 GiB file.
                response = requests.get(
                    hf_hub_url(repo_id, filename, repo_type="dataset"),
                    stream=True,
                    timeout=60,
                )
                response.raise_for_status()
                source = response.iter_lines()

            try:
                lines = source if response is not None else iter(source)
                count = 0
                for line in lines:
                    if not line or not line.strip():
                        continue
                    example = json.loads(line)
                    yield {
                        "messages_json": json.dumps(
                            example.get("messages"), ensure_ascii=False
                        ),
                        "tools_json": json.dumps(
                            example.get("tools"), ensure_ascii=False
                        ),
                        "split": split,
                    }
                    count += 1
                    if max_count is not None and count >= max_count:
                        break
            finally:
                if response is None:
                    source.close()
                else:
                    response.close()

    def load_dataset(self, max_count=None):
        splits = list(self.split)
        if max_count is None:
            max_counts = [None] * len(splits)
        else:
            max_counts = [
                max_count // len(splits) + (index < max_count % len(splits))
                for index in range(len(splits))
            ]

        features = datasets.Features({
            "messages_json": datasets.Value("string"),
            "tools_json": datasets.Value("string"),
            "split": datasets.Value("string"),
        })
        return datasets.Dataset.from_generator(
            self._generate_examples,
            gen_kwargs={
                "repo_id": self.url,
                "splits": splits,
                "max_counts": max_counts,
                "schema_version": 1,
            },
            features=features,
            cache_dir=self.generator_cache_dir(),
            num_proc=min(len(splits), max_count or len(splits)),
        )

    def map_fn(self, example):
        tools = example.get("tools_json")
        latent = _canonical_tools_hash(tools)
        conversation = _llama_tool_messages(
            example.get("messages_json"),
            tools,
            # Unlike Toucan's provider-specific system tool declaration, the
            # Nemotron system message often contains an agent policy needed
            # to understand and reproduce the trajectory.
            preserve_system=True,
        )
        return conversation, latent, bool(latent and conversation)


@register_handler
class NemotronSftAgenticV2Handler(_NemotronAgenticHandler):

    url = "nvidia/Nemotron-SFT-Agentic-v2"


@register_handler
class NemotronAgenticV1Handler(_NemotronAgenticHandler):

    url = "nvidia/Nemotron-Agentic-v1"


@register_handler
class MqaEnglishHandler(BaseHandler):

    url = "clips/mqa"
    subset = ["en-cqa-question", "en-faq-domain"]
    split = "train"

    kind = "qa"

    @staticmethod
    def _generate_examples(paths, max_counts, schema_version):
        del schema_version  # Included in gen_kwargs to invalidate old caches.
        for source_index, (subset, path) in enumerate(paths):
            source_max = None if max_counts is None else max_counts[source_index]
            count = 0
            with gzip.open(path, "rt", encoding="utf-8") as source:
                for line in source:
                    page = json.loads(line)
                    domain = page.get("domain") or ""
                    for question in page.get("questions") or []:
                        prompt = (
                            question.get("text")
                            or question.get("name")
                            or ""
                        )
                        yield {
                            "subset": subset,
                            "question": prompt,
                            "answers": [
                                answer.get("text") or ""
                                for answer in question.get("answers") or []
                            ],
                            "domain": domain,
                        }
                        count += 1
                        if source_max is not None and count >= source_max:
                            break
                    if source_max is not None and count >= source_max:
                        break

    def load_dataset(self, max_count=None):
        paths = [
            (
                "en-cqa-question",
                hf_hub_download(
                    self.url,
                    "data/data.en.cqa.json.gz",
                    repo_type="dataset",
                ),
            ),
            (
                "en-faq-domain",
                hf_hub_download(
                    self.url,
                    "data/data.en.faq.json.gz",
                    repo_type="dataset",
                ),
            ),
        ]
        features = datasets.Features({
            "subset": datasets.Value("string"),
            "question": datasets.Value("string"),
            "answers": datasets.List(datasets.Value("string")),
            "domain": datasets.Value("string"),
        })
        max_counts = None
        if max_count is not None:
            max_counts = [
                max_count // len(paths) + (index < max_count % len(paths))
                for index in range(len(paths))
            ]
        return datasets.Dataset.from_generator(
            self._generate_examples,
            gen_kwargs={
                "paths": paths,
                "max_counts": max_counts,
                "schema_version": 3,
            },
            features=features,
            cache_dir=self.generator_cache_dir(),
            # Each requested upstream config is a separate large gzip archive.
            # Shard the two paths rather than reading them serially for hours.
            num_proc=len(paths),
        )

    def map_fn(self, example):
        question = example["question"].strip()
        domain = example["domain"].strip()
        answers = [answer.strip() for answer in example["answers"] if answer.strip()]
        conversations = [simple_format(question, answer) for answer in answers]
        keep = bool(question and domain)
        return (
            conversations,
            [domain] * len(conversations),
            [keep] * len(conversations),
        )


@register_handler
class P3Handler(BaseHandler):

    url = "bigscience/P3"
    # P3 exposes every task/template as a separate dataset configuration.
    # The configuration names are discovered at runtime rather than copied
    # into this handler (there are several hundred of them).
    subset = datasets.get_dataset_config_names("bigscience/P3")
    split = "train"

    kind = "qa"

    def map_fn(self, example):
        prompt = (example.get("inputs_pretokenized") or "").strip()
        target = (example.get("targets_pretokenized") or "").strip()
        subset = example.get("subset")
        keep = bool(prompt and target and subset)
        return simple_format(prompt, target), subset, keep


@register_handler
class TasksourceInstructHandler(BaseHandler):

    url = "tasksource/tasksource-instruct-v0"
    subset = "default"
    split = "train"

    kind = "qa"

    def map_fn(self, example):
        prompt = (example.get("inputs") or "").strip()
        target = (example.get("targets") or "").strip()
        task = (example.get("task") or "").strip()
        keep = bool(prompt and target and task)
        return simple_format(prompt, target), task, keep


@register_handler
class FanaticFandomHandler(BaseHandler):

    url = "recursal/Fanatic-Fandom"
    subset = "default"
    split = "train"

    kind = "wiki"

    def map_fn(self, example):
        metadata = example.get("meta") or {}
        title = (metadata.get("title") or "").strip()
        text = (example.get("text") or "").strip()
        domain = (metadata.get("domain") or "").strip()
        keep = bool(title and text and domain)
        return simple_format(title, text), domain, keep


@register_handler
class SynthHandler(BaseHandler):

    url = "PleIAs/SYNTH"
    subset = "default"
    split = "train"

    kind = "qa"

    def map_fn(self, example):
        query = (example.get("query") or "").strip()
        answer = (example.get("synthetic_answer") or "").strip()
        seed_url = (example.get("query_seed_url") or "").strip()
        language = (example.get("language") or "").strip().lower()
        keep = bool(
            language in {"en", "english"}
            and query
            and answer
            and seed_url
        )
        return simple_format(query, answer), seed_url, keep


@register_handler
class LongAbcHandler(BaseHandler):

    url = "Lyun0912/LongABC"
    subset = "default"
    split = "train"

    kind = "document"

    chunk_tokenizer_url = "meta-llama/Llama-3.2-1B-Instruct"
    chunk_size = 256

    # A LongABC latent is an ordered document sequence. Preserve disjoint
    # consecutive chunk pairs and only emit trajectories containing a
    # complete horizon.
    shuffle_episodes = False
    drop_incomplete_trajectories = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._chunker = None

    def _get_chunker(self):
        if self._chunker is None:
            self._chunker = semchunk.chunkerify(
                self.chunk_tokenizer_url,
                chunk_size=self.chunk_size,
            )
        return self._chunker

    def map_fn(self, example):
        content = (example.get("content") or "").strip()
        sequence = (example.get("data_id") or "").strip()
        if not content or not sequence:
            return [], [], []

        chunks = [
            chunk.strip()
            for chunk in self._get_chunker()(content)
            if chunk.strip()
        ]
        conversations = [
            simple_format(current, following)
            for current, following in zip(chunks[::2], chunks[1::2])
        ]
        return (
            conversations,
            [sequence] * len(conversations),
            [True] * len(conversations),
        )


@register_handler
class DhsaLongDataCollectionsHandler(LongAbcHandler):

    url = "sxiong/DHSA_Long-Data-Collections"
    subset = ["32k_64k", "64k_128k", "gt_128k"]
    split = "pretrain"

    chunk_size = 448

    @staticmethod
    def _iter_file_examples(repo_id, filename, max_count=None):
        """Yield source documents from one compressed JSONL shard."""
        path = hf_hub_download(repo_id, filename, repo_type="dataset")
        with pa.input_stream(path) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as source:
                count = 0
                for line_number, line in enumerate(source):
                    if not line.strip():
                        continue
                    example = json.loads(line)
                    yield {
                        "content": example.get("text") or "",
                        # DHSA does not publish a document identifier. A
                        # repository path plus JSONL line number is stable
                        # and keeps every document's chunks in one latent.
                        "sequence": hashlib.sha256(
                            f"{filename}:{line_number}".encode("utf-8")
                        ).hexdigest(),
                    }
                    count += 1
                    if max_count is not None and count >= max_count:
                        return

    @classmethod
    def _generate_examples(cls, repo_id, filenames, max_counts, schema_version):
        del schema_version  # Included in gen_kwargs to invalidate old caches.
        for filename, max_count in zip(filenames, max_counts):
            yield from cls._iter_file_examples(repo_id, filename, max_count)

    @classmethod
    def _generate_balanced_examples(
        cls,
        repo_id,
        bucket_filenames,
        max_count,
        schema_version,
    ):
        """Round-robin a single global row budget across length buckets.

        Each bucket advances through its shards in filename order. Exhausted
        buckets leave the rotation, so their unused share is automatically
        redistributed among buckets that still contain rows.
        """
        del schema_version  # Included in gen_kwargs to invalidate old caches.

        def iter_bucket(filenames):
            for filename in filenames:
                yield from cls._iter_file_examples(repo_id, filename)

        active = [iter_bucket(filenames) for filenames in bucket_filenames]
        emitted = 0
        while active and emitted < max_count:
            remaining = []
            for source in active:
                if emitted >= max_count:
                    break
                try:
                    example = next(source)
                except StopIteration:
                    continue
                yield example
                emitted += 1
                remaining.append(source)
            active = remaining

    def load_dataset(self, max_count=None):
        repo_files = list_repo_files(self.url, repo_type="dataset")
        bucket_filenames = [
            sorted(
                filename
                for filename in repo_files
                if filename.startswith(f"pretrain/{bucket}/")
                and filename.endswith(".jsonl.zst")
            )
            for bucket in self.subset
        ]
        filenames = [
            filename
            for bucket_files in bucket_filenames
            for filename in bucket_files
        ]
        if not filenames:
            raise FileNotFoundError(
                f"No requested pretrain length buckets found in {self.url}."
            )

        features = datasets.Features({
            "content": datasets.Value("string"),
            "sequence": datasets.Value("string"),
        })
        if max_count == 0:
            return datasets.Dataset.from_dict(
                {"content": [], "sequence": []},
                features=features,
            )

        if max_count is not None:
            return datasets.Dataset.from_generator(
                self._generate_balanced_examples,
                gen_kwargs={
                    "repo_id": self.url,
                    "bucket_filenames": bucket_filenames,
                    "max_count": max_count,
                    "schema_version": 2,
                },
                features=features,
                cache_dir=self.generator_cache_dir(),
            )

        return datasets.Dataset.from_generator(
            self._generate_examples,
            gen_kwargs={
                "repo_id": self.url,
                "filenames": filenames,
                "max_counts": [None] * len(filenames),
                "schema_version": 2,
            },
            features=features,
            cache_dir=self.generator_cache_dir(),
            num_proc=min(4, len(filenames)),
        )

    def map_fn(self, example):
        content = (example.get("content") or "").strip()
        sequence = (example.get("sequence") or "").strip()
        if not content or not sequence:
            return [], [], []

        chunks = [
            chunk.strip()
            for chunk in self._get_chunker()(content)
            if chunk.strip()
        ]
        conversations = [
            simple_format(current, following)
            for current, following in zip(chunks[::2], chunks[1::2])
        ]
        return (
            conversations,
            [sequence] * len(conversations),
            [True] * len(conversations),
        )


@register_handler
class SporcHandler(BaseHandler):

    url = "blitt/SPoRC"
    subset = "episodes"
    split = "train"

    kind = "podcast"
    max_num_proc = 16

    @staticmethod
    def _generate_examples(repo_id, filenames, max_counts, schema_version):
        del schema_version  # Included in gen_kwargs to invalidate old caches.
        columns = [
            "ep_title",
            "ep_description",
            "transcript",
            "itunes_author",
            "language",
        ]
        for filename, max_count in zip(filenames, max_counts):
            path = hf_hub_download(repo_id, filename, repo_type="dataset")
            count = 0
            parquet_file = pq.ParquetFile(path)
            for row_group in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(row_group, columns=columns)
                for example in table.to_pylist():
                    yield example
                    count += 1
                    if max_count is not None and count >= max_count:
                        break
                if max_count is not None and count >= max_count:
                    break

    def load_dataset(self, max_count=None):
        filenames = sorted(
            filename
            for filename in list_repo_files(self.url, repo_type="dataset")
            if filename.startswith("episodes/part-")
            and filename.endswith(".parquet")
        )
        if max_count is not None:
            filenames = filenames[:1] if max_count > 0 else []

        features = datasets.Features({
            "ep_title": datasets.Value("string"),
            "ep_description": datasets.Value("string"),
            "transcript": datasets.Value("string"),
            "itunes_author": datasets.Value("string"),
            "language": datasets.Value("string"),
        })
        if not filenames:
            return datasets.Dataset.from_dict(
                {key: [] for key in features},
                features=features,
            )
        return datasets.Dataset.from_generator(
            self._generate_examples,
            gen_kwargs={
                "repo_id": self.url,
                "filenames": filenames,
                "max_counts": [max_count] * len(filenames),
                "schema_version": 2,
            },
            features=features,
            cache_dir=self.generator_cache_dir(),
            num_proc=min(4, len(filenames)),
        )

    def map_fn(self, example):
        title = (example.get("ep_title") or "").strip()
        description = _safe_html_to_markdown(example.get("ep_description"))
        prompt = f"{title}\n{description}" if description else title
        transcript = (example.get("transcript") or "").strip()
        author = (example.get("itunes_author") or "").strip()
        keep = bool(
            _is_english_language(example.get("language"))
            and prompt
            and transcript
            and author
        )
        return simple_format(prompt, transcript), author, keep


@register_handler
class YoutubeCommonsHandler(BaseHandler):

    url = "PleIAs/YouTube-Commons"
    subset = "default"
    split = "train"

    kind = "video"
    max_num_proc = 16

    @staticmethod
    def _generate_examples(repo_id, filenames, max_counts, schema_version):
        del schema_version  # Included in gen_kwargs to invalidate old caches.
        for filename, max_count in zip(filenames, max_counts):
            path = hf_hub_download(repo_id, filename, repo_type="dataset")
            count = 0
            parquet_file = pq.ParquetFile(path)
            available = set(parquet_file.schema_arrow.names)
            columns = [
                column
                for column in (
                    "description",
                    "video_description",
                    "title",
                    "text",
                    "channel",
                    "channel_id",
                    "transcription_language",
                )
                if column in available
            ]
            for row_group in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(row_group, columns=columns)
                for example in table.to_pylist():
                    yield {
                        "description": (
                            example.get("description")
                            or example.get("video_description")
                            or example.get("title")
                            or ""
                        ),
                        "transcript": example.get("text") or "",
                        "channel": example.get("channel") or "",
                        "channel_id": example.get("channel_id") or "",
                        "transcription_language": (
                            example.get("transcription_language") or ""
                        ),
                    }
                    count += 1
                    if max_count is not None and count >= max_count:
                        break
                if max_count is not None and count >= max_count:
                    break

    def load_dataset(self, max_count=None):
        filenames = sorted(
            filename
            for filename in list_repo_files(self.url, repo_type="dataset")
            if re.fullmatch(r"cctube_\d+\.parquet", filename)
        )
        if max_count is not None:
            filenames = filenames[:1] if max_count > 0 else []

        features = datasets.Features({
            "description": datasets.Value("string"),
            "transcript": datasets.Value("string"),
            "channel": datasets.Value("string"),
            "channel_id": datasets.Value("string"),
            "transcription_language": datasets.Value("string"),
        })
        if not filenames:
            return datasets.Dataset.from_dict(
                {key: [] for key in features},
                features=features,
            )
        return datasets.Dataset.from_generator(
            self._generate_examples,
            gen_kwargs={
                "repo_id": self.url,
                "filenames": filenames,
                "max_counts": [max_count] * len(filenames),
                "schema_version": 2,
            },
            features=features,
            cache_dir=self.generator_cache_dir(),
            num_proc=min(4, len(filenames)),
        )

    def map_fn(self, example):
        description = (example.get("description") or "").strip()
        transcript = (example.get("transcript") or "").strip()
        channel = (example.get("channel") or "").strip()
        channel_id = (example.get("channel_id") or "").strip()
        # The stable channel id is the author identity; retain the display
        # name only as a fallback for rows where it is absent.
        latent = channel_id or channel
        keep = bool(
            _is_english_language(example.get("transcription_language"))
            and description
            and transcript
            and latent
        )
        return simple_format(description, transcript), latent, keep


@register_handler
class GeniusLyricsCleanedHandler(BaseHandler):

    url = "theelderemo/genius-lyrics-cleaned"
    subset = "default"
    split = "train"

    kind = "lyrics"

    def map_fn(self, example):
        title = (example.get("title") or "").strip()
        lyrics = (example.get("lyrics") or "").strip()
        artist = (example.get("artist") or "").strip()
        keep = bool(title and lyrics and artist)
        return simple_format(title, lyrics), artist, keep


@register_handler
class MediumArticlesEnglishHandler(BaseHandler):

    url = "BEE-spoke-data/medium-articles-en"
    subset = "default"
    split = ["train", "validation", "test"]

    kind = "blog"

    @staticmethod
    def _authors(value):
        if isinstance(value, str):
            value = _parse_jsonish(value)
        if not isinstance(value, (list, tuple)):
            return []

        # Preserve source order while avoiding duplicate trajectories for a
        # malformed row that names the same author more than once.
        authors = []
        seen = set()
        for value in value:
            if not isinstance(value, str):
                continue
            author = value.strip()
            if author and author not in seen:
                authors.append(author)
                seen.add(author)
        return authors

    def map_fn(self, example):
        title = (example.get("title") or "").strip()
        article = (example.get("text") or "").strip()
        authors = self._authors(example.get("authors"))
        if not title or not article or not authors:
            return [], [], []

        conversation = simple_format(title, article)
        return (
            [conversation for _ in authors],
            authors,
            [True] * len(authors),
        )
