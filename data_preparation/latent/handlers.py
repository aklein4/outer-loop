
import ast
import gzip
import hashlib
import json
import re
import zipfile
from pathlib import Path

import datasets
import semchunk
from huggingface_hub import hf_hub_download

from base_handler import BaseHandler
from utils import html_to_markdown, simple_format


_HANDLER_REGISTRY: dict[str, type[BaseHandler]] = {}
_FIRST_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


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


def _llama_tool_messages(raw_messages, available_tools):
    """Convert Toucan's provider formats to Llama function-call messages."""
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

        # Original system prompts contain model-specific tool syntax. The
        # canonical Llama-compatible declaration above replaces them.
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
                        # OpenAI/Llama tool-call interchange represents
                        # arguments as a JSON-encoded string. Keeping that
                        # scalar schema also avoids Arrow trying to unify the
                        # unrelated argument structs of different tools.
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
            # Some OSS rows contain empty assistant messages whose useful
            # payload is provider-only hidden reasoning.
            continue
        converted.append({"role": role, "content": str(content)})

    return converted


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
        keep = bool(summary and content and author)
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
        conversation = _llama_tool_messages(
            example["messages"],
            example["available_tools"],
        )
        keep = bool(latent and conversation and self._is_correct(example))
        return conversation, latent, keep


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
    chunk_size = 448

    # A LongABC latent is an ordered document sequence. Preserve consecutive
    # chunk pairs and only emit trajectories containing a complete horizon.
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
            for current, following in zip(chunks, chunks[1:])
        ]
        return (
            conversations,
            [sequence] * len(conversations),
            [True] * len(conversations),
        )
