
import json
import re
import zipfile

import datasets
from huggingface_hub import hf_hub_download

from base_handler import BaseHandler
from utils import html_to_markdown, simple_format


_HANDLER_REGISTRY: dict[str, type[BaseHandler]] = {}
_FIRST_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


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
        return conversation, example["url"], True


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

    def load_dataset(self, max_count=None):
        parts = []
        remaining = max_count
        for subset in datasets.get_dataset_config_names(self.url):
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
