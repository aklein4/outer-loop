
from base_handler import BaseHandler
from utils import simple_format


_HANDLER_REGISTRY: dict[str, type[BaseHandler]] = {}


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
