
import sys
import inspect

import datasets
import random
import json

from base_handler import BaseHandler
from utils import simple_format


"""
All handlers must define:

def map_fn(self, example):
    ...
    return conversation, latent, keep
"""


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
        

HANDLERS = [
    x[1] for x in inspect.getmembers(sys.modules[__name__])
    if inspect.isclass(x[1]) and issubclass(x[1], BaseHandler) and x[1] is not BaseHandler
][::-1]
