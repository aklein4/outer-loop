
import sys
import inspect

import datasets
import random
import json

from base_handler import BaseHandler


""" ===== Helpers ===== """


def get_splits(url, subset, remove=[]):
    all_splits = list(datasets.get_dataset_split_names(url, subset))
    for s in remove:
        assert s in all_splits, f"Expected split {s} not found in dataset."
        all_splits.remove(s)
    return all_splits


def get_subsets(url, remove=[]):
    all_subsets = list(datasets.get_dataset_config_names(url))
    for s in remove:
        assert s in all_subsets, f"Expected subset {s} not found in dataset."
        all_subsets.remove(s)
    return all_subsets


def convert_role(role):
    if role in ["system", "user", "assistant"]:
        return role
    if role == "human":
        return "user"
    if role == "gpt":
        return "assistant"
    return role

def convert_role_conversation(conversation):
    for turn in conversation:

        if "role" not in turn:
            turn["role"] = turn.pop("from")
        if "content" not in turn:
            turn["content"] = turn.pop("value")

        turn["role"] = convert_role(turn["role"])

    return conversation


def remove_think(content):
    if "</think>" in content:
        return content.split("</think>")[-1].strip()
    return content


def key_filter(example, keys):
    for key in keys:
        if example.get(key, None) is None:
            return False
    return True


def single_turn(conversation):
    out = []
    for turn in conversation:
        out.append(turn)
        if turn["role"] == "assistant":
            break
    return out


def system_turn(content):
    return {"role": "system", "content": content}
def user_turn(content):
    return {"role": "user", "content": content}
def assistant_turn(content):
    return {"role": "assistant", "content": content}


""" ===== Post-Training ===== """


class NemotronPTHandler(BaseHandler):

    url = "nvidia/Nemotron-Post-Training-Dataset-v2"
    subset = None
    split = ["stem", "chat", "math", "code"]

    kind = "post_training"

    def map(self, example):
        return example["messages"]


class LlamaNemotronPTHandler(BaseHandler):

    url = "nvidia/Llama-Nemotron-Post-Training-Dataset"
    subset = "SFT"
    split = get_splits(
        "nvidia/Llama-Nemotron-Post-Training-Dataset",
        "SFT"
    )

    kind = "post_training"

    def map(self, example):
        return (
            example["input"] + 
            [assistant_turn(example["output"])]
        )


""" ===== Chat ===== """


class SmolTalkHandler(BaseHandler):

    url = "HuggingFaceTB/smoltalk"
    subset = get_subsets(
        "HuggingFaceTB/smoltalk",
        remove=["all"]
    )   
    split = "train"

    kind = "chat"

    def map(self, example):
        return example["messages"]


class SmolTalk2Handler(BaseHandler):

    url = "HuggingFaceTB/smoltalk2"
    subset = "SFT"
    split = get_splits(
        "HuggingFaceTB/smoltalk2",
        "SFT",
        [
            "smoltalk_multilingual8_Qwen3_32B_think",
            "LongAlign_64k_context_lang_annotated_lang_6_no_think",
            "smoltalk_multilingual_8languages_lang_5_no_think",   
        ],
    )

    kind = "chat"

    def map(self, example):
        return example["messages"]


class HermesHandler(BaseHandler):

    url = "teknium/OpenHermes-2.5"
    subset = None
    split = "train"

    kind = "chat"

    def map(self, example):
        out = convert_role_conversation(example["conversations"])
        for turn in out:
            turn.pop("weight", None)
        return out


class OrcaAgentInstructHandler(BaseHandler):

    url = "microsoft/orca-agentinstruct-1M-v1"
    subset = None
    split = get_splits(
        "microsoft/orca-agentinstruct-1M-v1",
        None,
    )

    kind = "chat"

    def map(self, example):
        return json.loads(example["messages"])



class WebInstructHandler(BaseHandler):

    url = "TIGER-Lab/WebInstructSub"
    subset = None
    split = "train"

    kind = "chat"

    def map(self, example):
        return [
            user_turn(example["question"]),
            assistant_turn(example["answer"])
        ]


class NemotronIFv1Handler(BaseHandler):

    url = "nvidia/Nemotron-Instruction-Following-Chat-v1"
    subset = None
    split = "chat_if"

    kind = "chat"

    def map(self, example):
        return example["messages"]


# class NemotronIFv2Handler(BaseHandler):

#     url = "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2"
#     subset = None
#     split = get_splits(
#         "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2",
#         None
#     )

#     kind = "chat"

#     def map(self, example):
#         return example["messages"]


class NemotronIFv3Handler(BaseHandler):

    url = "nvidia/Nemotron-SFT-Instruction-Following-Chat-v3"
    subset = None
    split = get_splits(
        "nvidia/Nemotron-SFT-Instruction-Following-Chat-v3",
        None
    )

    kind = "chat"

    def map(self, example):
        return example["messages"]


class OpenOrcaHandler(BaseHandler):

    url = "Open-Orca/OpenOrca"
    subset = None
    split = "train"

    kind = "chat"

    def map(self, example):
        return [
            system_turn(example["system_prompt"]),
            user_turn(example["question"]),
            assistant_turn(example["response"])
        ]


class UltraChatHandler(BaseHandler):

    url = "HuggingFaceH4/ultrachat_200k"
    subset = None
    split = ["train_sft", "train_gen"]

    kind = "chat"

    def map(self, example):
        return example["messages"]


class TuluV1Handler(BaseHandler):

    url = "allenai/tulu-v1-sft-mixture"
    subset = None
    split = "train"

    kind = "chat"

    def map(self, example):
        return example["messages"]


class TuluV2Handler(BaseHandler):

    url = "allenai/tulu-v2-sft-mixture"
    subset = None
    split = "train"

    kind = "chat"

    def map(self, example):
        return example["messages"]


class TuluV3Handler(BaseHandler):

    url = "allenai/tulu-3-sft-mixture"
    subset = None
    split = "train"

    kind = "chat"

    def map(self, example):
        return example["messages"]


class TuluV3p1Handler(BaseHandler):

    url = "allenai/tulu-v3.1-mix-preview-4096-OLMoE"
    subset = None
    split = "train"

    kind = "chat"

    def map(self, example):
        return example["messages"]


class MagpieLlama3ProHandler(BaseHandler):

    url = "Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1"
    subset = None
    split = "train"

    kind = "chat"

    def map(self, example):
        return convert_role_conversation(example["conversations"])


class MagpieLlama3AirHandler(BaseHandler):

    url = "Magpie-Align/Llama-3-Magpie-Air-3M-v0.1"
    subset = None
    split = "train"

    kind = "chat"

    def map(self, example):
        return convert_role_conversation(example["conversations"])


class MagpieQwen2p5ProHandler(BaseHandler):

    url = "Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1"
    subset = None
    split = "train"

    kind = "chat"

    def map(self, example):
        return convert_role_conversation(example["conversations"])


class MagpieQwen2p5MathProHandler(BaseHandler):

    url = "Magpie-Align/Magpie-Qwen2.5-Math-Pro-300K-v0.1"
    subset = None
    split = "train"

    kind = "chat"

    def map(self, example):
        return convert_role_conversation(example["conversations"])


class MagpieQwen2p5CoderProHandler(BaseHandler):

    url = "Magpie-Align/Magpie-Qwen2.5-Coder-Pro-300K-v0.1"
    subset = None
    split = "train"

    kind = "chat"

    def map(self, example):
        return convert_role_conversation(example["conversations"])


class MagpieQwen2ProHandler(BaseHandler):

    url = "Magpie-Align/Magpie-Qwen2-Pro-200K-English"
    subset = None
    split = "train"

    kind = "chat"

    def map(self, example):
        return convert_role_conversation(example["conversations"])


class TomeHandler(BaseHandler):

    url = "arcee-ai/The-Tome"
    subset = None
    split = "train"

    kind = "chat"

    def map(self, example):
        return convert_role_conversation(example["conversations"])



""" ===== Reasoning ===== """


class NaturalReasoningHandler(BaseHandler):

    url = "facebook/natural_reasoning"
    subset = None
    split = "train"

    kind = "reasoning"

    def map(self, example):
        return [
            user_turn(example["question"]),
            assistant_turn(example["responses"][0]["response"])
        ]


class AceReasonHandler(BaseHandler):

    url = "nvidia/AceReason-1.1-SFT"
    subset = None
    split = "train"

    kind = "reasoning"

    def map(self, example):
        return [
            user_turn(example["input"]),
            assistant_turn(example["output"])
        ]


class AceMathHandler(BaseHandler):

    url = "nvidia/AceMath-Instruct-Training-Data"
    subset = None
    split = get_splits(
        "nvidia/AceMath-Instruct-Training-Data",
        None,
    )

    kind = "reasoning"

    verification_mode = datasets.VerificationMode.NO_CHECKS

    def map(self, example):
        return (
            example["messages"] +
            [assistant_turn(example["answer"])]
        )


""" ===== MCQA ===== """


class OpenScienceHandler(BaseHandler):

    url = "nvidia/OpenScience"
    subset = get_subsets(
        "nvidia/OpenScience",
    )
    split = "train"

    kind = "mcqa"

    def map(self, example):
        return [
            user_turn(example["input"]),
            assistant_turn(example["output"])
        ]
    

class OpenScienceReasoningHandler(BaseHandler):

    url = "nvidia/OpenScienceReasoning-2"
    subset = None
    split = "train"

    kind = "mcqa"

    def map(self, example):
        # this is a good source of just-letter answers
        return [
            user_turn(example["input"]),
            assistant_turn(example["expected_answer"]),
        ]


class NemotronScienceHandler(BaseHandler):

    url = "nvidia/Nemotron-Science-v1"
    subset = None
    split = get_splits(
        "nvidia/Nemotron-Science-v1",
        None,
    )

    kind = "mcqa"

    def map(self, example):
        return example["messages"]


class SciqHandler(BaseHandler):

    url = "allenai/sciq"
    subset = None
    split = "train"

    kind = "mcqa"

    def map(self, example):
        
        options = [
            example["distractor1"],
            example["distractor2"],
            example["distractor3"],
            example["correct_answer"]
        ]
        random.shuffle(options)

        correct_ind = options.index(example["correct_answer"])
        answer = ["A", "B", "C", "D"][correct_ind]

        x = example["question"]+"\n"
        for i, opt in enumerate(options):
            x += f"{['A', 'B', 'C', 'D'][i]}: {opt}\n"
        x = x.strip()

        return [
            user_turn(x),
            assistant_turn(answer),
        ]


""" ===== Math ===== """


class StackMathHandler(BaseHandler):

    url = "math-ai/StackMathQA"
    subset = "stackmathqa800k"
    split = "train"

    kind = "math"

    def map(self, example):
        return [
            user_turn(example["Q"]),
            assistant_turn(example["A"])
        ]


class MetaMathHandler(BaseHandler):

    url = "meta-math/MetaMathQA"
    subset = None
    split = "train"

    kind = "math"

    def map(self, example):
        return [
            user_turn(example["query"]),
            assistant_turn(example["response"])
        ]


class MathPlusHandler(BaseHandler):

    url = "TIGER-Lab/MATH-plus"
    subset = None
    split = "train"

    kind = "math"

    def map(self, example):
        return [
            user_turn(example["instruction"]),
            assistant_turn(example["output"])
        ]


class OpenMathInstruct1Handler(BaseHandler):

    url = "nvidia/OpenMathInstruct-1"
    subset = None
    split = "train"

    kind = "math"

    def map(self, example):
        if not example["is_correct"]:
            return None

        return [
            user_turn(example["question"]),
            assistant_turn(example["generated_solution"]),
        ]


class OpenMathInstruct2Handler(BaseHandler):

    url = "nvidia/OpenMathInstruct-2"
    subset = None
    split = "train_2M" # use a smaller subset because idk how good the quality is

    kind = "math"

    def map(self, example):
        return [
            user_turn(example["problem"]),
            assistant_turn(example["generated_solution"]),
        ]


class PrismMathHandler(BaseHandler):

    url = "nvidia/Nemotron-PrismMath"
    subset = None
    split = "train"

    kind = "math"

    def map(self, example):
        return [
            user_turn(example["problem"]),
            assistant_turn(example["solution"]),
        ]


class ODAMathHandler(BaseHandler):

    url = "OpenDataArena/ODA-Math-460k"
    subset = None
    split = "train"

    kind = "math"

    def map(self, example):
        # this is a good source of just-answer outputs
        return [
            user_turn(example["question"]),
            assistant_turn(example["expected_answer"]),
        ]


""" ===== Code ===== """


class OpenCodeReasoningHandler(BaseHandler):

    url = "nvidia/OpenCodeReasoning"
    subset = "split_0"
    split = "split_0"

    kind = "code"

    def map(self, example):
        sol = example["solution"].strip()
        return [
            user_turn(example["input"]),
            assistant_turn(f"```python\n{sol}\n```")
        ]


LEAN_PROMPTS = [
    "Use Lean to formally prove the following statement.",
    "Use Lean to formally prove the following statement.",
    "Construct a formal proof in Lean for the following mathematical claim.",
    "Develop a rigorous proof in Lean for the stated mathematical proposition.",
    "Write the proof of the following mathematical statement in Lean.",
    "Demonstrate the proof of the given mathematical assertion using Lean.",
    "Formulate a detailed proof in Lean for the following mathematical claim.",
    "Using Lean, provide a formal proof for the stated mathematical proposition.",
    "",
    "",
]


class NemotronMathProofsHandler(BaseHandler):

    url = "nvidia/Nemotron-Math-Proofs-v1"
    subset = None
    split = "lean"

    kind = "code"

    def map(self, example):

        system = random.choice(LEAN_PROMPTS)

        head = example["lean_header"].strip()+"\n\n" if example["lean_header"] is not None else ""
        statement = example["formal_statement"].strip() if example["formal_statement"] is not None else None

        if statement is None:
            return None

        out = [
            user_turn(example["problem"]),
            assistant_turn(f"```lean\n{head}{statement}\n```")
        ]

        if system != "":
            out = [system_turn(system)] + out

        return out
    

class TinyCodesHandler(BaseHandler):

    url = "nampdn-ai/tiny-codes"
    subset = None
    split = "train"

    kind = "code"

    def map(self, example):
        if "```" not in example["response"]:
            return None

        code = "```" + example["response"].split("```")[1].strip() + "\n```"

        return [
            user_turn(example["prompt"]),
            assistant_turn(code)
        ]


class GoedelHandler(BaseHandler):

    url = "Goedel-LM/Goedel-Pset-v1"
    subset = None
    split = "train"

    kind = "code"

    def map(self, example):

        statement = example["formal_statement"].strip()
        system = random.choice(LEAN_PROMPTS)

        out = [
            user_turn(example["informal_statement"]),
            assistant_turn(f"```lean\n{statement}\n```"),
        ]

        if system != "":
            out = [system_turn(system)] + out

        return out


HANDLERS = [x[1] for x in inspect.getmembers(sys.modules[__name__]) if inspect.isclass(x[1]) and issubclass(x[1], BaseHandler) and x[1] is not BaseHandler][::-1]
