import torch

import numpy as np
import string
import sys
import inspect

import datasets
from transformers import PreTrainedTokenizer

import utils.constants as constants
import utils.chat_utils as chat

BS = 1000


class BaseBenchmark:

    name: str = None

    url: str = None
    subset: str = None
    split: str = None

    is_default = False


    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
        max_examples: int | None = None,
    ):
        
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.max_examples = max_examples

        self.dataset = self.get_dataset()

    
    def get_dataset(self):

        data = datasets.load_dataset(
            self.url,
            name=self.subset,
            split=self.split,
            streaming=False,
        )

        data = data.map(
            self.data_map_fn,
            batched=True,
            batch_size=BS,
            load_from_cache_file=False,
        )
        data = data.filter(
            self.data_filter_fn,
            batched=True,
            batch_size=BS,
            load_from_cache_file=False,
        )
        data = data.remove_columns("keep")

        self.dataset = data
        data = data.map(
            self.truncate_map_fn,
            batched=True,
            batch_size=BS,
            load_from_cache_file=False,
        )
        self.dataset = None

        if self.max_examples is not None:
            data = data.select(range(min(self.max_examples, len(data))))

        return data


    def tokenize(self, text, in_or_out):

        if in_or_out == "output":
            max_length = self.max_output_length + 1
        elif in_or_out == "input":
            max_length = self.max_input_length + 1
        else:
            raise ValueError(f"in_or_out must be either 'input' or 'output', got {in_or_out}")

        ids = self.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).input_ids

        keep = ids[:, -1] == self.tokenizer.pad_token_id

        return ids[:, :-1], keep


    def data_map_fn(self, batch):
        """
        Return a dict containing "input_ids", "output_ids", "keep", and other keys.
        """
        raise NotImplementedError("data_map_fn must be implemented by subclasses of BaseBenchmark")


    def data_filter_fn(self, batch):
        return batch["keep"]


    def truncate_map_fn(self, batch):
        l_in = self.largest_input()
        l_out = self.largest_output()

        batch["input_ids"] = [
            ids[:l_in] for ids in batch["input_ids"]
        ]
        batch["output_ids"] = [
            ids[:l_out] for ids in batch["output_ids"]
        ]

        return batch


    def largest_input(self):
        lengths = [
            np.sum(np.array(ids) != self.tokenizer.pad_token_id) for ids in self.dataset["input_ids"]
        ]
        return np.max(lengths)
    
    def largest_output(self):
        lengths = [
            np.sum(np.array(ids) != self.tokenizer.pad_token_id) for ids in self.dataset["output_ids"]
        ]
        return np.max(lengths)
    

    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        self._curr_index = 0
        return self

    def __next__(self):
        if self._curr_index >= len(self):
            raise StopIteration

        example = self[self._curr_index]
        self._curr_index += 1

        return example


    def __getitem__(self, idx):
        return {k: self.dataset[k][idx] for k in self.dataset.column_names}


    def collate_fn(self, batch):

        d = {}
        for k in batch[0].keys():
            ex = batch[0][k]

            if isinstance(ex, (int, float, list)):
                
                if isinstance(ex, list):
                    if isinstance(ex[0], (int, float)):
                        d[k] = torch.tensor([example[k] for example in batch], device=constants.DEVICE)
                        continue
                else:
                    try:
                        d[k] = torch.tensor([example[k] for example in batch], device=constants.DEVICE)
                        continue
                    except:
                        pass

            d[k] = [example[k] for example in batch]

        return d


    def grade(self, batch, model_logits):
        """
        Determine whether the model is correct using its output_id logits.
        """
        raise NotImplementedError("grade must be implemented by subclasses of BaseBenchmark")


class MCQABenchmark(BaseBenchmark):

    is_default = True

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
        max_examples: int | None = None,
    ):
        
        self.prefix_length = len(
            tokenizer(chat.NO_COT_PREFIX, add_special_tokens=False).input_ids
        )
        self.letter_ids = tokenizer(
            list(string.ascii_uppercase),
            add_special_tokens=False,
        ).input_ids
        self.torch_letter_ids = torch.tensor(self.letter_ids, device=constants.DEVICE, dtype=torch.long)[:, 0]

        super().__init__(
            tokenizer,
            max_input_length,
            max_output_length,
            max_examples,
        )


    def extract_example(self, example):
        """
        Should return prompt, choices, and answer (letter)
        """
        raise NotImplementedError("extract_example must be implemented by subclasses of MCQABenchmark")


    def mcqa_format(self, prompt, choices, answer):
        answer = answer.strip().upper()

        input_text = chat.format_no_cot(
            chat.mcqa_question(prompt, choices), "_", "_"
        )[0]
        output_text = chat.no_cot_format(answer)

        keep = answer in string.ascii_uppercase[:len(choices)]
        answer_index = string.ascii_uppercase.index(answer) if keep else None

        return {
            "input_text": input_text,
            "output_text": output_text,
            "num_choices": len(choices),
            "answer_letter": answer,
            "keep": keep,
            "answer_index": answer_index,
        }


    def data_map_fn(self, batch):

        b = []
        for i in range(len(list(batch.values())[0])):
            example = {k: v[i] for k, v in batch.items()}
            b.append(example)
        batch = b

        l = []
        for example in batch:

            prompt, choices, answer = self.extract_example(example)
            formatted = self.mcqa_format(prompt, choices, answer)
            l.append(formatted)

        d = {}
        for k in l[0].keys():
            d[k] = [x[k] for x in l]

        input_ids, keep_input = self.tokenize(d["input_text"], "input")
        output_ids, keep_output = self.tokenize(d["output_text"], "output")

        keep = (
            keep_input & keep_output & np.array(d["keep"], dtype=bool)
        )

        d.update({
                "input_ids": input_ids,
                "output_ids": output_ids,
                "keep": keep,
        })

        return d

    
    def grade(self, batch, logits):
        logits = logits[:, self.prefix_length]

        logits = torch.index_select(logits, -1, self.torch_letter_ids)

        mask = (
            torch.arange(logits.shape[-1], device=logits.device).long()[None] >=
            batch["num_choices"][:, None]
        )
        logits = logits.masked_fill(mask, float("-inf"))

        pred_indices = torch.argmax(logits, dim=-1)

        return pred_indices == batch["answer_index"]



class MathBenchmark(BaseBenchmark):

    is_default = True

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
        max_examples: int | None = None,
    ):
        
        self.prefix_length = len(
            tokenizer(chat.NO_COT_PREFIX, add_special_tokens=False).input_ids
        )

        super().__init__(
            tokenizer,
            max_input_length,
            max_output_length,
            max_examples
        )


    def extract_example(self, example):
        """
        Should return prompt, choices, and answer (letter)
        """
        raise NotImplementedError("extract_example must be implemented by subclasses of MCQABenchmark")


    def math_format(self, prompt, answer):
        answer = answer.strip()

        input_text = chat.format_no_cot(
            prompt, "_", "_"
        )[0]
        output_text = chat.no_cot_format(answer)

        keep = True
        for c in answer:
            if c not in string.digits+".e":
                keep = False
                break

        return {
            "input_text": input_text,
            "output_text": output_text,
            "answer": answer,
            "keep": keep,
        }


    def data_map_fn(self, batch):

        b = []
        for i in range(len(list(batch.values())[0])):
            example = {k: v[i] for k, v in batch.items()}
            b.append(example)
        batch = b

        l = []
        for example in batch:

            prompt, answer = self.extract_example(example)
            formatted = self.math_format(prompt, answer)
            l.append(formatted)

        d = {}
        for k in l[0].keys():
            d[k] = [x[k] for x in l]

        input_ids, keep_input = self.tokenize(d["input_text"], "input")
        output_ids, keep_output = self.tokenize(d["output_text"], "output")

        keep = (
            keep_input & keep_output & np.array(d["keep"], dtype=bool)
        )

        d.update({
                "input_ids": input_ids,
                "output_ids": output_ids,
                "keep": keep,
        })

        return d

    
    def grade(self, batch, logits):

        target_ids = batch["output_ids"][:, self.prefix_length:]
        logits = logits[:, self.prefix_length:]

        pred_indices = torch.argmax(logits, dim=-1)

        correct = (pred_indices == target_ids) | (target_ids == self.tokenizer.pad_token_id)
        correct = correct.all(dim=-1)

        return correct


class arc_e(MCQABenchmark):

    name = "ARC-Easy"

    url = "allenai/ai2_arc"
    subset = "ARC-Easy"
    split = "test"

    def extract_example(self, example):
        return (
            example["question"],
            example["choices"]["text"],
            example["answerKey"],
        )


class arc_c(arc_e):

    name = "ARC-Challenge"
    subset = "ARC-Challenge"


class arc_e_train(arc_e):

    name = "ARC-E-Train"
    split = "train"

    is_default = False

class arc_c_train(arc_c):

    name = "ARC-C-Train"
    split = "train"

    is_default = False


class sciq(MCQABenchmark):
    
    name = "SciQ"

    url = "allenai/sciq"
    split = "test"

    def extract_example(self, example):
        prompt = example["question"]

        choices = [
            example["distractor1"],
            example["distractor2"],
            example["distractor3"],
            example["correct_answer"]
        ]

        answer = 'D'

        return prompt, choices, answer


class piqa(MCQABenchmark):

    name = "PIQA"

    url = "lighteval/piqa"
    split = "test"

    def extract_example(self, example):
        prompt = example["goal"]

        choices = [
            example["sol1"],
            example["sol2"],
        ]

        answer = 'A' if example["label"] == 0 else 'B'

        return prompt, choices, answer


class mmlu(MCQABenchmark):

    name = "MMLU"

    url = "lighteval/mmlu"
    subset = "all"
    split = "test"

    def extract_example(self, example):
        return (
            example["question"],
            example["choices"],
            string.ascii_uppercase[example["answer"]],
        )


class mmlu_pro(MCQABenchmark):

    name = "MMLU-Pro"

    url = "TIGER-Lab/MMLU-Pro"
    split = "test"

    def extract_example(self, example):
        return (
            example["question"],
            example["options"],
            example["answer"],
        )


class gpqa(MCQABenchmark):

    name = "GPQA"

    url = "Idavidrein/gpqa"
    subset = "gpqa_main"
    split = "train"

    def extract_example(self, example):
        return (
            example["Question"],
            [
                example["Correct Answer"],
                example["Incorrect Answer 1"],
                example["Incorrect Answer 2"],
                example["Incorrect Answer 3"],
            ],
            'A',
        )


class strategy_qa(MCQABenchmark):

    name = "StrategyQA"

    url = "tasksource/strategy-qa" 
    split = "train"

    def extract_example(self, example):

        prompt = ".".join(example["facts"]) + " " + example["question"]

        return (
            prompt,
            ["Yes", "No"],
            'A' if example["answer"] else 'B',
        )


class ar_lsat(MCQABenchmark):

    name = "AR-LSAT"

    url = 'olegbask/AR-LSAT'
    split = 'test'


    def extract_example(self, example):
        
        prompt = example["context"] + " " + example["question"]

        return (
            prompt,
            example["answers"],
            string.ascii_uppercase[example["label"]],
        )


class gsm8k(MathBenchmark):

    name = "GSM8K"

    url = "openai/gsm8k"
    subset = "main"
    split = "test"

    def extract_example(self, example):
        return (
            example["question"],
            example['answer'].split("####")[-1].strip(),
        )


class math500(MathBenchmark):

    name = "MATH-500"

    url = "HuggingFaceH4/MATH-500"
    split = "test"

    def extract_example(self, example):
        return (
            example["problem"],
            example["answer"].strip(),
        )


class minervamath(MathBenchmark):

    name = "Minerva-Math"

    url = "math-ai/minervamath"
    split = "test"

    def extract_example(self, example):
        return (
            example["question"],
            example["answer"].strip(),
        )


class amc23(MathBenchmark):

    name = "AMC-23"

    url = "math-ai/amc23"
    split = "test"

    def extract_example(self, example):
        return (
            example["question"],
            example["answer"].strip(),
        )


class aime2025(MathBenchmark):

    name = "AIME-2025"

    url = "MathArena/aime_2025"
    split = "train"

    def extract_example(self, example):
        return (
            example["problem"],
            str(example["answer"]),
        )


class aime2026(aime2025):

    name = "AIME-2026"

    url = "MathArena/aime_2026"


class svamp(MathBenchmark):

    name = "SVAMP"

    url = "MU-NLPC/Calc-svamp"
    subset = "default"
    split = "test"

    def extract_example(self, example):
        return (
            example["question"],
            example["result"].replace("_", "").strip(),
        )


class ManyICLBench(BaseBenchmark):

    name = "ManyICLBench"

    url = "launch/ManyICLBench"
    subset = None
    split = None
    context_length = None

    requires_batch_size_one = True


    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
        max_examples: int | None = None,
        **kwargs,
    ):
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                raise ValueError("ManyICLBench requires a tokenizer pad token or eos token.")
            tokenizer.pad_token = tokenizer.eos_token

        super().__init__(
            tokenizer,
            max_input_length,
            max_output_length,
            max_examples,
            **kwargs,
        )


    def get_dataset(self):
        available_subsets = self.get_subsets()
        selected_subsets = self.selected_values(
            getattr(self, "subsets", self.subset),
            available_subsets,
            "subset",
        )

        rows = {
            "input_text": [],
            "output_text": [],
            "subset": [],
            "seed": [],
            "context_length": [],
            "keep": [],
        }
        skip_targetless = "subset" not in self.benchmark_kwargs and "subsets" not in self.benchmark_kwargs

        for subset in selected_subsets:
            available_seeds = self.get_seeds(subset)
            selected_seeds = self.selected_values(
                getattr(self, "seeds", self.split),
                available_seeds,
                "seed",
            )
            targetless_seeds = [
                seed for seed in selected_seeds
                if not self.has_targets(subset, seed)
            ]
            if targetless_seeds and skip_targetless:
                selected_seeds = tuple(
                    seed for seed in selected_seeds
                    if seed not in targetless_seeds
                )
            elif targetless_seeds:
                self.add_dataset_rows(rows, subset, targetless_seeds[0], None, skip_targetless)

            if len(selected_seeds) == 0:
                continue

            available_context_lengths = self.get_context_lengths(subset, selected_seeds[0])
            context_length = self.context_length
            if context_length is None:
                context_length = available_context_lengths[0]
            elif context_length not in available_context_lengths:
                raise ValueError(
                    f"Unsupported ManyICLBench context length {context_length}. "
                    f"Choose one of {', '.join(available_context_lengths)}."
                )

            for seed in selected_seeds:
                self.add_dataset_rows(rows, subset, seed, context_length, skip_targetless)

        if len(rows["input_text"]) == 0:
            raise ValueError(f"No examples were loaded for {self.name}.")

        data = datasets.Dataset.from_dict(rows)

        data = data.map(
            self.data_map_fn,
            batched=True,
            batch_size=1,
            load_from_cache_file=False,
        )
        data = data.filter(
            self.data_filter_fn,
            batched=True,
            batch_size=BS,
            load_from_cache_file=False,
        )

        if len(data) == 0:
            raise ValueError(
                f"No {self.name} examples fit max_input_length={self.max_input_length} "
                f"and max_output_length={self.max_output_length}."
            )

        data = data.remove_columns("keep")

        self.dataset = data
        data = data.map(
            self.truncate_map_fn,
            batched=True,
            batch_size=1,
            load_from_cache_file=False,
        )
        self.dataset = None

        return data


    def get_subsets(self):
        return tuple(datasets.get_dataset_config_names(self.url))


    def get_seeds(self, subset):
        return tuple(datasets.get_dataset_split_names(self.url, subset))


    def get_context_lengths(self, subset, seed):
        data = datasets.load_dataset(
            self.url,
            name=subset,
            split=seed,
            streaming=False,
        )
        if len(data) != 1:
            raise ValueError(
                f"Expected one row for {self.name}:{subset}:{seed}, got {len(data)} rows."
            )

        row = data[0]
        return tuple(
            column for column, value in row.items()
            if isinstance(value, str)
        )


    def has_targets(self, subset, seed):
        data = datasets.load_dataset(
            self.url,
            name=subset,
            split=seed,
            streaming=False,
        )
        if len(data) != 1:
            raise ValueError(
                f"Expected one row for {self.name}:{subset}:{seed}, got {len(data)} rows."
            )

        return "Test Target" in data[0]


    def selected_values(self, selected, default, label):
        if selected is None:
            selected = default
        elif isinstance(selected, str):
            selected = (selected,)
        else:
            selected = tuple(selected)

        invalid = [value for value in selected if value not in default]
        if invalid:
            raise ValueError(
                f"Unsupported ManyICLBench {label}: {', '.join(invalid)}. "
                f"Choose from {', '.join(default)}."
            )

        return selected


    def add_dataset_rows(self, rows, subset, seed, context_length, skip_targetless):
        data = datasets.load_dataset(
            self.url,
            name=subset,
            split=seed,
            streaming=False,
        )

        if len(data) != 1:
            raise ValueError(
                f"Expected one row for {self.name}:{subset}:{seed}, got {len(data)} rows."
            )

        row = data[0]
        if "Test Target" not in row:
            if skip_targetless:
                return
            raise ValueError(
                f"{self.name} cannot be built directly from {self.url}: "
                f"{subset} stores MATH file names but not target text."
            )

        if context_length is None:
            context_lengths = self.get_context_lengths(subset, seed)
            context_length = context_lengths[0] if len(context_lengths) > 0 else None

        if context_length not in row:
            context_lengths = self.get_context_lengths(subset, seed)
            raise ValueError(
                f"Unsupported ManyICLBench context length {context_length}. "
                f"Choose one of {', '.join(context_lengths)}."
            )

        context = row[context_length]
        if not isinstance(context, str):
            raise ValueError(
                f"{self.name} expected text in column {context_length}, "
                f"got {type(context).__name__}."
            )

        test_data = row["Test Data"]
        test_targets = row["Test Target"]
        if len(test_data) != len(test_targets):
            raise ValueError(
                f"{self.name} has {len(test_data)} test prompts but "
                f"{len(test_targets)} targets."
            )

        if self.max_examples is not None:
            max_examples = min(self.max_examples, len(test_data))
            test_data = test_data[:max_examples]
            test_targets = test_targets[:max_examples]

        rows["input_text"].extend(context + prompt for prompt in test_data)
        rows["output_text"].extend(str(target) for target in test_targets)
        rows["subset"].extend([subset] * len(test_data))
        rows["seed"].extend([seed] * len(test_data))
        rows["context_length"].extend([context_length] * len(test_data))
        rows["keep"].extend([True] * len(test_data))


    def tokenize_target(self, text):
        ids = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.max_output_length + 1,
        ).input_ids

        keep = ids[:, -1] == self.tokenizer.pad_token_id

        return ids[:, :-1], keep


    def data_map_fn(self, batch):
        input_ids, keep_input = self.tokenize(batch["input_text"], "input")
        output_ids, keep_output = self.tokenize_target(batch["output_text"])

        keep = (
            keep_input & keep_output & np.array(batch["keep"], dtype=bool)
        )

        batch.update({
            "input_ids": input_ids,
            "output_ids": output_ids,
            "keep": keep,
        })

        return batch


    def collate_fn(self, batch):
        if len(batch) != 1:
            raise ValueError(f"{self.name} requires batch_size=1.")

        d = super().collate_fn(batch)
        pad_token_id = self.tokenizer.pad_token_id

        input_length = int((d["input_ids"][0] != pad_token_id).sum().item())
        output_length = int((d["output_ids"][0] != pad_token_id).sum().item())

        d["input_ids"] = d["input_ids"][:, :input_length]
        d["output_ids"] = d["output_ids"][:, :output_length]

        return d


    def grade(self, batch, logits):
        pad_token_id = self.tokenizer.pad_token_id
        target_ids = batch["output_ids"].to(logits.device)
        target_mask = target_ids != pad_token_id

        if logits.shape[1] == target_ids.shape[1]:
            pred_ids = torch.argmax(logits, dim=-1)
            return self.grade_predictions(pred_ids, target_ids, target_mask)

        pred_ids = torch.full_like(target_ids, pad_token_id)
        input_length = batch["input_ids"].shape[1]
        for i in range(target_ids.shape[0]):
            target_length = int(target_mask[i].sum().item())
            if target_length == 0:
                continue

            start = input_length - 1
            end = start + target_length
            if start < 0 or end > logits.shape[1]:
                raise ValueError(
                    f"{self.name} received logits with sequence length {logits.shape[1]}, "
                    f"but needs positions [{start}, {end})."
                )

            pred_ids[i, :target_length] = torch.argmax(logits[i, start:end], dim=-1)

        return self.grade_predictions(pred_ids, target_ids, target_mask)


    def grade_predictions(self, pred_ids, target_ids, target_mask):
        correct = []
        for i in range(target_ids.shape[0]):
            target_length = int(target_mask[i].sum().item())
            pred_text = self.tokenizer.decode(
                pred_ids[i, :target_length].detach().cpu().tolist(),
                skip_special_tokens=True,
            ).strip()
            target_text = self.tokenizer.decode(
                target_ids[i, :target_length].detach().cpu().tolist(),
                skip_special_tokens=True,
            ).strip()
            correct.append(pred_text == target_text)

        return torch.tensor(correct, device=pred_ids.device, dtype=torch.bool)


BENCHMARK_DICT = {
    cls[1].name: cls[1]
    for cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if issubclass(cls[1], BaseBenchmark) and cls[1].name is not None
}

DEFAULT_BENCHMARK_DICT = {
    name: cls
    for name, cls in BENCHMARK_DICT.items()
    if cls.is_default
}
