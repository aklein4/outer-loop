import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import math
from collections.abc import Sequence

import datasets
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

import balanced_assignment


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


INPUT_DATASET = "aklein4/single-turn-compilation-SmolLM2-1024"
OUTPUT_DATASET = "aklein4/few-shot-TrackStar"
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

MESSAGES_COLUMN = "messages"
SOURCE_COLUMN = "source"
CLUSTER_COLUMN = "cluster"
CLUSTER_MEAN_COSINE_SIMILARITY_COLUMN = "similarity_to_cluster_mean"
NUM_TOKENS_COLUMN = "num_tokens"

MAX_LENGTH = 1024
COLLECTION_DIM = 8192 + 1024
PROJECTION_DIM = 8192
BATCH_SIZE = 64

KMEANS_STEPS = 500
ASSIGNMENT_STEPS = 100
EPS = 1e-12

ASSISTANT_MASK_CHAT_TEMPLATE = """{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
' }}{% endif %}{% if message['role'] == 'assistant' %}{{ '<|im_start|>assistant
' }}{% generation %}{{ message['content'] }}{% endgeneration %}{{ '<|im_end|>
' }}{% else %}{{ '<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>' + '
' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}"""


DEFAULT_CONFIGS = [
    # "Goedel-LM--Goedel-Pset-v1--train",
    # "HuggingFaceH4--ultrachat_200k",
    # "HuggingFaceTB--smoltalk--train",
    # "HuggingFaceTB--smoltalk2--SFT",
    "Magpie-Align--Llama-3-Magpie-Air-3M-v0.1--train",
    "Magpie-Align--Llama-3-Magpie-Pro-1M-v0.1--train",
    "Magpie-Align--Magpie-Qwen2-Pro-200K-English--train",
    "Magpie-Align--Magpie-Qwen2.5-Coder-Pro-300K-v0.1--train",
    "Magpie-Align--Magpie-Qwen2.5-Math-Pro-300K-v0.1--train",
    "Magpie-Align--Magpie-Qwen2.5-Pro-1M-v0.1--train",
    "Open-Orca--OpenOrca--train",
    "OpenDataArena--ODA-Math-460k--train",
    "TIGER-Lab--MATH-plus--train",
    "TIGER-Lab--WebInstructSub--train",
    "allenai--sciq--train",
    "allenai--tulu-3-sft-mixture--train",
    "allenai--tulu-v1-sft-mixture--train",
    "allenai--tulu-v2-sft-mixture--train",
    "allenai--tulu-v3.1-mix-preview-4096-OLMoE--train",
    "arcee-ai--The-Tome--train",
    "facebook--natural_reasoning--train",
    "math-ai--StackMathQA--stackmathqa800k--train",
    "meta-math--MetaMathQA--train",
    "microsoft--orca-agentinstruct-1M-v1",
    "nampdn-ai--tiny-codes--train",
    "nvidia--AceMath-Instruct-Training-Data",
    "nvidia--AceReason-1.1-SFT--train",
    # "nvidia--Llama-Nemotron-Post-Training-Dataset--SFT",
    "nvidia--Nemotron-Instruction-Following-Chat-v1--chat_if",
    # "nvidia--Nemotron-Math-Proofs-v1--lean",
    "nvidia--Nemotron-Post-Training-Dataset-v2",
    "nvidia--Nemotron-PrismMath--train",
    "nvidia--Nemotron-SFT-Instruction-Following-Chat-v3",
    "nvidia--Nemotron-Science-v1",
    "nvidia--OpenCodeReasoning--split_0--split_0",
    "nvidia--OpenMathInstruct-1--train",
    "nvidia--OpenMathInstruct-2--train_2M",
    "nvidia--OpenScience--train",
    "nvidia--OpenScienceReasoning-2--train",
    "teknium--OpenHermes-2.5--train",
]


class GradientSketchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activations, outputs, mod):
        ctx.save_for_backward(activations)
        ctx.mod = mod
        return outputs
    @staticmethod
    def backward(ctx, grad_output):
        activations, = ctx.saved_tensors
        ctx.mod.backward_fn(activations, grad_output)
        return None, grad_output, None


class GradientSketchLinear(nn.Module):

    def __init__(self, linear, sketch_dim):
        super().__init__()
        self.linear = linear
        self.sketch_dim = sketch_dim

        self.activation_indices = torch.randint(linear.in_features, (sketch_dim,), device=DEVICE)
        self.gradient_indices = torch.randint(linear.out_features, (sketch_dim,), device=DEVICE)

        self.G = []


    def forward(self, x):
        y = self.linear(x)
        activations = x[..., self.activation_indices]
        return GradientSketchFunction.apply(activations, y, self)

    def backward_fn(self, activations, grad_output):
        grad_output = grad_output[..., self.gradient_indices]
        g = activations.float() * grad_output.float()
        g = g.sum(1)
        self.G.append(g.detach())

    @torch.no_grad()
    def get_G(self):
        out = torch.cat(self.G, dim=0)
        self.G = []
        return out


class GradientSketchModel(nn.Module):

    def __init__(self, model, args):
        super().__init__()
       
        self.model = model
        self.collection_dim = args.collection_dim

        num_linears = 0
        for module in model.model.modules():
           if isinstance(module, nn.Linear):
               num_linears += 1
        sketch_dim_per = math.ceil(self.collection_dim / num_linears)
        
        self.total_dim = sketch_dim_per * num_linears
        self.total_indices = torch.randperm(self.total_dim, device=DEVICE)[:args.collection_dim]

        # replace all linears in model.model with sketching versions
        for name, module in list(model.model.named_modules()):
            for child_name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    sketch = GradientSketchLinear(child, sketch_dim_per)
                    setattr(module, child_name, sketch)


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    @torch.no_grad()
    def get_G(self):
        G = []
        for module in self.model.model.modules():
            if isinstance(module, GradientSketchLinear):
                G.append(module.get_G())
        G = torch.cat(G, dim=-1)
        G = G[:, self.total_indices]
        return G
    

@torch.no_grad()
def balanced_kmeans(x, k, steps=KMEANS_STEPS, assignment_steps=ASSIGNMENT_STEPS):
    n = len(x)

    k = min(k, n)
    if k <= 1:
        return [0] * n
    if k == n:
        return list(range(n))
    if n % k != 0:
        raise ValueError(f"number of samples ({n}) must be divisible by number of clusters ({k}) for balanced k-means")

    centers = x[torch.randperm(n, device=DEVICE)[:k]].clone()

    per_cluster = n // k
    score_gib = n * k * x.element_size() / 2**30
    assignment_gib = score_gib * 4
    print(
        f"Balanced k-means: n={n:_}, k={k:_}, per_cluster={per_cluster}, "
        f"score_matrix={score_gib:.2f} GiB, assignment_working_set~{assignment_gib:.2f} GiB"
    )

    with tqdm(range(steps), desc="Balanced k-means", leave=True) as progress:
        for _ in progress:

            scores = x @ centers.T

            order, auction_iterations = balanced_assignment.balanced_assignment(scores.contiguous(), assignment_steps)
            progress.set_postfix(auction_iters=int(auction_iterations))
            labels = torch.empty(n, dtype=torch.long, device=x.device)
            labels[order.long()] = torch.arange(k, device=x.device).repeat_interleave(per_cluster)

            new_centers = torch.zeros_like(centers)
            new_centers.index_add_(0, labels, x)
            counts = torch.bincount(labels, minlength=k).clamp_min(1).to(new_centers.dtype)
            new_centers = F.normalize(new_centers / counts[:, None], dim=1)
            if torch.allclose(new_centers, centers):
                progress.write("Balanced k-means converged")
                break
            centers = new_centers

    return labels.cpu().tolist()


def tokenize_messages(
    messages: Sequence[Sequence[dict[str, str]]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
):
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_dict=True,
        return_assistant_tokens_mask=True,
    )

    assistant_mask = encoded.pop("assistant_masks", None)
    if assistant_mask is None:
        assistant_mask = encoded.pop("assistant_tokens_mask", None)
    if assistant_mask is None:
        raise RuntimeError("tokenizer did not return assistant token masks")

    return (
        encoded.input_ids.to(DEVICE),
        assistant_mask.to(DEVICE),
    )


def assistant_response_loss(logits, input_ids, assistant_mask):

    shift_logits = logits[:, :-1].float().contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = assistant_mask[:, 1:].float().contiguous()

    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)

    losses = (losses * shift_mask.float()).sum(1) / shift_mask.sum(1).clamp_min(1.0)
    
    return losses.sum()


def gradient_vectors(messages, model: GradientSketchModel, tokenizer, args):

    for start in tqdm(range(0, len(messages), args.batch_size), desc="Gradient vectors"):
        
        batch_messages = messages[start : start + args.batch_size]
        input_ids, assistant_mask = tokenize_messages(batch_messages, tokenizer, args.max_length)
        
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
            outputs = model(input_ids)
            loss = assistant_response_loss(outputs.logits, input_ids, assistant_mask)
        loss.backward()

    G = model.get_G()
    model.zero_grad(True)

    return G


@torch.no_grad()
def whiten_vectors(x, args):
    x = x.float()

    x = F.normalize(x, dim=1, eps=EPS) # this preserves more spectra
    x = x * torch.rsqrt(x.pow(2).mean(dim=0) + EPS)

    hessian = (x.T @ x) / (len(x) - 1)
    eig_vals, eig_vecs = torch.linalg.eigh(hessian.double())
    eig_vals, eig_vecs = eig_vals.float(), eig_vecs.float()

    eig_vals = eig_vals[-args.projection_dim:]
    eig_vecs = eig_vecs[:, -args.projection_dim:]

    inv_sqrt = torch.rsqrt(eig_vals.clamp_min(0) + EPS)

    x = (x @ eig_vecs) * inv_sqrt
    x = F.normalize(x, dim=1)
    
    if not torch.isfinite(x).all():
        raise ValueError("normalized vectors contain non-finite values")
    
    return x


@torch.no_grad()
def get_cluster_cosine_similarities(x, labels):
    if len(x) != len(labels):
        raise ValueError(f"got {len(x)} vectors but {len(labels)} cluster labels")

    x = x.float()
    labels = torch.tensor(labels, dtype=torch.long, device=x.device)
    num_clusters = int(labels.max().item()) + 1
    means = torch.zeros(num_clusters, x.shape[1], dtype=x.dtype, device=x.device)
    means.index_add_(0, labels, x)
    counts = torch.bincount(labels, minlength=num_clusters).clamp_min(1).to(x.dtype)
    means = means / counts[:, None]

    similarities = F.cosine_similarity(x, means[labels], dim=1, eps=EPS)
    if not torch.isfinite(similarities).all():
        raise ValueError("cluster cosine similarities contain non-finite values")
    return similarities.tolist()


def cluster_piece(piece_name, ds, model, tokenizer, args, cluster_offset):
    
    ds = ds.select(range(args.cluster_size * (len(ds) // args.cluster_size)))
    print(f"{piece_name}: {len(ds):_} examples after trimming to cluster size")
    if len(ds) < args.cluster_size:
        print(f"Skipping {piece_name} because it has fewer examples than the cluster size ({len(ds)} < {args.cluster_size})")
        return None, cluster_offset
    
    ds = ds.sort(NUM_TOKENS_COLUMN, reverse=True)

    vectors = gradient_vectors(
        ds[MESSAGES_COLUMN], model, tokenizer, args
    )

    vectors = whiten_vectors(vectors, args)
    if len(vectors) != len(ds):
        raise ValueError(f"got {len(vectors)} vectors but expected {len(ds)}")
    
    if len(vectors) % args.cluster_size != 0:
        raise ValueError(f"number of vectors ({len(vectors)}) is not divisible by cluster size ({args.cluster_size})")
    labels = balanced_kmeans(
        vectors,
        len(vectors) // args.cluster_size,
        steps=args.kmeans_steps,
        assignment_steps=args.assignment_steps,
    )

    cluster_similarities = get_cluster_cosine_similarities(vectors, labels)
    labels = [label + cluster_offset for label in labels]

    ds = ds.add_column(CLUSTER_COLUMN, labels)
    ds = ds.add_column(CLUSTER_MEAN_COSINE_SIMILARITY_COLUMN, cluster_similarities)

    return ds, max(labels) + 1


def select_unique_examples(ds):

    def example_key(messages):
        return tuple((turn.get("role", ""), turn.get("content", "")) for turn in messages)

    seen = set()
    keep = []
    for i, messages in enumerate(ds[MESSAGES_COLUMN]):

        key = example_key(messages)

        if key not in seen:
            seen.add(key)
            keep.append(i)

    return ds.select(keep)


def split_source_subset(ds, source, args):
    ds = select_unique_examples(ds)

    if args.max_piece_size is None or len(ds) <= args.max_piece_size:
        return [source], [ds]

    names = []
    pieces = []
    for start in range(0, len(ds), args.max_piece_size):
        end = min(start + args.max_piece_size, len(ds))

        piece = ds.select(range(start, end))

        source_name = f"{source}/{len(pieces):04d}"
        piece = piece.remove_columns([SOURCE_COLUMN])
        piece = piece.add_column(SOURCE_COLUMN, [source_name] * len(piece))

        names.append(source_name)
        pieces.append(piece)
    
    return names, pieces


def cluster_subset(name, ds, model, tokenizer, args):
    if SOURCE_COLUMN not in ds.column_names:
        raise ValueError(f"{name} is missing required {SOURCE_COLUMN!r} column")

    clustered = []
    cluster_offset = 0
    sources = sorted(ds.unique(SOURCE_COLUMN))

    print(f"{name}: clustering {len(sources):_} sources")

    for source in sources:
        
        source_ds = ds if len(sources) == 1 else ds.filter(lambda row, s=source: [sc == s for sc in row[SOURCE_COLUMN]], batched=True, batch_size=args.batch_size)
        piece_names, pieces = split_source_subset(source_ds, source, args)

        print(f"{name}/{source}: split into {len(pieces)} pieces")

        for piece_name, piece in zip(piece_names, pieces):

            print(f"{name}/{source}: clustering piece {piece_name} with {len(piece):_} examples")

            piece, cluster_offset = cluster_piece(
                piece_name,
                piece,
                model,
                tokenizer,
                args,
                cluster_offset,
            )
            if piece is not None:
                clustered.append(piece)

    if not clustered:
        print(f"{name}: no clustered examples")
        return None

    return datasets.concatenate_datasets(clustered).sort(CLUSTER_COLUMN)


def load_tokenizer(args) -> PreTrainedTokenizerBase:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "right"
    tokenizer.chat_template = ASSISTANT_MASK_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation="flash_attention_2",
    ).to(DEVICE)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.model.embed_tokens.weight.requires_grad_(True)
    model = GradientSketchModel(model, args)
    model = torch.compile(model, fullgraph=False)
    return model


def main():

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = load_tokenizer(args)
    model = load_model(args)

    print(
        f"Using {args.model_name}: collection_dim={args.collection_dim}, projection_dim={args.projection_dim}"
    )

    configs = args.configs or datasets.get_dataset_config_names(args.input_dataset)
    for i, config in enumerate(configs):
        print(f"\n === [{i + 1}/{len(configs)}] Clustering {config} === ")
        
        ds = datasets.load_dataset(args.input_dataset, config, split="train")
        ds = ds.shuffle(seed=args.seed)
        if args.size_limit is not None:
            ds = ds.select(range(args.size_limit))

        ds = cluster_subset(config, ds, model, tokenizer, args)
        if ds is None:
            continue

        print(f"{config}: {len(ds):_} clustered examples, {len(ds.unique(CLUSTER_COLUMN)):_} clusters")
        
        if not args.dry_run:
            ds.push_to_hub(args.output_dataset, config_name=config, split="train", private=args.private)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dataset", default=INPUT_DATASET)
    parser.add_argument("--output-dataset", default=OUTPUT_DATASET)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--cluster-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--collection-dim", type=int, default=COLLECTION_DIM)
    parser.add_argument("--projection-dim", type=int, default=PROJECTION_DIM)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--kmeans-steps", type=int, default=KMEANS_STEPS)
    parser.add_argument("--assignment-steps", type=int, default=ASSIGNMENT_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size-limit", type=int)
    parser.add_argument("--max-piece-size", type=int, default=int((3/4)*2**19)) # 2^19=512K
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    main()
