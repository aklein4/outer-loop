
import os
from argparse import Namespace


from evaluate import main as evaluate_main
from utils import constants


SAVE_FOLDER = "ManyICLBench_results_20"

CONTEXT_LENGTHS = ["1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]
MAX_TEST_EXAMPLES = 20


def args_with_context(name, args, context_length):
    args_dict = vars(args).copy()

    args_dict["save_folder"] = os.path.join(SAVE_FOLDER, context_length, name)
    args_dict["benchmark_kwargs"] = {
        "ManyICLBench": {
            "context_length": context_length,
            "max_test_examples": MAX_TEST_EXAMPLES,
        },
    }

    return Namespace(**args_dict)


def main():

    icl_args = Namespace(
        checkpoint_url="aklein4/Llama-3.2-1B-TPU",
        checkpoint_step=0,
        checkpoint_not_strict=False,
        config_name=None,
        tokenizer="meta-llama/Llama-3.2-1B",
        max_input_length=256,
        max_output_length=512,
        batch_size=1,
        max_examples=None,
        no_autocast=False,
        save_folder=None,
        seed=42,
        model_kwargs={"cpu_logits": True},
        benchmark_kwargs=None,
        benchmarks=["ManyICLBench"],
    )

    oloop_args = Namespace(
        checkpoint_url="aklein4/outer-loop_oloop-hyper",
        checkpoint_step=500,
        checkpoint_not_strict=False,
        config_name=None,
        tokenizer="meta-llama/Llama-3.2-1B",
        max_input_length=256,
        max_output_length=512,
        batch_size=1,
        max_examples=None,
        no_autocast=True,
        save_folder=None,
        seed=42,
        model_kwargs={"cpu_logits": True, "verbose": True, "chunk_size": 1024},
        benchmark_kwargs=None,
        benchmarks=["ManyICLBench"],
    )

    lora_args = Namespace(
        checkpoint_url="aklein4/Llama-3.2-1B-TPU",
        checkpoint_step=0,
        checkpoint_not_strict=False,
        config_name="oloop-lora-llama3p2-1b",
        tokenizer="meta-llama/Llama-3.2-1B",
        max_input_length=256,
        max_output_length=512,
        batch_size=1,
        max_examples=None,
        no_autocast=True,
        save_folder=None,
        seed=42,
        model_kwargs={"cpu_logits": True, "verbose": True, "chunk_size": 1024},
        benchmark_kwargs=None,
        benchmarks=["ManyICLBench"],
    )

    sliding_args = Namespace(
        checkpoint_url="aklein4/Llama-3.2-1B-TPU",
        checkpoint_step=0,
        checkpoint_not_strict=False,
        config_name="oloop-disable-llama3p2-1b",
        tokenizer="meta-llama/Llama-3.2-1B",
        max_input_length=256,
        max_output_length=512,
        batch_size=1,
        max_examples=None,
        no_autocast=True,
        save_folder=None,
        seed=42,
        model_kwargs={"cpu_logits": True, "verbose": True, "chunk_size": 1024},
        benchmark_kwargs=None,
        benchmarks=["ManyICLBench"],
    )

    setups = {
        "Sliding": sliding_args,
        "ICL": icl_args,
        "LoRA": lora_args,
        "OLoop": oloop_args,
    }

    for cl in CONTEXT_LENGTHS:
        print(f"\n ===== Evaluating with context length {cl} ===== ")

        for name, args in setups.items():

            print(f"--- Setup: {name} ---\n")

            setup_args = args_with_context(name, args, cl)
            
            if os.path.exists(os.path.join(constants.LOCAL_DATA_PATH, setup_args.save_folder)):
                print(f"Results already exist for {name} with context length {cl} at {setup_args.save_folder}; skipping evaluation.\n")
                continue
            
            evaluate_main(setup_args)


if __name__ == "__main__":
    main()
