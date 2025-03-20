import fnmatch
import json
import pathlib
import os
import datasets
import torch
import transformers
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForCausalLM, 
    AutoTokenizer, 
    HfArgumentParser
)
from gen_eval.arguments import EvalArguments
from gen_eval.evaluator import Evaluator
from gen_eval.tasks import ALL_TASKS

# these imports are needed to avoid potential
# circular import? see https://github.com/dask/distributed/issues/4168
import multiprocessing
import multiprocessing.popen_fork
import multiprocessing.managers

class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice

def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_args():
    parser = HfArgumentParser(EvalArguments)
    parser.add_argument(
        "--model",
        default="codeparrot/codeparrot-small",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        help="The path to the used tokenizer."
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision to use",
    )
    parser.add_argument(
        "--use_auth_token",
        default=None,
        type=none_or_str,
        help="Use the token generated when running `huggingface-cli login` (necessary for private model).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Use a model with custom code, this requires executing code by the author of the model.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        choices=MultiChoice(ALL_TASKS),
        help=f"Evaluation tasks from {ALL_TASKS}",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation on each worker, can be larger for HumanEval",
    )
    parser.add_argument(
        "--max_length_generation",
        type=int,
        default=512,
        help="Maximum length of generated sequence (prompt+generation)",
    )
    parser.add_argument(
        "--max_new_tokens_generation",
        type=int,
        default=None,
        help="Maximum length of generated sequence (generation)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        help="Model precision, from: fp32, fp16 or bf16",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4bit",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    parser.add_argument(
        "--postprocess",
        action="store_false",
        help="Postprocess model outputs before execution, always on except during generation tests",
    )
    parser.add_argument(
        "--allow_code_execution",
        action="store_true",
        help="Allow code evaluation to execute external/untrusted Python code on your machine",
    )
    parser.add_argument(
        "--generation_only",
        action="store_true",
        help="Do code generation but no evaluation",
    )
    parser.add_argument(
        "--load_generations_path",
        type=str,
        default=None,
        help="Path of file with previously generated solutions, if provided generation is skipped and only evaluation is done",
    )
    parser.add_argument(
        "--load_data_path",
        type=str,
        default=None,
        help="Path of additional data to load for the tasks",
    )
    parser.add_argument(
        "--metric_output_path",
        type=str,
        default="evaluation_results.json",
        help="Path to save the results",
    )
    parser.add_argument(
        "--save_generations",
        action="store_true",
        help="Whether to save code generations",
    )
    parser.add_argument(
        "--save_generations_path",
        type=str,
        default="generations.json",
        help="Path for saving the code generations",
    )
    parser.add_argument(
        "--save_references",
        action="store_true",
        help="Whether to save reference solutions/tests",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Number of processes to run evaluation",
    )
    parser.add_argument(
        "--auto_device_map", 
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--check_references",
        action="store_true",
        help="Don't run generation but benchmark groundtruth (useful for debugging)",
    )
    parser.add_argument(
        "--multi_byte_decoding",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--instruct_format",
        action="store_true",
        default=False
    )
    return parser.parse_args()


def pattern_match(patterns, source_list):
    """Returns a list containing all values of the source_list that
    match at least one of the patterns"""
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)

def get_gpus_max_memory(max_memory, num_gpus):
    max_memory = {i: max_memory for i in range(num_gpus)}
    print("Loading model via these GPUs & max memories: ", max_memory)
    return max_memory

def main():
    args = parse_args()
    print(args)
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()
    if args.num_processes:
        os.environ["HF_CODE_EVAL_NUM_PROC"] = str(args.num_processes)
    else:
        os.environ["HF_CODE_EVAL_NUM_PROC"] = str(max(1, multiprocessing.cpu_count() // 2))
    
    save_generation_path = pathlib.Path(args.save_generations_path)
    save_generation_path.parent.mkdir(parents=True, exist_ok=True) 
    metric_output_path = pathlib.Path(args.metric_output_path)
    metric_output_path.parent.mkdir(parents=True, exist_ok=True) 
    if args.tasks is None:
        task_names = ALL_TASKS
    else:
        task_names = sorted(pattern_match(args.tasks.split(","), ALL_TASKS))

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Selected Tasks: {task_names}")

    results = {}
    if args.load_generations_path:
        # here we don't generate code but only evaluate previously computed generations
        if accelerator.is_main_process:
            print("evaluation only mode")
        evaluator = Evaluator(accelerator, None, None, args)
        for task_name in task_names:
            generations, references = evaluator.load_generations(task_name)
            results[task_name] = evaluator.evaluate(task_name, generations, references)
    else:
        # here we generate code and save it (evaluation is optional but True by default)
        dict_precisions = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if args.precision not in dict_precisions:
            raise ValueError(
                f"Non valid precision {args.precision}, choose from: fp16, fp32, bf16"
            )

        model_kwargs = {
            "revision": args.revision,
            "trust_remote_code": args.trust_remote_code,
            "use_auth_token": args.use_auth_token if args.use_auth_token else False,
        }
        if args.load_in_8bit:
            print("Loading model in 8bit")
            model_kwargs["load_in_8bit"] = args.load_in_8bit
            model_kwargs["device_map"] = {"": accelerator.process_index}
        elif args.load_in_4bit:
            print("Loading model in 4bit")
            model_kwargs["load_in_4bit"] = args.load_in_4bit
            model_kwargs["device_map"] = {"": accelerator.process_index}
        else:
            model_kwargs["torch_dtype"] = dict_precisions[args.precision]
            print(f"Loading model in {args.precision}")

            if args.auto_device_map:
                model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            **model_kwargs,
        )
        model.eval()
        print(model)
        print("=====> Model params: {}".format(model.num_parameters()))
        print("=====> Model mode : {}".format("training" if model.training else "evaluation"))

        if args.tokenizer_path:
            tokenizer_path = args.tokenizer_path
        else:
            tokenizer_path = args.model

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            use_auth_token=args.use_auth_token if args.use_auth_token else False,
            truncation_side="right",
            padding_side="left",
            # disable fast tokenizer for now,
            # as it might behave inconsistently across HF versions
            use_fast=False,
        )

        if not tokenizer.eos_token:
            if tokenizer.bos_token:
                tokenizer.eos_token = tokenizer.bos_token
                print("bos_token used as eos_token")
            else:
                raise ValueError("No eos_token or bos_token found")

        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        # For supressing warnings in 4.34.0
        if not tokenizer.mask_token:
            tokenizer.mask_token = tokenizer.eos_token
        if not tokenizer.sep_token:
            tokenizer.sep_token = tokenizer.eos_token
        if not tokenizer.cls_token:
            tokenizer.cls_token = tokenizer.eos_token
        if not tokenizer.bos_token:
            tokenizer.bos_token = tokenizer.eos_token
        if not tokenizer.unk_token:
            tokenizer.unk_token = tokenizer.eos_token
        
        evaluator = Evaluator(accelerator, model, tokenizer, args)

        generations_by_task = {}
        references_by_task = {}
        for task_name in task_names:
            generations_by_task[task_name], references_by_task[task_name] = evaluator.generate_text(task_name)
            if args.generation_only:
                if accelerator.is_main_process:
                    print("generation mode only for Task : {}".format(task_name))
            else:
                results[task_name] = evaluator.evaluate(
                    task_name, 
                    generations_by_task[task_name],
                    references_by_task[task_name]
                )

        if args.generation_only:
            evaluator.save_generations(generations_by_task, references_by_task)
    
    # Save all args to config
    results["config"] = vars(args)
    if not args.generation_only:
        dumped = json.dumps(results, indent=2)
        if accelerator.is_main_process:
            print(dumped)

        with open(args.metric_output_path, "w") as f:
            f.write(dumped)


if __name__ == "__main__":
    main()
