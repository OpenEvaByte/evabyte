"""Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation
https://arxiv.org/abs/2305.01210

The HumanEval and MBPP plus dataset

Homepage: https://github.com/evalplus/evalplus
"""

import json
import multiprocessing
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from warnings import warn

import numpy as np
from termcolor import cprint
from tqdm import tqdm

from evalplus.config import (
    DEFAULT_GT_TIME_LIMIT_FACTOR,
    DEFAULT_MIN_TIME_LIMIT,
)
from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
    load_solutions,
    write_jsonl
)
from evalplus.data.mbpp import mbpp_serialize_inputs
from evalplus.eval import (
    PASS,
    compatible_eval_result,
    estimate_pass_at_k,
)
from evalplus.evaluate import (
    get_groundtruth,
    check_correctness
)
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from gen_eval.tasks.humaneval import GeneralHumanEval

_CITATION = """
@inproceedings{evalplus,
  title = {Is Your Code Generated by Chat{GPT} Really Correct? Rigorous Evaluation of Large Language Models for Code Generation},
  author = {Liu, Jiawei and Xia, Chunqiu Steven and Wang, Yuyao and Zhang, Lingming},
  booktitle = {Thirty-seventh Conference on Neural Information Processing Systems},
  year = {2023},
  url = {https://openreview.net/forum?id=1qvx610Cu7},
}
"""

def evaluate(
    dataset: str,
    samples: Optional[str] = None,
    base_only: bool = False,
    parallel: Optional[int] = None,
    i_just_wanna_run: bool = False,
    test_details: bool = False,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
    mini: bool = False,
    noextreme: bool = False,
    version: str = "default",
):
    assert samples is not None, "No samples provided"

    n_workers = parallel or max(1, multiprocessing.cpu_count() // 2)

    if os.path.isdir(samples):
        result_path = os.path.join(samples, "eval_results.json")
    else:
        assert samples.endswith(".jsonl")
        result_path = samples.replace(".jsonl", "_eval_results.json")

    if os.path.isfile(result_path) and not i_just_wanna_run:
        print(f"Load from previous results from {result_path}")
        with open(result_path, "r") as f:
            results = json.load(f)

        results = compatible_eval_result(results)
    else:
        if dataset == "humaneval":
            problems = get_human_eval_plus(
                mini=mini, noextreme=noextreme, version=version
            )
            dataset_hash = get_human_eval_plus_hash(
                mini=mini, noextreme=noextreme, version=version
            )
            expected_output = get_groundtruth(problems, dataset_hash, [])
        elif dataset == "mbpp":
            problems = get_mbpp_plus(mini=mini, noextreme=noextreme, version=version)
            dataset_hash = get_mbpp_plus_hash(
                mini=mini, noextreme=noextreme, version=version
            )
            expected_output = get_groundtruth(
                problems,
                dataset_hash,
                MBPP_OUTPUT_NOT_NONE_TASKS,
            )

        results = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "hash": dataset_hash,
            "eval": {},
        }

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            completion_id = Counter()
            n_samples = 0
            eval_results = defaultdict(list)  # task_id ->
            remainings = set()

            print("Reading samples...")
            for sample in tqdm(load_solutions(samples)):
                task_id = sample["task_id"]
                if task_id not in problems:
                    warn(
                        f"Task {task_id} is found in the samples but not found in the dataset"
                    )
                    continue
                solution = (
                    sample["solution"]
                    if "solution" in sample
                    else problems[task_id]["prompt"] + sample["completion"]
                )
                remainings.add(sample["_identifier"])
                args = (
                    dataset,
                    completion_id[task_id],
                    problems[task_id],
                    solution,
                    expected_output[task_id],
                    base_only,
                    not test_details,  # fast_check
                    sample["_identifier"],
                    min_time_limit,
                    gt_time_limit_factor,
                )
                futures.append(executor.submit(check_correctness, *args))
                completion_id[task_id] += 1
                n_samples += 1

            assert n_samples == len(remainings), "Missing problems in unfinished"
            assert len(completion_id) == len(problems), "Missing problems in samples"

            def stucking_checker():
                while remainings:
                    last_size = len(remainings)
                    time.sleep(20)
                    if last_size != len(remainings) or len(remainings) == 0:
                        continue
                    # Potential stucking
                    warn("No samples had finished testing in the last 20s")
                    warn(f"{len(remainings)} samples to be tested: {remainings}")

            threading.Thread(target=stucking_checker).start()

            for future in tqdm(as_completed(futures), total=n_samples):
                result = future.result()
                remainings.remove(result["_identifier"])
                eval_results[result["task_id"]].append(result)

        # sort the results for each problem by completion_id
        for task_id, task_results in eval_results.items():
            task_results.sort(key=lambda x: x["completion_id"])
            results["eval"][task_id] = []
            for res in task_results:

                def get_failed_tests(stat, details, inputs) -> List[Any]:
                    if stat == PASS or not details:
                        return []

                    if test_details:
                        return [
                            inputs[i] for i in range(len(details)) if not details[i]
                        ]

                    # else => simply return the only and the last fail test
                    return [inputs[len(details) - 1]]

                base_stat, base_details = res["base"]
                base_fail_tests = get_failed_tests(
                    base_stat, base_details, problems[task_id]["base_input"]
                )

                # initialize plus tests
                plus_stat = None
                plus_fail_tests = []

                # with plus tests
                if not base_only:
                    plus_stat, plus_details = res["plus"]
                    plus_fail_tests = get_failed_tests(
                        plus_stat, plus_details, problems[task_id]["plus_input"]
                    )

                if dataset == "mbpp":
                    base_fail_tests = mbpp_serialize_inputs(task_id, base_fail_tests)
                    plus_fail_tests = mbpp_serialize_inputs(task_id, plus_fail_tests)

                results["eval"][task_id].append(
                    {
                        "task_id": task_id,
                        "solution": res["solution"],
                        "base_status": base_stat,
                        "plus_status": plus_stat,
                        "base_fail_tests": base_fail_tests,
                        "plus_fail_tests": plus_fail_tests,
                    }
                )

    # Calculate pass@k.
    total = np.array([len(r) for r in results["eval"].values()])
    base_correct = []
    new_correct = []

    for res in results["eval"].values():
        bc = sum([r["base_status"] == PASS for r in res])
        base_correct.append(bc)
        if not base_only:
            new_correct.append(
                sum(
                    [
                        res[i]["base_status"] == res[i]["plus_status"] == PASS
                        for i in range(len(res))
                    ]
                )
            )
    base_correct = np.array(base_correct)

    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, base_correct, k).mean()
        for k in [1, 10, 100]
        if total.min() >= k
    }
    cprint(f"{dataset} (base tests)", "red")
    results["pass_at_k_base"] = pass_at_k
    for k, v in pass_at_k.items():
        cprint(f"{k}:\t{v:.3f}", "red")

    if new_correct:
        cprint(f"{dataset}+ (base + extra tests)", "green")
        pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(total, np.array(new_correct), k).mean()
            for k in [1, 10, 100]
            if (total >= k).all()
        }
        results["pass_at_k_plus"] = pass_at_k
        for k, v in pass_at_k.items():
            cprint(f"{k}:\t{v:.3f}", "green")

    if not os.path.isfile(result_path):
        with open(result_path, "w") as f:
            json.dump(results, f)


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {"humaneval_plus": GeneralHumanEvalPlus, "mbpp_plus": GeneralMBPPPlus}

class GeneralHumanEvalPlus(GeneralHumanEval):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self):
        super().__init__()
        self.dataset = list(get_human_eval_plus().values())
        self.dataset_name = "humaneval"

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point
    
    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        assert len(generations) == len(self.dataset)
        # get the directory
        samples_for_eval_path = os.path.join(
            os.path.dirname(os.getenv("RUN_STATS_SAVE_PATH", "")),
            "evalplus_samples.jsonl"
        )
        
        # delete the result file if exists
        result_path = samples_for_eval_path.replace(".jsonl", "_eval_results.json")
        if os.path.isfile(result_path):
            os.remove(result_path)

        samples_for_eval = [
            dict(
                task_id = self.dataset[idx]["task_id"],
                solution = gen
            )
            for idx, generation in enumerate(generations)
            for gen in generation
        ]
        write_jsonl(samples_for_eval_path, samples_for_eval)
        evaluate(
            dataset=self.dataset_name,
            samples=samples_for_eval_path,
            base_only=False,
            parallel=int(os.getenv("HF_CODE_EVAL_NUM_PROC", "1")),
            i_just_wanna_run=False,
            test_details=False,
            min_time_limit=1,
            gt_time_limit_factor=4.0,
            mini=False,
            noextreme=False,
        )
        return generations

class GeneralMBPPPlus(GeneralHumanEvalPlus):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self):
        super().__init__()
        self.stop_words = ["\nclass", "\n#", "\n@", "\nprint", "\nif", "\n```"]
        self.dataset = list(get_mbpp_plus().values())
        self.dataset_name = "mbpp"

    def get_instruct_prompt(self, doc, tokenizer):
        _MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

        response_prefix = "Here is a solution to this programming problem:"

        task_prompt = f"""{doc["prompt"].strip()}\n"""
        response = f"""{response_prefix}\n```python\n{_MAGIC_SPLITTER_}```"""
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": task_prompt},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
            add_generation_prompt=False
        ).split(_MAGIC_SPLITTER_)[0]
        if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
            prompt = prompt[len(tokenizer.bos_token):]
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        test_func = doc["assertion"]
        return test_func
