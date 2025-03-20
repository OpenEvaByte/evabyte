"""
DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation

https://arxiv.org/pdf/2211.11501.pdf

DS-1000 is a code generation benchmark with a thousand data science questions spanning seven Python libraries that (1) reflects diverse, realistic, and practical use cases, (2) has a reliable metric, (3) defends against memorization by perturbing questions.

Homepage: https://ds1000-code-gen.github.io/
"""
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool
from dataclasses import dataclass
from typing import List, Union
import numpy as np
import fcntl
import functools
import os
import json
import io
import itertools
import pathlib
import warnings
import zipfile

import requests
import tqdm

from gen_eval.base import Task

_CITATION = """
@article{Lai2022DS1000,
  title={DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation},
  author={Yuhang Lai and Chengxi Li and Yiming Wang and Tianyi Zhang and Ruiqi Zhong and Luke Zettlemoyer and Scott Wen-tau Yih and Daniel Fried and Sida Wang and Tao Yu},
  journal={ArXiv},
  year={2022},
  volume={abs/2211.11501}
}
"""

def report_acc(score_by_lib):
    """
    Report the percentage of correct programs computed from a `score_by_lib` dictionary
    Report will be splitted by library and by perturbation type

    Input:
    score_by_lib["Pandas"][0] is the ScoreRecord object of the Pandas/q0
    """

    num_problem = 0
    for lib in score_by_lib:
        num_problem += len(score_by_lib[lib])
    print(f"Total Questions: {num_problem}")

    avg_score_total = 0
    avg_score_dict = {}
    for lib in score_by_lib:
        avg_score_lib = 0
        for problem_id in range(len(score_by_lib[lib])):
            score_problem = sum(score_by_lib[lib][problem_id].test_results) / len(
                score_by_lib[lib][problem_id].test_results
            )
            avg_score_lib += score_problem
            avg_score_total += score_problem
        avg_score_lib /= len(score_by_lib[lib])
        print(f"{lib} Avg. Acc: {avg_score_lib:.2%}")
        avg_score_dict[lib] = avg_score_lib
    avg_score_total /= num_problem
    print(f"DS-1000 Avg. Acc: {avg_score_total:.2%}")
    avg_score_dict["all"] = avg_score_total

    result_by_pertb = defaultdict(list)
    perturbation_types = ["Origin", "Surface", "Semantic", "Difficult-Rewrite"]
    for lib in score_by_lib:
        for problem_id in range(len(score_by_lib[lib])):
            score_problem = np.mean(score_by_lib[lib][problem_id].test_results)
            for perturbation_type in perturbation_types:
                if perturbation_type == score_by_lib[lib][problem_id].perturbation_type:
                    result_by_pertb[perturbation_type].append(score_problem)
    print("---------------------------")
    print("Accuracy by perturbation type")
    for perturbation_type in perturbation_types:
        if len(result_by_pertb[perturbation_type]) > 0:
            pertb_avg_acc = np.mean(result_by_pertb[perturbation_type])
            print(f"{perturbation_type}: {pertb_avg_acc:.2%}")
        else:
            print(f"No {perturbation_type}")

    drop_by_pertb = defaultdict(list)
    perturbation_types = perturbation_types[1:]
    for lib in score_by_lib:
        for problem_id in range(len(score_by_lib[lib])):
            if "Origin" in score_by_lib[lib][problem_id].perturbation_type:
                continue
            score_problem = np.mean(score_by_lib[lib][problem_id].test_results)
            score_diff = score_problem - np.mean(
                score_by_lib[lib][
                    score_by_lib[lib][problem_id].perturbation_origin_id
                ].test_results
            )
            for perturbation_type in perturbation_types:
                if perturbation_type == score_by_lib[lib][problem_id].perturbation_type:
                    drop_by_pertb[perturbation_type].append(score_diff)
    print("---------------------------")
    print("Accuracy drop by perturbation type")
    for perturbation_type in perturbation_types:
        if len(drop_by_pertb[perturbation_type]) > 0:
            pertb_acc_drop = np.mean(drop_by_pertb[perturbation_type])
            print(f"{perturbation_type}: {pertb_acc_drop:.2%}")
        else:
            print(f"No {perturbation_type}")

    return avg_score_dict

def test_helper(prob_code_tuple):
    problem, code_list = prob_code_tuple
    test_results = []
    for code in code_list:
        test_results.append(problem.test(code))
    return test_results


@dataclass
class ScoreRecord:
    """
    A Record object containing the test results of all samples corresponding
    to a single question.

    Args:
    library: str, name of library a question is from
    problem_id: str
    perturbation_type: str, options among {Origin, Surface, Semantic, Difficult-Rewrite}
    perturbation_origin_id: str, id of the original question from which this question is
    perturbed
    test_results: List[int], each number represeting the test result of a single program sample

    use ``write_to_json'' Function to dump attributes into a json file
    """

    library: str
    problem_id: int
    perturbation_type: str
    perturbation_origin_id: int
    test_results: List[int]

    def write_to_json(self, path: Union[str, Path]):
        with open(path, "w", encoding="UTF-8") as f:
            result = dict()
            result["library"] = self.library
            result["perturbation_type"] = self.perturbation_type
            result["id"] = self.problem_id
            result["original_id"] = self.perturbation_origin_id
            result["test_results"] = self.test_results
            json.dump(result, f)

def create_all_tasks():
    def create_task(key, mode):
        class DS1000(GeneralDS1000):
            def __init__(self):
                super().__init__(key, mode)

        return DS1000

    return {
        f"ds1000-{key.lower()}-{mode.lower()}": create_task(key, mode)
        for key in [
            "All",
            "Numpy",
            "Pandas",
            "Scipy",
            "Matplotlib",
            "Sklearn",
            "Tensorflow",
            "Pytorch",
        ]
        for mode in ["Completion", "Insertion"]
    }


class GeneralDS1000(Task):
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self, key, mode):
        super().__init__(
            stop_words=["</code>", "# SOLUTION END", "\nEND SOLUTION"],
            requires_execution=True
        )
        self._key = key
        self._mode = mode
        self._dir = pathlib.Path(__file__).parent / "ds"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._src = self._dir / "ds1000.py"
        self._data = self._dir / "ds1000_data"
        self._download_source()
        self._download_dataset()

    def _download_source(self):
        url = "https://github.com/HKUNLP/DS-1000/blob/49c1c543ada8b58138181333cdc62e613204efcf/ds1000.py?raw=true"
        lock = self._src.with_suffix(".lock")
        with open(lock, "w") as f_lock:
            fcntl.flock(f_lock, fcntl.LOCK_EX)
            if not self._src.exists():
                warnings.warn(f"DS-1000 source is being saved to {self._src}.")
                print("Downloading source code...")
                r = requests.get(url, stream=True)
                with open(self._src, "wb") as f_src:
                    f_src.write(r.content)
                open(self._src.parent / "__init__.py", "w").close()
                print("Done.")
            fcntl.flock(f_lock, fcntl.LOCK_UN)

    def _download_dataset(self):
        url = "https://github.com/HKUNLP/DS-1000/blob/49c1c543ada8b58138181333cdc62e613204efcf/ds1000_data.zip?raw=true"
        lock = self._data.with_suffix(".lock")
        with open(lock, "w") as f_lock:
            fcntl.flock(f_lock, fcntl.LOCK_EX)
            if not self._data.exists():
                warnings.warn(f"DS-1000 data is being saved to {self._data}.")
                print("Downloading dataset...")
                r = requests.get(url, stream=True)
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(self._dir)
                print("Done.")
            fcntl.flock(f_lock, fcntl.LOCK_UN)

    @functools.lru_cache()
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        from .ds.ds1000 import DS1000Dataset

        data = DS1000Dataset(self._data, mode=self._mode).data
        if self._key == "All":
            dataset = list(itertools.chain(*data.values()))
        else:
            dataset = data[self._key]
        return dataset

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str | dict[str: str]
        """
        if self._mode == "Completion":
            prompt = doc["prompt"]
            return prompt
        elif self._mode == "Insertion":
            prefix = ""
            suffix = ""
            insert_flag = False
            first_line_flag = True
            # extract prefix and suffix of the prompt
            for line in doc["prompt"].split("\n"):
                if "[insert]" in line:
                    insert_flag = True
                    continue
                if first_line_flag:
                    first_line_flag = False
                else:
                    line = "\n" + line
                if not insert_flag:
                    prefix += line
                else:
                    suffix += line

            prefix = f"{prefix.strip()}\n"
            suffix = f"\n{suffix.strip()}\n"
            return {"prefix": prefix, "suffix": suffix}
        else:
            raise ValueError(f"Invalid mode: {self._mode}")

    def get_reference(self, doc):
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["reference_code"]

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens, include_stop_tokens=False):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
                if include_stop_tokens:
                    # retain the stop token.
                    min_stop_index = min_stop_index + len(stop_token)
        return decoded_string[:min_stop_index]

    def trim_generation(self, generation, idx):
        """Remove any code beyond the current completion scope."""
        return self._stop_at_stop_token(generation, self.stop_words, include_stop_tokens=True)

    # def postprocess_generation(self, generation, idx):
    #     """Defines the postprocessing for a LM generation.
    #     :param generation: str
    #         code generation from LM
    #     :param idx: int
    #         index of doc in the dataset to which the generation belongs
    #         (not used for Humaneval-Task)
    #     """
    #     if self._mode == "Completion":
    #         prompt = self.get_prompt(self.get_dataset()[idx])["prompt"]
    #     elif self._mode == "Insertion":
    #         prompt = ""
    #     generation = generation[len(prompt) :]
    #     return prompt + self._stop_at_stop_token(generation, self.stop_words, include_stop_tokens=True)

    def postprocess_generation(self, generation, idx):
        if self._mode == "Completion":
            extracted_generation = self._stop_at_stop_token(
                generation, 
                self.stop_words, 
                include_stop_tokens=False
            )
        elif self._mode == "Insertion":
            extracted_generation = generation
        return extracted_generation

    def process_results(self, generations, references):
        """
        Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        We encourage to directly load the metric from `evaluate` library to keep the code concise.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        results = self.compute_metric(
            generations=generations, 
            references=references,
            num_procs=int(os.getenv("HF_CODE_EVAL_NUM_PROC", "1")),
        )
        return results
        # dataset = self.get_dataset()
        # num_correct = 0
        # print("Scoring generations...")
        # for i, ref in tqdm.tqdm(enumerate(references), total=len(references)):
        #     test = [doc for doc in dataset if doc["reference_code"] == ref][0]
        #     for gen in generations[i]:
        #         is_correct = test.test(gen)
        #         if is_correct:
        #             num_correct += 1
        # accuracy = num_correct / len(references) / len(generations[0])
        # return {f"mean pass@1 accuracy ({len(generations[0])} samples)": accuracy}

    @functools.lru_cache()
    def get_ds1000(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        from .ds.ds1000 import DS1000Dataset
        
        if self._key == "All":
            libs = [
                "Pandas",
                "Numpy",
                "Matplotlib",
                "Tensorflow",
                "Scipy",
                "Sklearn",
                "Pytorch",
            ]
        else:
            libs = [self._key]
        ds1000 = DS1000Dataset(self._data, libs=libs, mode=self._mode)
        return ds1000
    
    def compute_metric(
            self,
            generations, 
            references,
            num_procs: int = 16,
    ):
        """
        Use multiprocessing to test a certain model's generated codes, and dump the score records into json files

        `model` is a str, helping to specify the path where generated codes should be stored
        `mode` is a str, can be "Insertion" or "Completion"
        `num_procs` is the number of processes to set
        `output_dir` is the path where the generated codes are stored
        `source_dir` is the path where the dataset is stored

        a single generated code's path can be like `output_dir` / `model` / Pandas / `mode` / q0 / 0.py
        """
        ds1000 = self.get_ds1000()
        problem_code_pairs_grouped_by_lib = defaultdict(list)
        for i, ref in tqdm.tqdm(enumerate(references), total=len(references)):
            for lib, problems in ds1000.data.items():
                for problem in problems:
                    if problem["reference_code"] == ref:
                        problem_code_pairs_grouped_by_lib[lib].append((problem, generations[i]))

        score = defaultdict(list)
        for lib in ds1000.libs:
            lib_results = []
            problem_code_pair = problem_code_pairs_grouped_by_lib[lib]

            # sklearn requires fitting model and doesn't work well in multiprocessing
            if num_procs > 1 and lib != "Sklearn":
                with Pool(processes=num_procs) as pool:
                    for test_results in tqdm.tqdm(
                            pool.imap(test_helper, problem_code_pair),
                            total=len(ds1000[lib]),
                            desc=f"Executing test for {lib} questions",
                    ):
                        lib_results.append(test_results)
            else:
                for problem_code_pair in tqdm.tqdm(
                            problem_code_pair, 
                            desc=f"Executing test for {lib} questions",
                ):
                    lib_results.append(test_helper(problem_code_pair))

            for problem_id in range(len(ds1000[lib])):
                score[lib].append(
                    ScoreRecord(
                        library=lib,
                        problem_id=problem_id,
                        perturbation_type=ds1000[lib][problem_id]["perturbation_type"],
                        perturbation_origin_id=int(
                            ds1000[lib][problem_id]["perturbation_origin_id"]
                        ),
                        test_results=lib_results[problem_id],
                    )
                )
        # for lib in ds1000.libs:
        #     result_cache_path = Path(output_dir) / f"{model}_{mode}_result_cache" / lib
        #     os.makedirs(result_cache_path, exist_ok=True)
        #     for record in score[lib]:
        #         record.write_to_json(result_cache_path / f"{record.problem_id}.json")
        avg_score_dict = report_acc(score)
        return avg_score_dict

