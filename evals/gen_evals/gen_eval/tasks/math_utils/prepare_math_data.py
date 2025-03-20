# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import random
import tarfile
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path
import re
from math_grader import normalize_answer_string

DOWNLOAD_LINK = "https://people.eecs.berkeley.edu/~hendrycks/MATH.tar"

def _post_fix(problem_id, soln_string):
    """Post fixing some answer strings"""
    if problem_id == "test/intermediate_algebra/78.json":
        soln_string = re.sub(r"\\(\d+)", r"\1", soln_string)

    if problem_id == "train/number_theory/7115.json":
        return "A"

    if problem_id == "train/number_theory/1012.json":
        return "E"

    if problem_id == "train/prealgebra/666.json":
        return "125"

    if problem_id == "train/intermediate_algebra/172.json":
        return "two lines"

    if problem_id == "train/prealgebra/1691.json":
        return "1.85"

    if problem_id == "train/geometry/6177.json":
        return "C"

    if problem_id == "train/number_theory/7117.json":
        return "A"

    if problem_id == "train/geometry/6202.json":
        return "D"

    if problem_id == "train/precalculus/268.json":
        return "A"

    return soln_string


def _post_fix_multi_answer(problem_id, results):
    """Fixing cases where there are multiple boxed entries."""

    if problem_id == "train/prealgebra/452.json":
        # Two ptions are mathematically equivalent
        return results[0]

    if problem_id == "train/algebra/1771.json":
        return ";".join(results)

    if problem_id == "train/algebra/152.json":
        return ";".join(results)

    if problem_id == "train/algebra/2156.json":
        # Same fraction but without antlr-4.11 can't verify if the two are equal
        return results[-1]

    if problem_id == "train/intermediate_algebra/1609.json":
        # Same fraction but without antlr-4.11 can't verify if the two are equal
        return results[-1]

    if problem_id == "train/precalculus/865.json":
        # The question has 2 answers, we just choose the last answer for now
        # TODO - Fix handling of OR questions
        return results[-1]

    if problem_id == "train/precalculus/982.json":
        # Question has many answers since it's a phase shift question.
        # Choosing the rightmost answer for now.
        # TODO - Fix handling of OR questions
        return results[-1]

    if problem_id == "train/precalculus/1149.json":
        # This question has two solutions, each being 30,150. We can pick any of the results entries
        return results[-1]

    if problem_id == "train/number_theory/837.json":
        # The two answers are for 24 hr clock vs 12 hr clock. Choosing the 24 version
        return results[0]

    if problem_id == "train/intermediate_algebra/396.json":
        # Or question, picking the rightmost answer
        return results[-1]

    if problem_id == "train/counting_and_probability/955.json":
        # The first boxed entry is an intermediate step
        return results[-1]

    # Test set fixes
    if problem_id == "test/prealgebra/1088.json":
        # Two solutions are mathematically equivalent
        return results[0]

    if problem_id == "test/algebra/1197.json":
        # The first entry is an intermediate result
        return results[-1]

    if problem_id == "test/geometry/66.json":
        # The two entries are same, choosing the first one expressed in frac
        return results[0]

    if problem_id == "test/geometry/1125.json":
        # Both are 0.25, choosing the first one
        return results[0]

    if problem_id == "test/prealgebra/1407.json":
        # There are intermediate values which are not answers
        return results[-1]

    if problem_id == "test/prealgebra/224.json":
        # Two answers are same, choosing rightmost
        return results[-1]

    if problem_id == "test/prealgebra/177.json":
        # The answer is 12 the last entry
        return results[-1]

    if problem_id == "test/number_theory/459.json":
        # Two answers are same, choosing rightmost
        return results[-1]

    if problem_id == "test/intermediate_algebra/702.json":
        # OR question. Choosing the rightmost answer
        return results[-1]

    if problem_id == "test/intermediate_algebra/25.json":
        # OR question. Choosing the rightmost answer
        return results[-1]

    if problem_id == "test/intermediate_algebra/747.json":
        # OR question. Choosing the rightmost answer
        return results[-1]

    return ",".join(results)


def _fix_solution(problem_id, ref_soln):
    if problem_id == "train/algebra/24014.json":
        return ref_soln.replace("$\\boxed 2$", "$\\boxed{2}$")

    if problem_id == "train/algebra/25040.json":
        return ref_soln.replace("\\boxed 9$", "\\boxed{9}$")

    if problem_id == "train/algebra/535.json":
        # Original soln
        # $\\log_24=\\boxed{2}$, so $\\log_2(4^2) = \\log_2((2^2)^2) = \\log_2 (2^4) = \\boxed{4}$
        return "$\\log_24=2$, so $\\log_2(4^2) = \\log_2((2^2)^2) = \\log_2 (2^4) = \\boxed{4}$"

    if problem_id == "train/geometry/892.json":
        return ref_soln.replace("\\boxed{144}", "144")

    if problem_id == "train/number_theory/7041.json":
        return ref_soln.replace("\\boxed{j-i \\equiv 0 \\pmod{6}}", "j-i \\equiv 0 \\pmod{6}")

    if problem_id == "train/intermediate_algebra/1266.json":
        return ref_soln.replace(
            "(x^2-\\boxed{\\phantom{09}})(x^2-\\boxed{\\phantom{25}})", "(x^2-\\phantom{09})(x^2-\\phantom{25})"
        )

    return ref_soln

def find_boxed_entries(answer_str):
    stack = []
    results = []
    i = 0

    while i < len(answer_str):
        if answer_str[i : i + 7] == '\\boxed{':
            stack.append(i + 7)
            i += 7
        elif answer_str[i] == '{':
            if stack:
                stack.append(i + 1)
            i += 1
        elif answer_str[i] == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    results.append(answer_str[start:i])
            i += 1
        else:
            i += 1

    if len(results) == 0:
        raise ValueError("Not enough boxed entries")
    else:
        results = [normalize_answer_string(result) for result in results]

    if len(results) == 1:
        # Single boxed entry, trivial case
        return results

    else:
        # Multiple boxed entries. There are two cases possible
        # (a) The reference solution has the same question answered in multiple ways
        # (b) The answer is split across multiple boxed entries and we need to merge
        result_equal = True
        for idx in range(len(results) - 1):
            if not (results[idx] == results[idx + 1]):
                result_equal = False
                break

        if result_equal:
            # Same problem solved in multiple ways
            return [results[0]]
        else:
            return results


def extract_attributes_from_name(file_name):
    """Extract attributes from file path."""
    eval_set, problem_type, fileid = file_name.split("/")[1:]
    fileid = fileid.split(".")[0]
    return eval_set, problem_type, fileid


def extract_answer_string_2(answer_str):
    """For two cases, inside the boxed expression, we needed a second iteration of parsing."""
    left_string = "\\boxed"
    idx = answer_str.rfind(left_string)

    stripped_answer = answer_str[idx + len(left_string) :]
    right_idx = stripped_answer.rfind("$")

    stripped_answer = stripped_answer[:right_idx]
    return stripped_answer


def save_data(split, random_seed, validation_size):
    output_dir = Path(__file__).absolute().parent
    output_dir.mkdir(exist_ok=True)
    actual_split = "test" if split == "test" else "train"
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_filename = os.path.join(temp_dir, "temp.tar")
        urllib.request.urlretrieve(DOWNLOAD_LINK, archive_filename)

        split_instances_dict = defaultdict(list)

        with tarfile.TarFile(archive_filename, mode="r") as reader_f:
            for tar_member in reader_f:
                filename = tar_member.name
                if not filename.endswith(".json"):
                    continue

                eval_set, problem_type, fileid = extract_attributes_from_name(filename)
                # TODO: we should just process all at ones, not do duplicate computation
                if eval_set != actual_split:
                    continue

                content = json.loads(reader_f.extractfile(tar_member).read())
                content["id"] = f"{eval_set}/{problem_type}/{fileid}.json"
                # Load the solution with our identified fixes
                content["reference_solution"] = _fix_solution(content["id"], content["solution"])
                del content["solution"]

                entries = find_boxed_entries(content["reference_solution"])
                if len(entries) == 1:
                    parsed_answer = entries[0]
                if len(entries) > 1:
                    parsed_answer = _post_fix_multi_answer(content["id"], entries)

                if not (
                    ("Find the equation" in content["problem"])
                    or ("Enter the equation" in content["problem"])
                    or ("What is the equation") in content["problem"]
                    or ("described by the equation") in content["problem"]
                    or ("Find an equation") in content["problem"]
                ) and ("=" in parsed_answer):
                    if parsed_answer.count("=") == 1:
                        # For greater count, it means we're just predicting values of multiple variables
                        parsed_answer = parsed_answer.split("=")[1]
                content["expected_answer"] = parsed_answer

                # Sanity check that content type matches the parent dir
                content_type = content["type"].lower()
                content_type = content_type.replace(" ", "_")
                content_type = content_type.replace("&", "and")
                assert problem_type == content_type

                content["expected_answer"] = _post_fix(content["id"], content["expected_answer"])

                split_instances_dict[eval_set].append(content)

        assert len(split_instances_dict) == 1
        for instances in split_instances_dict.values():
            # always shuffling to make it easier to get validation/train out of train_full
            if split != "test":
                random.seed(random_seed)
                random.shuffle(instances)
            if split == "validation":
                data = instances[:validation_size]
            elif split == "train":
                data = instances[validation_size:]
            else:
                data = instances
            output_file = os.path.join(output_dir, f"{split}.jsonl")
            with open(output_file, "wt", encoding="utf-8") as writer_f:
                for instance in data:
                    writer_f.write(json.dumps(instance) + "\n")


def process_data():
    """Download tar and condense data into single jsonl file."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="all",
        choices=("all", "test", "validation", "train", "train_full"),
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--validation_size", type=int, default=1000)
    args = parser.parse_args()

    if args.split == "all":
        for split in ["test", "validation", "train", "train_full"]:
            save_data(split, args.random_seed, args.validation_size)
    else:
        save_data(args.split, args.random_seed, args.validation_size)


if __name__ == "__main__":
    process_data()