"""BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions
https://arxiv.org/abs/2406.15877

The BigCodeBench dataset

Homepage: https://bigcode-bench.github.io
"""

import json
import os
from typing import Dict, Generator, List, Optional, Set, Tuple
from bigcodebench.data import write_jsonl
from gen_eval.base import Task
from gradio_client import Client, handle_file
import time
from concurrent.futures._base import CancelledError
import httpx

import ast
import traceback
from tree_sitter import Language, Node, Parser
import tree_sitter_python

_CITATION = """
@article{zhuo2024bigcodebench,
  title={BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions},
  author={Zhuo, Terry Yue and Vu, Minh Chien and Chim, Jenny and Hu, Han and Yu, Wenhao and Widyasari, Ratnadira and Yusuf, Imam Nur Bani and Zhan, Haolan and He, Junda and Paul, Indraneil and others},
  journal={arXiv preprint arXiv:2406.15877},
  year={2024}
}
"""

def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {"bigcodebench-full": create_task("full"), "bigcodebench-hard": create_task("hard")}

def create_task(task_subset):
    class BigCodeBench(GeneralBigCodeBench):
        def __init__(self):
            super().__init__(task_subset)

    return BigCodeBench


CLASS_TYPE = "class_definition"
FUNCTION_TYPE = "function_definition"
IMPORT_TYPE = ["import_statement", "import_from_statement"]
IDENTIFIER_TYPE = "identifier"
ATTRIBUTE_TYPE = "attribute"
RETURN_TYPE = "return_statement"
EXPRESSION_TYPE = "expression_statement"
ASSIGNMENT_TYPE = "assignment"

def syntax_check(code, verbose=False):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False

def code_extract(text: str) -> str:
    lines = text.split("\n")
    longest_line_pair = (0, 0)
    longest_so_far = 0

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            current_lines = "\n".join(lines[i : j + 1])
            if syntax_check(current_lines):
                current_length = sum(1 for line in lines[i : j + 1] if line.strip())
                if current_length > longest_so_far:
                    longest_so_far = current_length
                    longest_line_pair = (i, j)

    return "\n".join(lines[longest_line_pair[0] : longest_line_pair[1] + 1])

def get_deps(nodes: List[Tuple[str, Node]]) -> Dict[str, Set[str]]:

    def dfs_get_deps(node: Node, deps: Set[str]) -> None:
        for child in node.children:
            if child.type == IDENTIFIER_TYPE:
                deps.add(child.text.decode("utf8"))
            else:
                dfs_get_deps(child, deps)

    name2deps = {}
    for name, node in nodes:
        deps = set()
        dfs_get_deps(node, deps)
        name2deps[name] = deps
    return name2deps

def get_function_dependency(entrypoint: str, call_graph: Dict[str, str]) -> Set[str]:
    queue = [entrypoint]
    visited = {entrypoint}
    while queue:
        current = queue.pop(0)
        if current not in call_graph:
            continue
        for neighbour in call_graph[current]:
            if not (neighbour in visited):
                visited.add(neighbour)
                queue.append(neighbour)
    return visited

def get_definition_name(node: Node) -> str:
    for child in node.children:
        if child.type == IDENTIFIER_TYPE:
            return child.text.decode("utf8")

def traverse_tree(node: Node) -> Generator[Node, None, None]:
    cursor = node.walk()
    depth = 0

    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                depth += 1
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent() or depth == 0:
            break
        else:
            depth -= 1

def has_return_statement(node: Node) -> bool:
    traverse_nodes = traverse_tree(node)
    for node in traverse_nodes:
        if node.type == RETURN_TYPE:
            return True
    return False

def extract_target_code_or_empty(code: str, entrypoint: Optional[str] = None) -> str:
    code = code_extract(code.strip())
    code_bytes = bytes(code, "utf8")
    parser = Parser(Language(tree_sitter_python.language()))
    tree = parser.parse(code_bytes)
    class_names = set()
    function_names = set()
    variable_names = set()

    root_node = tree.root_node
    import_nodes = []
    definition_nodes = []

    for child in root_node.children:
        if child.type in IMPORT_TYPE:
            import_nodes.append(child)
        elif child.type == CLASS_TYPE:
            name = get_definition_name(child)
            if not (
                name in class_names or name in variable_names or name in function_names
            ):
                definition_nodes.append((name, child))
                class_names.add(name)
        elif child.type == FUNCTION_TYPE:
            name = get_definition_name(child)
            if not (
                name in function_names or name in variable_names or name in class_names
            ):
                definition_nodes.append((name, child))
                function_names.add(get_definition_name(child))
        elif (
            child.type == EXPRESSION_TYPE and child.children[0].type == ASSIGNMENT_TYPE
        ):
            subchild = child.children[0]
            name = get_definition_name(subchild)
            if not (
                name in variable_names or name in function_names or name in class_names
            ):
                definition_nodes.append((name, subchild))
                variable_names.add(name)

    if entrypoint:
        name2deps = get_deps(definition_nodes)
        reacheable = get_function_dependency(entrypoint, name2deps)

    sanitized_output = b""

    for node in import_nodes:
        sanitized_output += code_bytes[node.start_byte : node.end_byte] + b"\n"

    for pair in definition_nodes:
        name, node = pair
        if entrypoint and not (name in reacheable):
            continue
        sanitized_output += code_bytes[node.start_byte : node.end_byte] + b"\n"
        
    sanitized_output = sanitized_output[:-1].decode("utf8")
    
    # ad-hoc approach to remove unnecessary lines, but it works
    lines = sanitized_output.splitlines()
    outer_lines = []
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith(" "):
            break
        if not lines[i].startswith(" ") and entrypoint in lines[i]:
            outer_lines.append(i)
    if outer_lines:
        sanitized_output = "\n".join(lines[: outer_lines[-1]])
    return sanitized_output

def sanitize(code: str, entrypoint: Optional[str] = None) -> str:
    sanitized_code = extract_target_code_or_empty(code, entrypoint).strip()
    if not sanitized_code:
        return code_extract(code)
    return sanitized_code


def evaluate(
    split: str,
    subset: str,
    samples: Optional[str] = None,
    result_path: Optional[str] = None,
    remote_execute_api: str = "https://bigcode-bigcodebench-evaluator.hf.space/",
    pass_k: str = "1,5,10",
    parallel: int = -1,
    min_time_limit: float = 1,
    max_as_limit: int = 30*1024,
    max_data_limit: int = 30*1024,
    max_stack_limit: int = 10,
    check_gt_only: bool = False,
    no_gt: bool = False,
):  
    assert samples is not None, "No samples provided"
        
    while True:
        try:
            client = Client(remote_execute_api)
            results, pass_at_k = client.predict(
                split=split,
                subset=subset,
                samples=handle_file(samples),
                pass_k=pass_k,
                parallel=parallel,
                min_time_limit=min_time_limit,
                max_as_limit=max_as_limit,
                max_data_limit=max_data_limit,
                max_stack_limit=max_stack_limit,
                check_gt_only=check_gt_only,
                no_gt=no_gt,
                api_name="/predict"
            )
            break
        except (httpx.ReadTimeout, CancelledError):
            print("Read timeout error. Retrying in 4s...")
            time.sleep(4)
    gt_pass_rate = pass_at_k["gt_pass_rate"]
    failed_tasks = pass_at_k["failed_tasks"]
            
    extra = subset.capitalize()
    split = split.capitalize()
    print(f"BigCodeBench-{split} ({extra})", "green")

    if no_gt:
        print(f"Groundtruth is not checked", "yellow")
    else:
        if gt_pass_rate > 0.99:
            print(f"Groundtruth pass rate: {gt_pass_rate:.3f}", "green")
        else:
            print(f"Groundtruth pass rate: {gt_pass_rate:.3f}\nPlease be cautious!", "red")
        
        if len(failed_tasks) > 0:
            print(f"Failed tasks: {failed_tasks}", "red")
    
    for k, v in pass_at_k.items():
        if k.startswith("pass@"):
            print(f"{k}:\t{v:.3f}", "green")

    if not os.path.isfile(result_path):
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

class GeneralBigCodeBench(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "bigcode/bigcodebench"
    DATASET_NAME = None
    DATASET_VERSION = "v0.1.2"

    def __init__(self, subset):
        extra = "-" + subset if subset != "full" else ""
        self.DATASET_PATH = self.DATASET_PATH + extra
        super().__init__(
            stop_words=["\nif __name__", "\nprint(", "\nclass ", "\ndef ", "\n#", "\n@", "\nif ", "\n```", "\nimport ", "\nfrom ", "\nassert "],
            requires_execution=True,
        )
        self.data_subset = subset
        self._mode = "complete"
        self.dataset = self.dataset[self.DATASET_VERSION]
    
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        return doc["complete_prompt"]

    def get_instruct_prompt(self, doc, tokenizer):
        _MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

        instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
        response_prefix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"

        task_prompt = f"""{instruction_prefix}\n{doc["instruct_prompt"].strip()}\n"""
        response = f"""{response_prefix}\n```python\n{_MAGIC_SPLITTER_}\n```"""
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
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point
    
    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
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
        return decoded_string[:min_stop_index]

    def trim_generation(self, generation, idx):
        """Intermediate Removal of any code beyond the current completion scope.
        :param generation: str
            code generation from LM (w/o the prompt)
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        return self._stop_at_stop_token(generation, self.stop_words)

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM (w/o the prompt)
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        prompt = self.get_prompt(self.get_dataset()[idx])
        # generation = generation[len(prompt) :]
        trimmed_gen_code = prompt + self._stop_at_stop_token(generation, self.stop_words)
        # sometimes models generate 3 spaces as indentation
        # post-process the code to fix the inconsistency
        tmp_gen_code = ""
        for line in trimmed_gen_code.splitlines():
            lspace = len(line) - len(line.lstrip())
            if lspace == 3:
                tmp_gen_code += " "
            tmp_gen_code += line + "\n"
        return tmp_gen_code

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        print(len(generations), len(self.dataset))
        assert len(generations) == len(self.dataset)
        # get the directory
        samples_for_eval_path = os.path.join(
            os.path.dirname(os.getenv("RUN_STATS_SAVE_PATH", "")),
            "bigcodebench_{}_samples.jsonl".format(self.data_subset)
        )
        samples_for_eval = [
            dict(
                task_id = self.dataset[idx]["task_id"],
                solution = sanitize(gen, self.dataset[idx]["entry_point"]),
                raw_solution = gen
            )
            for idx, generation in enumerate(generations)
            for gen in generation
        ]
        write_jsonl(samples_for_eval_path, samples_for_eval)
        
        # delete the result file if exists
        result_path = samples_for_eval_path.replace(".jsonl", "_eval_results.json")
        if os.path.isfile(result_path):
            os.remove(result_path)
        evaluate(
            split=self._mode,
            subset=self.data_subset,
            samples=samples_for_eval_path,
            result_path=result_path,
        )
        return generations
