from gen_eval.base import Task
import os
import re
import json
import evaluate
from collections import defaultdict
from datasets import load_dataset
exact_match = evaluate.load("exact_match")

TASKS = [
    'spell',
    'spell_inverse',
    'contains_char',
    'contains_word',
    'orth',
    'sem',
    'ins_char',
    'ins_word',
    'del_char',
    'del_word',
    'sub_char',
    'sub_word',
    'swap_char',
    'swap_word',
]

def create_all_tasks():
    def create_task(task_name=None):
        class GeneralCUTE(CUTE):
            def __init__(self):
                super().__init__(task_name)
        return GeneralCUTE
    
    task_dict = {
        f"cute": create_task(task_name=None),
    }
    for task_name in TASKS:
        task_dict[f"cute-{task_name}"] = create_task(task_name=task_name)
    return task_dict

class CUTE(Task):
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self, task_name):
        super().__init__(
            stop_words=["\"\n", "Answer:", "Question:"],
            requires_execution=True
        )
        self.dataset = []
        if task_name is None:
            tasks = TASKS
        else:
            tasks = [task_name]
        for task_name in tasks:
            _dataset = load_dataset(path="leukas/cute")[task_name]
            for example in _dataset:
                self.dataset.append(
                    {
                        "task_name": task_name, 
                        "prompt": example["prompt"],
                        "answer": example["answer"]
                    }
                )
        self.task_name = task_name

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        return doc["prompt"] + " Answer:"

    def get_instruct_prompt(self, doc, tokenizer):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": doc["prompt"]}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
            prompt = prompt[len(tokenizer.bos_token):]
        prompt += "Answer:"
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return {
            "task_name": doc["task_name"],
            "answer": doc["answer"]
        }

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
        """Remove any code beyond the current completion scope."""
        return self._stop_at_stop_token(generation, self.stop_words)

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this task)
        """
        output = self._stop_at_stop_token(generation, self.stop_words)
        output = output.strip()
        # Use regex to find all content within double quotes
        matches = re.findall(r'"(.*?)"', output)

        if matches:
            # Return the stripped content of the first match
            return matches[0].strip()
        else:
            stripped_string = output.strip()
            if stripped_string.startswith('"'):
                stripped_string = stripped_string[1:]
            if stripped_string.endswith('"'):
                stripped_string = stripped_string[:-1]
            return stripped_string.strip()

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(float)
            list of references
        """
        performance = {}
        assert len(generations) == len(references), "number of predictions and targets are not the same."
        
        generations_by_task = defaultdict(list)
        references_by_task = defaultdict(list)
        for generation, reference in zip(generations, references):
            task_name = reference["task_name"]
            if len(generation) == 1:
                # we only generate 1 sample for each question
                generations_by_task[task_name].append(generation[0])
            else:
                raise ValueError("Self-consistency inference is not implemented yet!")
            references_by_task[task_name].append(reference["answer"])

        for task_name, generations in generations_by_task.items():
            performance[task_name] = exact_match.compute(
                predictions=generations, 
                references=references_by_task[task_name], 
                ignore_case=True, 
                ignore_punctuation=True
            )["exact_match"]

            print(f"Task {task_name} - EM: {performance[task_name]}")

        # save the performance
        with open(os.getenv("RUN_STATS_SAVE_PATH", "metrics.json"), "w") as fout:
            performance["average_exact_match"] = sum(performance.values()) / len(performance)
            print(f"Average EM: {performance['average_exact_match']}")
            json.dump(performance, fout, indent=4)
