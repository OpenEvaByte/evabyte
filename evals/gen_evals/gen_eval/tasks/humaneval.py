"""Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374

The HumanEval dataset released by OpenAI includes 164 programming problems with a function signature,
docstring, body, and several unit tests. 
They were handwritten to ensure not to be included in the training set of code generation models.

Homepage: https://github.com/openai/human-eval
"""

import re
import os
import json
from evaluate import load

from gen_eval.base import Task

_CITATION = """
@misc{chen2021evaluating,
      title={Evaluating Large Language Models Trained on Code},
      author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
      year={2021},
      eprint={2107.03374},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""

def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {"humaneval": GeneralHumanEval}

class GeneralHumanEval(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "openai_humaneval"

    def __init__(self):
        super().__init__(
            stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"],
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        return doc["prompt"]

    def get_instruct_prompt(self, doc, tokenizer):
        _MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

        instruction_prefix = "Please write a self-contained solution to the following problem in a markdown code block and make sure it passes the tests:"
        response_prefix = "Here is the completed Python function:"

        task_prompt = f"""{instruction_prefix}\n```\n{doc["prompt"].strip()}\n```\n"""
        response = f"""{response_prefix}\n```python\n{doc["prompt"]}{_MAGIC_SPLITTER_}```"""
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
        code_metric = load("code_eval")
        results, logs = code_metric.compute(
            references=references,
            predictions=generations,
            num_workers=int(os.getenv("HF_CODE_EVAL_NUM_PROC", "1")),
        )
        
        with open(os.getenv("RUN_STATS_SAVE_PATH", "logs_humaneval.json"), "w") as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)

        return results
