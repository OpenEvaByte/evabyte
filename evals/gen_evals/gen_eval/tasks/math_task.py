"""MATH dataset
"""

from gen_eval.base import Task
from gen_eval.tasks.math_utils.eval_math import eval_all_outputs


from gen_eval.tasks.math_utils.hendrycks_utils import is_equiv as hendrycks_is_equiv
from gen_eval.tasks.math_utils.minerva_utils import (
    get_unnormalized_answer,
    is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
)
from gen_eval.tasks.math_utils.minerva_utils import (
    process_results as minerva_math_process_results,
)
from gen_eval.tasks.math_utils.minerva_utils import remove_boxed
import re

import random
import pathlib
import json

_CITATION = """
@article{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the MATH Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}

@misc{2206.14858,
  Author = {Aitor Lewkowycz and Anders Andreassen and David Dohan and Ethan Dyer and Henryk Michalewski and Vinay Ramasesh and Ambrose Slone and Cem Anil and Imanol Schlag and Theo Gutman-Solo and Yuhuai Wu and Behnam Neyshabur and Guy Gur-Ari and Vedant Misra},
  Title = {Solving Quantitative Reasoning Problems with Language Models},
  Year = {2022},
  Eprint = {arXiv:2206.14858},
}
"""
# Number of few shot examples to consider
NUM_SHOTS = 4

REGULAR_EXAMPLARS = [
    {
        "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
        "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
    },
    {
        "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
        "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
    },
    {
        "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
        "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
    },
    {
        "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
        "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
    },
]

BOXED_EXAMPLARS = [
    {
        "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
        "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.",
    },
    {
        "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
        "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$",
    },
    {
        "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
        "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}",
    },
    {
        "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
        "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$",
    },
]

INSTRUCT_ZEROSHOT_PROMPT = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: "
# NOTE: all use more detailed examplars in 
# https://github.com/Kipok/NeMo-Skills/blob/1a45affd1f6d60738f24f8791707cae2dc211a8e/nemo_skills/prompt/few_shot_examples/math.py#L37

def create_all_tasks():
    def create_task(prompt_format, num_examples, multiturn_examplars):
        class GeneralMATH(MATH):
            def __init__(self):
                super().__init__(prompt_format, num_examples, multiturn_examplars)

        return GeneralMATH

    task_dict = {
        f"math-subsampled": create_task("regular", 1000, False),
        f"math-boxed-subsampled": create_task("boxed", 1000, False),
        f"math": create_task("regular", None, False),
        f"math-boxed": create_task("boxed", None, False),
        f"math-multiturn": create_task("regular", None, True),
        f"math-boxed-multiturn": create_task("boxed", None, True),
    }
    return task_dict

class MATH(Task):
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self, prompt_format, num_examples=None, multiturn_examplars=False):
        stop_words = ["\n\n\n\n", "\nQuestion:"]
        requires_execution = True
        super().__init__(stop_words, requires_execution)
        self.prompt_format = prompt_format
        assert self.prompt_format in ["boxed", "regular"]
        if self.prompt_format == "regular":
            self.suffix_prompt = "\nFinal Answer: The final answer is "
        else:
            self.suffix_prompt = ""

        self.multiturn_examplars = multiturn_examplars
        with open(pathlib.Path(__file__).parent / "math_utils" / "math_testset.jsonl", 'r') as f:
            self.dataset = [json.loads(line) for line in f]
            if num_examples:
                self.dataset = random.Random(42).sample(self.dataset, num_examples)

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset
    
    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        if self.prompt_format == "regular":
            examplar_list = REGULAR_EXAMPLARS
        else:
            examplar_list = BOXED_EXAMPLARS
        
        prompt = ""
        for example in examplar_list:
            question, solution = example["problem"], example["solution"]
            prompt += f"Question: {question}\nAnswer: {solution}\n\n\n\n"
        prompt += f"""Question: {doc["problem"]}\nAnswer: """
        return prompt

    def get_instruct_prompt(self, doc, tokenizer):
        messages = []
        if self.multiturn_examplars:
            problem_prefix = "Given the following problem, reason and give a final answer to the problem.\nProblem: "
            if self.prompt_format == "regular":
                problem_suffix = '\nYour response should end with "\nFinal Answer: The final answer is $[answer]$. I hope it is correct." where [answer] is the response to the problem.'
                examplars = REGULAR_EXAMPLARS[:NUM_SHOTS]
            else:
                problem_suffix = "\nSolve the question below by reasoning step by step, and put the final answer within \\boxed{}."
                examplars = BOXED_EXAMPLARS[:NUM_SHOTS]
            for example in examplars:
                messages.append({"role": "user", "content": problem_prefix + example["problem"] + problem_suffix})
                messages.append({"role": "assistant", "content": example["solution"]})
            messages += [{"role": "user", "content":  problem_prefix + doc["problem"] + problem_suffix}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token):]
            return prompt
        else:
            context = "You are supposed to provide a solution to a given problem.\n\n"
            if self.prompt_format == "regular":
                examplars = REGULAR_EXAMPLARS[:NUM_SHOTS]
            else:
                examplars = BOXED_EXAMPLARS[:NUM_SHOTS]
            for example in examplars:
                context += '\n' + 'Problem:\n{query}\nSolution:\n{response}\n'.format(query=example["problem"], response=example["solution"])

            context += '\n' + 'Problem:\n{query}\n'.format(query=doc["problem"])
            prompt = context.strip()
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], 
                tokenize=False, 
                add_generation_prompt=True
            )
            if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token):]
            prompt += "Solution:\n"
            return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc["expected_answer"]

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
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        return self._stop_at_stop_token(generation, self.stop_words)

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this task)
        """
        return generation

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(float)
            list of references
        """
        if self.prompt_format == "regular":
            extract_from_boxed = False
            extract_regex = self.suffix_prompt + r" (.+)$"
        else:
            extract_from_boxed = True
            extract_regex = r"\\boxed{(.+)}$"
        outputs = eval_all_outputs(
            generations,
            references,
            extract_from_boxed,
            extract_regex
        )
        # also calculate OLMES style
        outputs.update(self.olmes_metrics(generations, references))
        return outputs

    def olmes_extract_answers(self, raw_answer):
        # Flexible answer extraction for multiple answers, the first one is the "primary" answer
        boxed_answer = last_boxed_only_string(raw_answer)
        if boxed_answer is not None:
            try:
                boxed_answer = remove_boxed(boxed_answer)
            except AssertionError:
                boxed_answer = None
        all_answers = []
        if self.prompt_format == "regular":
            minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
            if minerva_answer is not None and minerva_answer != "[invalidanswer]":
                all_answers.append(minerva_answer)
        if boxed_answer is not None:
            all_answers.append(normalize_final_answer(boxed_answer))
        if len(all_answers) == 0:
            dollars = [m.start() for m in re.finditer("\\$", raw_answer)]
            if len(dollars) > 1:
                # Add the answer between the second to last and last dollar sign
                answer = normalize_final_answer(raw_answer[dollars[-2] + 1 : dollars[-1]])
                all_answers.append(answer)
        if len(all_answers) == 0:
            all_answers.append(normalize_final_answer(raw_answer))
        return all_answers

    def olmes_metrics(self, generations, references):
        outputs = {}
        all_predictions = []
        for task_id, (gens, ref) in enumerate(zip(generations, references)):
            predictions = []
            for gen in gens:
                metrics: dict = {}
                if self.prompt_format == "boxed":
                    extracted_answer = last_boxed_only_string(gen)
                    if extracted_answer is None:
                        extracted_answer = gen
                    else:
                        try:
                            extracted_answer = remove_boxed(extracted_answer)
                        except AssertionError:
                            extracted_answer = gen
                    answer = normalize_final_answer(extracted_answer)
                    metrics = {"exact_match": 1 if hendrycks_is_equiv(answer, ref) else 0}
                else:
                    metrics = minerva_math_process_results({"answer": ref}, [gen])

                max_flex_match = metrics.get("exact_match", 0)
                all_extracted_answers = self.olmes_extract_answers(gen)
                for answer in all_extracted_answers:
                    if max_flex_match == 1:
                        break
                    if is_equiv(answer, ref):
                        max_flex_match = 1
                    elif hendrycks_is_equiv(answer, ref):
                        max_flex_match = 1
                metrics["exact_match_flex"] = max_flex_match
                if all_extracted_answers:
                    metrics["model_answer"] = all_extracted_answers[0]
                predictions.append(metrics)
            all_predictions.append(predictions)

        total = 0
        em_correct = 0
        flex_correct = 0
        no_answer = 0
        for predictions in all_predictions:
            em_correct += predictions[0]['exact_match']
            flex_correct += predictions[0]['exact_match_flex']
            no_answer += (predictions[0].get("model_answer", None) is None)
            total += 1
        outputs["olmes_em"] = em_correct / total * 100.0
        outputs["olmes_flex"] = flex_correct / total * 100.0
        outputs["olmes_no_answer"] = no_answer / total * 100.0

        return outputs