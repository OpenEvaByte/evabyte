"""PAL: Program-aided Language Models
https://arxiv.org/abs/2211.10435

GSM-8k: Training Verifiers to Solve Math Word Problems
https://arxiv.org/abs/2110.14168

In PaL, Large Language Model solves reasoning problems that involve complex arithmetic and procedural tasks by generating 
reasoning chains of text and code.This offloads the execution of the code to a program runtime, in our case, a Python interpreter.

This task implements PAL methodology to evaluate GSM-8k and GSM-Hard benchmarks.
"""

from gen_eval.base import Task
from gen_eval.tasks.math_utils.eval_math import eval_all_outputs

_CITATION = """
@article{gao2022pal,
  title={PAL: Program-aided Language Models},
  author={Gao, Luyu and Madaan, Aman and Zhou, Shuyan and Alon, Uri and Liu, Pengfei and Yang, Yiming and Callan, Jamie and Neubig, Graham},
  journal={arXiv preprint arXiv:2211.10435},
  year={2022}
}

@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
"""
# Number of few shot examples to consider
NUM_SHOTS = 8

COT_EXAMPLARS = {
    "questions": ["Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
                  "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
                  "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
                  "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
                  "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
                  "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
                  "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
                  "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"],
    "solutions": ["Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.",
                "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.",
                "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.",
                "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.",
                "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.",
                "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.",
                "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.",
                "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.",
    ],
    "short_answers": [
        "8",
        "33",
        "29",
        "9",
        "8",
        "39",
        "5",
        "6"
    ]
}

BOXED_EXAMPLARS = [
    {
        "problem": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "solution": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is \\boxed{6}.",
    },
    {
        "problem": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "solution": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is \\boxed{5}.",
    },
    {
        "problem": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "solution": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is \\boxed{39}.",
    },
    {
        "problem": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "solution": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is \\boxed{8}.",
    },
    {
        "problem": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "solution": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is \\boxed{9}.",
    },
    {
        "problem": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "solution": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is \\boxed{29}.",
    },
    {
        "problem": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "solution": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is \\boxed{33}.",
    },
    {
        "problem": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "solution": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is \\boxed{8}.",
    },
]

def create_all_tasks():
    def create_gsm_task(prompt_format, multiturn_examplars):
        class GeneralGSM8K(GSM8K):
            def __init__(self):
                super().__init__(prompt_format, multiturn_examplars)

        return GeneralGSM8K

    # for GSM-8k tasks, majority voting will be used if multiple samples are available
    task_dict = {
        f"gsm8k": create_gsm_task("regular", False),
        f"gsm8k-boxed": create_gsm_task("boxed", False),
        f"gsm8k-multiturn": create_gsm_task("regular", True),
        f"gsm8k-boxed-multiturn": create_gsm_task("boxed", True),
    }
    return task_dict

class GSM8K(Task):
    DATASET_PATH = "gsm8k"
    DATASET_NAME = "main"
    SPLIT = "test"

    def __init__(self, prompt_format, multiturn_examplars):
        stop_words = ["\n\n\n\n", "\nQuestion:"]
        requires_execution = True
        super().__init__(stop_words, requires_execution)
        self.prompt_format = prompt_format
        assert self.prompt_format in ["boxed", "regular"]
        if self.prompt_format == "regular":
            # self.suffix_prompt = "\nTherefore, the answer is"
            self.suffix_prompt = "The final answer is"
        else:
            self.suffix_prompt = ""
        self.multiturn_examplars = multiturn_examplars

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        if self.SPLIT:
            return self.dataset[self.SPLIT]
        return self.dataset
    
    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        prompt = ""
        if self.prompt_format == "regular":
            for question, solution, answer in zip(
                COT_EXAMPLARS["questions"][:NUM_SHOTS], 
                COT_EXAMPLARS["solutions"][:NUM_SHOTS],
                COT_EXAMPLARS["short_answers"][:NUM_SHOTS],
            ):
                prompt += f"Question: {question}\nAnswer: {solution} {self.suffix_prompt} {answer}.\n\n\n\n"
            prompt += f"""Question: {doc["question"]}\nAnswer: """
        else:
            # use \boxed{} in the prompt
            for example in BOXED_EXAMPLARS[:NUM_SHOTS]:
                question, solution = example["problem"], example["solution"]
                prompt += f"Question: {question}\nAnswer: {solution}\n\n\n\n"
            prompt += f"""Question: {doc["question"]}\nAnswer: """
        return prompt

    def get_instruct_prompt(self, doc, tokenizer):
        messages = []
        if self.multiturn_examplars:
            if self.prompt_format == "regular":
                problem_prefix = "Given the following problem, reason and give a final answer to the problem.\nProblem: "
                problem_suffix = '\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.'
                for question, solution, answer in zip(
                    COT_EXAMPLARS["questions"][:NUM_SHOTS], 
                    COT_EXAMPLARS["solutions"][:NUM_SHOTS],
                    COT_EXAMPLARS["short_answers"][:NUM_SHOTS],
                ):
                    messages.append({"role": "user", "content": problem_prefix + question + problem_suffix})
                    messages.append({"role": "assistant", "content": solution + " " + self.suffix_prompt + " " + answer})
                messages += [{"role": "user", "content":  problem_prefix + doc["question"] + problem_suffix}]
            else:
                problem_prefix = "Given the following problem, reason and give a final answer to the problem.\nProblem: "
                problem_suffix = '\nYour response should end with "So the answer is $\\boxed{[answer]}$" where [answer] is the response to the problem.'
                for example in BOXED_EXAMPLARS[:NUM_SHOTS]:
                    messages.append({"role": "user", "content": problem_prefix + example["problem"] + problem_suffix})
                    messages.append({"role": "assistant", "content": example["solution"]})
                messages += [{"role": "user", "content":  problem_prefix + doc["question"] + problem_suffix}]
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
                for question, solution, answer in zip(
                    COT_EXAMPLARS["questions"][:NUM_SHOTS], 
                    COT_EXAMPLARS["solutions"][:NUM_SHOTS],
                    COT_EXAMPLARS["short_answers"][:NUM_SHOTS],
                ):
                    context += f"Question: {question}\nAnswer: {solution} {self.suffix_prompt} {answer}.\n\n"
            else:
                # use \boxed{} in the prompt
                for example in BOXED_EXAMPLARS[:NUM_SHOTS]:
                    question, solution = example["problem"], example["solution"]
                    context += f"Question: {question}\nAnswer: {solution}\n\n"
            context += f"""Question: {doc["question"]}\n"""
            prompt = context.strip()
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], 
                tokenize=False, 
                add_generation_prompt=True
            )
            if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token):]
            prompt += "Answer: "
            return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        target = doc["answer"].split("####")[-1]
        expected_answer = float(target.replace(",", ""))
        if int(expected_answer) == expected_answer:
            expected_answer = int(expected_answer)
        return str(expected_answer)

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
        return outputs