import fnmatch

class BaseIOProcessor:
    def __init__(self, task, task_name, tokenizer):
        self.task = task
        self.task_name = task_name
        self.tokenizer = tokenizer

    def process_input(self, doc):
        return self.task.get_prompt(doc)

    def process_output(self, output, task_id):
        """
            clean up the generation and extract the appropriate snippet
            for subsequent evaluation
            @output: the full solution (include both prompt and code)
        """
        if "insertion" in self.task_name:
            gen_code = self.task.postprocess_generation(output, int(task_id))
            return gen_code
        dataset = self.task.get_dataset()
        prompt = self.process_input(dataset[task_id])
        gen_code = output[len(prompt) :]
        gen_code = self.task.postprocess_generation(gen_code, int(task_id))
        return gen_code

    def trim_output(self, output, task_id):
        """
            remove any code beyond the current completion scope
            @output: the full solution (include both prompt and code)
        """
        dataset = self.task.get_dataset()
        prompt = self.process_input(dataset[task_id])
        gen_code = output[len(prompt) :]
        gen_code = self.task.trim_generation(gen_code, int(task_id))
        return prompt + gen_code

class ByteLMBaseIOProcessor(BaseIOProcessor):
    def __init__(self, task, task_name, tokenizer):
        super().__init__(task, task_name, tokenizer)

    def process_input(self, doc):
        prompt = self.task.get_prompt(doc)
        if self.task_name.startswith("cute"):
            # we add an extra space for byte models so that the prediction of choices
            # is much more natural
            prompt = prompt + " "
        # print("Input prompt =====>", [prompt], flush=True)
        return prompt

class BaseInstructIOProcessor(BaseIOProcessor):
    def __init__(self, task, task_name, tokenizer):
        super().__init__(task, task_name, tokenizer)

    def process_input(self, doc):
        return self.task.get_instruct_prompt(doc, self.tokenizer)

class ByteLMInstructIOProcessor(BaseIOProcessor):
    def __init__(self, task, task_name, tokenizer):
        super().__init__(task, task_name, tokenizer)
        if self.task_name in [
            "humaneval",
            "humaneval_plus",
            "mbpp",
            "mbpp_plus"
        ]:
            self.task.stop_words = [
                "\nclass", 
                "\nprint(", 
                "\nif __name__", 
                "\ndef main(", 
                "\n```", 
                '\n"""', 
                "\nassert", 
                "\n#"
            ]

        if self.task_name.startswith("bigcodebench"):
            self.task.stop_words = [
                "\nif __name__",
                "\ndef main(",
                "\nprint(",
                "\n```\n",
            ]
            # TODO: hacky. will refactor
            self.task._mode = "instruct"
        self.task.stop_words.append("<|eot_id|>")
        # eos tokens are added in evaluator.py

    def process_input(self, doc):
        prompt = self.task.get_instruct_prompt(doc, self.tokenizer)
        if self.task_name in ["mmlu"] or self.task_name.startswith("cute"):
            # we add an extra space for byte models so that the prediction of choices
            # is much more natural
            prompt = prompt + " "
        # print("Input prompt =====>", [prompt], flush=True)
        return prompt

    def process_output(self, output, task_id):
        dataset = self.task.get_dataset()
        prompt = self.process_input(dataset[task_id])
        gen_code = output[len(prompt) :]
        if self.task_name.startswith("bigcodebench"):
            gen_code = self.task.trim_generation(gen_code, int(task_id))
            print("Input prompt =====>", [prompt], flush=True)
            print("Output code =====>", [output[len(prompt) :]], [gen_code], flush=True)
            return gen_code
        gen_code = self.task.postprocess_generation(gen_code, int(task_id))
        if self.task_name.startswith("math") or self.task_name.startswith("gsm"):
            print("Input prompt =====>", [prompt], flush=True)
            print("Output code =====>", [output[len(prompt) :]], [gen_code], flush=True)
        return gen_code

class StarCoderIOProcessor(BaseIOProcessor):
    def process_input(self, doc):
        """Builds the prompt for the LM to generate from."""

        ###################################
        # set up task attributes
        # for codellama and deepseek, do not strip() since 
        # \n is encoded into a single token,
        # but for starcoder models, we need to strip prompt, as
        # \n and 4 indentations are encoded into a single token
        # if (
        #     (
        #         "starcoder" in self.tokenizer.name_or_path or
        #         "santacoder" in self.tokenizer.name_or_path or
        #         "WizardCoder" in self.tokenizer.name_or_path
        #     ) and 
        #     (
        #         "humaneval" in task_name or 
        #         "mbpp" in task_name
        #     )
        # ):
        #     task.strip_prompt = True
        # else:
        #     # DS1000 and GSM8k does not apply strip() because
        #     # otherwise the model might continue to complete 
        #     # the start indicator.
        #     task.strip_prompt = False
        if self.task_name in [
            "humaneval",
            "humaneval_plus",
            "mbpp",
            "mbpp_plus"
        ]:
            return super().process_input(doc).strip()
        else:
            return super().process_input(doc)

class CodeLlamaInstructIOProcessor(BaseIOProcessor):
    def __init__(self, task, task_name, tokenizer):
        super().__init__(task, task_name, tokenizer)
        if self.task_name in [
            "mbpp_plus"
        ]:
            self.task.stop_words.append("\n[/PYTHON]")

    def process_input(self, doc):
        prompt = super().process_input(doc)
        if self.task_name in ["humaneval", "humaneval_plus"]:
            prompt = '''[INST] Write a Python function to solve the following problem:
{} [/INST] {}'''.format(prompt.rstrip(), prompt)
            return prompt
        elif self.task_name == "mbpp":
            prompt = '''[INST] {}'''.format(prompt.rstrip())[:-len("[BEGIN]\n")] + " [/INST] \n[BEGIN]\n"
            return prompt
        elif self.task_name == "mbpp_plus":
            # codellama models seem to overfit to [PYTHON]
            prompt = '''[INST] {} [/INST] \n[PYTHON]\n'''.format(prompt.rstrip())
            return prompt
        else:
            return prompt
