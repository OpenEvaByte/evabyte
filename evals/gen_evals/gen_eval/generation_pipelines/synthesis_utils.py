import math
import numpy as np
from multiprocessing.sharedctypes import Value
import torch
from tqdm import tqdm

def vanilla_completion(
    input_dict,
    model,
    tokenizer,
    task,
    device,
    batch_size,
    gen_kwargs,
    tokenizer_kwargs,
):
    task_id = input_dict["task_id"]
    prompt = input_dict["prompt"]
    with torch.no_grad():
        outputs = tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_tensors="pt",
            **tokenizer_kwargs
        )
        batch = {
            "ids": outputs.input_ids.to(device),
            "task_id": task_id,
            "input_len": outputs.attention_mask.sum(),
        }
        max_input_len = batch["input_len"].item()
        if task.stop_words:
            if gen_kwargs.get("stopping_criteria", None) is not None:
                # Set the start_length after which to check for stopping to be the longest input ignoring padding
                gen_kwargs["stopping_criteria"][0].start_length = max_input_len
        if hasattr(task, "max_length_multiplier") and task.max_length_multiplier:
            idx = 1 if task.stop_words else 0
            gen_kwargs["stopping_criteria"][idx].input_length = max_input_len
        inputs = batch["ids"][:, : batch["input_len"]]
        generated_tokens = model.generate(
            input_ids=inputs,
            num_return_sequences=batch_size,
            **gen_kwargs,
        )
        generated_tokens = generated_tokens.cpu().numpy()
        generated_code = []
        for generated_token in generated_tokens:
            if generated_token[0] == tokenizer.bos_token_id:
                generated_token = generated_token[1:]

            generated_code.append(tokenizer.decode(
                generated_token, skip_special_tokens=False, clean_up_tokenization_spaces=False
            ))
    return generated_code

def vanilla_multibyte_completion(
    input_dict,
    model,
    tokenizer,
    task,
    device,
    batch_size,
    gen_kwargs,
    tokenizer_kwargs,
):
    task_id = input_dict["task_id"]
    prompt = input_dict["prompt"]
    with torch.no_grad():
        outputs = tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_tensors="pt",
            **tokenizer_kwargs
        )
        batch = {
            "ids": outputs.input_ids.to(device),
            "task_id": task_id,
            "input_len": outputs.attention_mask.sum(),
        }
        max_input_len = batch["input_len"].item()
        if task.stop_words:
            if gen_kwargs.get("stopping_criteria", None) is not None:
                # Set the start_length after which to check for stopping to be the longest input ignoring padding
                gen_kwargs["stopping_criteria"][0].start_length = max_input_len
        if hasattr(task, "max_length_multiplier") and task.max_length_multiplier:
            idx = 1 if task.stop_words else 0
            gen_kwargs["stopping_criteria"][idx].input_length = max_input_len
        inputs = batch["ids"][:, : batch["input_len"]]
        assert hasattr(model, "multi_byte_generate")
        gen_kwargs.pop("top_k", None)
        generated_tokens = model.multi_byte_generate(
            input_ids=inputs,
            **gen_kwargs,
        )
        generated_tokens = generated_tokens.cpu().numpy()
        generated_code = []
        for generated_token in generated_tokens:
            if generated_token[0] == tokenizer.bos_token_id:
                generated_token = generated_token[1:]

            generated_code.append(tokenizer.decode(
                generated_token, skip_special_tokens=False, clean_up_tokenization_spaces=False
            ))
    return generated_code
