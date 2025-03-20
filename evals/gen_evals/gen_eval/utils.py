import torch
from transformers import (
    StoppingCriteria, 
)

class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""
    def __init__(self, start_length, eof_strings, tokenizer, check_fn=None):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer
        if check_fn is None:
            check_fn = lambda decoded_generation: any(
                [stop_string in decoded_generation for stop_string in self.eof_strings]
            )
        self.check_fn = check_fn

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        return all([self.check_fn(decoded_generation) for decoded_generation in decoded_generations])

class TooLongFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if the generated function is too long by a certain multiplier based on input length."""

    def __init__(self, input_length, multiplier):
        self.input_length = input_length
        self.multiplier = multiplier

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if generated sequence is too long."""
        return input_ids.shape[1] > int(self.input_length * self.multiplier)
