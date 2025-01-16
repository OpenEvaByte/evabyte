# EvaByte: Efficient Byte-level Language Models at Scale

<p align="center">
   <a href="https://huggingface.co/collections/linzheng/evabyte-6781cfc1793bdaf579fc4461" target="_blank"  style="margin-right: 10px;">
      <img alt="Huggingface" src="https://img.shields.io/badge/🤗-HF_Models-blue" />
   </a>
   <a href="https://hkunlp.github.io/blog/2025/evabyte" target="_blank" style="margin-right: 10px;">
      <img alt="Blog" src="https://img.shields.io/badge/📰-Blog-red" />
   </a>
      <img alt="Paper" src="https://img.shields.io/badge/📜-Paper_(Coming_Soon)-gray" />
</p>

**EvaByte** is a 6.5B **byte-level language model** built upon an improved architecture with multibyte prediction and EVA -- an efficient attention mechanism designed for scalability and performance. Trained on 1.5T bytes of natural language text, math, and code, EvaByte rivals top open-source tokenizer-based LMs using 5x less training data, excelling in coding tasks, and decoding up to 2x faster.

This repository provides the model implementation and inference examples for EvaByte.

## 🛠️ Model Implementation

Our implementation of EvaByte is based on the [Huggingface `transformers`](https://github.com/huggingface/transformers) library, located in the `evabyte_hf` folder. Its attention mechanism, EVA, is adapted from this [repository](https://github.com/HKUNLP/efficient-attention) and implemented with a `triton` kernel -- performant but with room for further optimization. We also provide a native PyTorch implementation in `evabyte_hf/eva_pt_ref.py` for reference.

## 🚀 Inference with `transformers`

Inference on GPUs with `transformers` is supported -- please make sure you have `torch>=2.4`, `transformers`, and `triton>=3.0` installed.

### 📄 Base Model

**Note:** Make sure to set `trust_remote_code=True` when loading the model (or tokenizer), as our implementation includes custom code.

The code snippet below, which is also available in `example_completion.py`, demonstrates EvaByte-6.5B for completion:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("evabyte/EvaByte", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("evabyte/EvaByte", torch_dtype=torch.bfloat16, trust_remote_code=True).eval().to("cuda")

prompt = "The quick brown fox jumps "

# Tokenize input
# Option 1: standard HF tokenizer interface
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

# Option 2: Direct UTF-8 byte encoding with offset
# Note: Each byte is offset by 64 with <bos> prepended.
input_ids = torch.tensor([[1] + [b + 64 for b in prompt.encode("utf-8")]]).to("cuda")

# byte-by-byte generation (default)
generation_output = model.generate(
    input_ids=input_ids, 
    max_new_tokens=32
)
# alternatively, use faster multibyte generation
generation_output = model.multi_byte_generate(
    input_ids=input_ids, 
    max_new_tokens=32
)

# Decode and print the output
response = tokenizer.decode(
    generation_output[0][input_ids.shape[1]:], 
    skip_special_tokens=False,
    clean_up_tokenization_spaces=False
)
print(response)
# Sample output:
# over the lazy dog.\n\nThe quick
```

### 💬 SFT Model

**Note:** Make sure to set `trust_remote_code=True` when loading the model (or tokenizer), as our implementation includes custom code.

Below (also available in `example_chat.py`) is an example of using **EvaByte-6.5B-SFT** for chat or instruction-following tasks. The model is trained with the [Llama-3 chat format](https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py#L202), and we do not use a specific system prompt during supervised fine-tuning.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("evabyte/EvaByte-SFT", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("evabyte/EvaByte-SFT", torch_dtype=torch.bfloat16, trust_remote_code=True).eval().to("cuda")

# Prepare input messages
messages = [
    {"role": "user", "content": "Write me an English pangram."}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt", 
).to("cuda")

# Byte-by-byte generation (default)
generation_output = model.generate(
    input_ids=input_ids, 
    max_new_tokens=256
)
# Multibyte generation (faster alternative)
generation_output = model.multi_byte_generate(
    input_ids=input_ids, 
    max_new_tokens=256
)

response = tokenizer.decode(
    generation_output[0][input_ids.shape[1]:], 
    skip_special_tokens=False,
    clean_up_tokenization_spaces=False
)
print(response)
# Sample output:
# An English pangram is a sentence that uses every letter of the alphabet at least once. Here's a simple pangram:\n\n"The quick brown fox jumps over the lazy dog."<|eot_id|>
```

### ⚙️ Generation Modes

EvaByte supports two generation interfaces:
- `model.generate()`: The default generation method compatible with Huggingface `transformers` library. This approach generates one byte at a time and might be slow.
- `model.multi_byte_generate()`: A faster alternative that generates multiple bytes per step and usually yields the same result as `model.generate()` under greedy decoding, with the implementation adapted from [Medusa](https://github.com/FasterDecoding/Medusa). `model.multi_byte_generate()` supports a subset of arguments in `model.generate()`:
    - `input_ids`: the input byte ids.
    - `temperature`: the temperature for sampling.
    - `max_length`: the maximum length of the generated sequence.
    - `max_new_tokens`: the maximum number of new bytes to generate.
    - `stopping_criteria`: the [stopping criteria](https://huggingface.co/docs/transformers/v4.47.1/en/internal/generation_utils#transformers.StoppingCriteria) for generation.
    - `top_p`: the top-p parameter for sampling.
    - `do_sample`: greedy decoding or sampling.

**Notes and Limitations:**
- `device_map="auto"` is not supported for >2 GPUs.
- Only batch size of 1 (with `attention_mask=None`) is supported for decoding.
- `torch_dtype=torch.bfloat16` is required.
- The multibyte generation `model.multi_byte_generate()` might return extra bytes after the end-of-sequence sentinel, due to the nature of the multibyte decoding. Manual truncation or cleaning may be needed.

## 📊 Evaluation

For detailed evaluation results, check out our [blog post](https://hkunlp.github.io/blog/2025/evabyte).

- For HumanEval(-Plus), MBPP(-Plus), MATH, and GSM8k tasks, we use a fork of [BigCode Evaluation Harness](https://github.com/bigcode-project/bigcode-evaluation-harness) to evaluate EvaByte.
- For remaining tasks, we use the [OLMES](https://github.com/allenai/olmes/tree/main) evaluation framework.

Evaluation scripts will be released soon.

## 📚 Citation
```bibtex
@misc{evabyte,
    title = {EvaByte: Efficient Byte-level Language Models at Scale},
    url = {https://hkunlp.github.io/blog/2025/evabyte},
    author = {Lin Zheng and Xueliang Zhao and Guangtao Wang and Chen Wu and David Dong and Angela Wang and Mingran Wang and Yun Du and Haige Bo and Tony Zhang and Changran Hu and Urmish Thakker and Lingpeng Kong},
    year = {2025}
}
```
