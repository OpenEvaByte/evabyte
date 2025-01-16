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
_input_ids = torch.tensor([[1] + [b + 64 for b in prompt.encode("utf-8")]]).to("cuda")
assert torch.allclose(input_ids, _input_ids)

# byte-by-byte generation (default)
generation_output = model.generate(
    input_ids=input_ids, 
    max_new_tokens=32
)

# Decode and print the output
response = tokenizer.decode(
    generation_output[0][input_ids.shape[1]:], 
    skip_special_tokens=False,
    clean_up_tokenization_spaces=False
)
print("===========================================")
print("[Byte-by-byte generation] via model.generate():")
print(prompt)
print(f"> {response}")

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
print("===========================================")
print("[Multibyte generation] via model.multi_byte_generate():")
print(prompt)
print(f"> {response}")
print("===========================================")
