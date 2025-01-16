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
response = tokenizer.decode(
    generation_output[0][input_ids.shape[1]:], 
    skip_special_tokens=False,
    clean_up_tokenization_spaces=False
)

print(f"User: {messages[0]['content']}\n")
print("===========================================")
print("[Byte-by-byte generation] via model.generate():")
print(f"> Assistant: {response}")

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
print("===========================================")
print("[Multibyte generation] via model.multi_byte_generate():")
print(f"> Assistant: {response}")
print("===========================================")
