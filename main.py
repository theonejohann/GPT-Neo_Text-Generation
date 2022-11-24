from transformers import pipeline
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import tensorflow as tf
import torch

print("Tensorflow Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Torch GPU is available:", torch.cuda.is_available())
print()

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

file = open('input.txt', 'r')
prompt = file.readlines()
prompt = "".join(sentence)
print("*Prompt is:* \n", prompt)
print()

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)

gen_text = tokenizer.batch_decode(gen_tokens)[0]