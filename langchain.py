# from transformers import pipeline
# from langchain_huggingface import HuggingFacePipeline
# from langchain.prompts import PromptTemplate
# import torch

# print(torch.mps.is_available())
# device = torch.device("mps")

# model = pipeline(task="text-generation", model="FacebookAI/roberta-large-mnli", device=device)
# response = model("This is a good restaurant")
# print(response)

# Just an experiment with langchain and hugging face :)
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
  'Nanbeige/Nanbeige4.1-3B',
  use_fast=False,
  trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
  'Nanbeige/Nanbeige4.1-3B',
  torch_dtype='auto',
  device_map='auto',
  trust_remote_code=True
)

llm = HuggingFacePipeline(pipeline=model)

messages = [
  {'role': 'user', 'content': 'Which number is bigger, 9.11 or 9.8?'}
]

prompt = tokenizer.apply_chat_template(
  messages,
  add_generation_prompt=True,
  tokenize=False
)

input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids
output_ids = model.generate(input_ids.to('mps'), eos_token_id=166101)
resp = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
print(resp)