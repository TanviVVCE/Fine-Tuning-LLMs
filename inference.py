import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer
from huggingface_hub import notebook_login
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from huggingface_hub import login
import torch
from peft import PeftModelForCausalLM


#SAME AS BEFORE
base_model_id = "meta-llama/Llama-2-13b-chat-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True, # Load model in 4 bit
    bnb_4bit_use_double_quant= True, # Double quantise it to save space
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config = bnb_config)

tokenizer = LlamaTokenizer.from_pretrained(base_model_id,trust_remote_code=True,add_eos_token=True)

#NOTE: STEP - 5 - Update weights
modelFinetuned = PeftModelForCausalLM.from_pretrained(base_model,"./finetunedModel/checkpoint-20") # This is where our LoRA Adapters are merged with the base model as we had dome 20 steps in each epoch for 3 times all the finetuned weights are stored in ./finetunedModel/checkpoint-20

#NOTE: STEP - 6 - INFERENCE
### ENTER YOUR QUESTION BELOW

question = "Just answer this question: Tell me about the role of Maui Emergency Management Agency (MEMA) in the 2023 wildfires??"

# Format the question
eval_prompt = f"{question}\n\n Just answer this question accurately and correctly"

promptTokenized = tokenizer(eval_prompt, return_tensors="pt").to("mps") #Now I intially tried on google COLAB which was working welll because of CUDA Support but we must understand that MPS cannot support certain functionalities because of hardware limitations

# Thats why I would enchorage you to have NVIDIA supported Laptops to carry on this process efficiently 
modelFinetuned.eval()
with torch.no_grad():
    print(tokenizer.decode(modelFinetuned.generate(**promptTokenized, max_new_tokens = 1024)[0], skip_special_tokens=True))
