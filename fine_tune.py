import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer
from huggingface_hub import notebook_login
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from huggingface_hub import login
import torch

#NOTE: STEP 1 - Load the Model with relevant configurations 
base_model_id = "meta-llama/Llama-2-13b-chat-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True, # Load model in 4 bit
    bnb_4bit_use_double_quant= True, # Double quantise it to save space
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config = bnb_config)


#NOTE: STEP 2 - Load the dataset
train_dataset = load_dataset("text", data_files={"train":
                ["./Fine-tuning-LLMs/hawaii_wf_1.txt", "./Fine-tuning-LLMs/hawaii_wf_2.txt",
                 "./Fine-tuning-LLMs/hawaii_wf_3.txt","./Fine-tuning-LLMs/hawaii_wf_4.txt",
                 "./Fine-tuning-LLMs/hawaii_wf_5.txt","./Fine-tuning-LLMs/hawaii_wf_6.txt",
                 "./Fine-tuning-LLMs/hawaii_wf_7.txt","./Fine-tuning-LLMs/hawaii_wf_8.txt",
                "./Fine-tuning-LLMs/hawaii_wf_9.txt","./Fine-tuning-LLMs/hawaii_wf_10.txt","./Fine-tuning-LLMs/hawaii_wf_11.txt"]}, split='train')


#NOTE: STEP 3 - Since all the LLM models understand tokens and embeddings we need to convert our dataset into tokens
tokenizer = LlamaTokenizer.from_pretrained(base_model_id,trust_remote_code=True,add_eos_token=True)

#As humans we can understand when a sentence is ending but the LLM doesnt so we need to pad the ending so that LLMs will know that a new sentence starts from here that is after the end of sequence token - Now important thing to keep in mind is that al LLMs have different EOS tokens so just know via "tokenizer.eos_token"

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

tokenized_train_dataset = []
for phrase in train_dataset:
    tokenized_train_dataset.append(tokenizer(phrase["text"]))

#NOTE: STEP 3 - TRAINING STARTS

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(r=8,lora_alpha=64,target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],lora_dropout=0.05, bias=None, task_type= "CAUSAL_LM") # We mention the task_type to be Causal LM because its for a specific task which involves text_generation now suppose you want a model to return generated images then you dont need to specify but if you are using AutoModelForCausalLM then make sure to use PeftModelForCausalLM because that will not only wrap your updated weights but also help in language generation.


# We have different projections like key, value and query projections or matrices in self attention layers, which then are combined to form output projection which basically combines all the words required for communicating with the user. Now the output of self attention model goes to the Feed Forward Network (FNN) which has gated projections which are important to induce certain rules and restrictions to LLMs, Up Projection which is used for analysing each token given by self attention in deeper way to understand and may be tweak as per the creativity and down projection brings back the elaborated form of values back to its original projections. FNN also acts as a memory unit to store certain information like - "Whats the Capital of Canada" - To generate - "Ottawa" LLM doesn't have to run all the layers just FNN Retrieval is sufficient.

model = get_peft_model(model, config) # Suppose you are confused , then this line will figure out if you need PeftModel or PeftModelForCausalLM

#NOTE: STEP 4 - Batching, Forward Pass, Accumulate Gradients

#NOTE: IMPORTANT - We are storing the weights from 3 epochs over 20 steps each in ./fineTunedModel directory, and saving logs in ./log directory.
trainer = transformers.trainer(output_dir = './fineTunedModel' ,model=model, train_dataset= tokenized_train_dataset, args = transformers.TrainingArguments(per_device_eval_batch_size=1, gradient_accumulation_steps=4, num_train_epochs=3, learning_rate= 1e-4, max_steps=20, bf16= False, optim="adamw_torch", logging_dir="./log", save_strategy="epoch", save_steps=50, logging_steps=10),

data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)) # This is used to select the sequence lenght to capture all the sequences and thus set to the largest lenght.

model.config.use_cache = False
trainer.train()






