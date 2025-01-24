# from datasets import load_dataset
import torch
import transformers
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import LlamaTokenizer
from trl import SFTTrainer
from datasets import load_dataset



def formatting_func(example):

  input_prompt = (f" Below is an instruction that describes a task, paired with an input that provides further context. "
  "Write a response that appropriately completes the request.\n\n"
  " ### Instruction:\n"
  f"{example['instruction']}\n\n"
  f"### Input: \n"
  f"{example['context']}\n\n"
  f"### Response: \n"
  f"{example['response']}")


  return {"text" : input_prompt}





def prepare_data(path):
    data = load_dataset("json", data_files=path)
    formatted_data = data.map(formatting_func)
    # print( formatted_data["train"])
    return formatted_data["train"]



train_path="all_training_new.jsonl"
test_path="all_test_new.jsonl"


train = prepare_data(train_path)
dev = prepare_data(test_path)


# model_id = '/scratch/ahcie-gpu2/openllama-models/MedLLaMA_13B'
# model_id = '/scratch/ahcie-gpu2/openllama-models/open_llama_7b_v2'
# model_id="/scratch/ahcie-gpu2/llama-models-meta-hf/Llama-2-13b-hf"
# model_id="/scratch/ahcie-gpu2/llama-models-meta-hf/Llama-2-70b-hf"  #  8192
model_id = "/data/data_user/public_models/Llama3-OpenBioLLM-8B"

qlora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,torch_dtype=torch.float16 
)


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side="right"

supervised_finetuning_trainer=SFTTrainer(
base_model,
train_dataset=train,
eval_dataset=dev,
    args=transformers.TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    logging_steps=1,
    learning_rate=2e-5,
    num_train_epochs=5,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    output_dir="output_dir",
    optim="paged_adamw_8bit",
    fp16=True,
    evaluation_strategy="epoch",
    eval_steps=0.2,
    metric_for_best_model="loss",
    greater_is_better=False,
    save_strategy='epoch',

    save_total_limit=1,
        load_best_model_at_end=True
    ),
tokenizer=tokenizer,
peft_config=qlora_config,
dataset_text_field="text",
max_seq_length=2000
)
supervised_finetuning_trainer.train()

