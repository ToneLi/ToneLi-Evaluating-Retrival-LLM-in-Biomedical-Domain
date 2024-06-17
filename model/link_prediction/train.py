import torch
import transformers
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import LlamaTokenizer
from trl import SFTTrainer
from datasets import load_dataset

# transformers==4.31.0


model_id = "chaoyi-wu/MedLLaMA_13B"

qlora_config = LoraConfig(
    r=64,
    lora_alpha=32,
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

tokenizer = LlamaTokenizer.from_pretrained(model_id)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,torch_dtype=torch.float16, device_map='auto',
)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})


def formatting_func(example):
    if example.get("context", "") != "":
        input_prompt = (f"Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        f"### Input: \n"
        f"{example['context']}\n\n"
        f"### Response: \n"
        f"{example['response']}")

    else:
        input_prompt = (f"Below is an instruction that describes a task. "
          "Write a response that appropriately completes the request.\n\n"
          "### Instruction:\n"
          f"{example['instruction']}\n\n"
          f"### Response:\n"
          f"{example['response']}")

    return {"text": input_prompt}


def prepare_data(path, splite):
    data = load_dataset("json", data_files=path)
    formatted_data = data.map(formatting_func)
    # print( formatted_data["train"])
    return formatted_data[splite]


def main(train_path, test_path):
    train_path = "hetionet5k_lp_train_instruction_instructor.json"
    test_path = "hetionet5k_lp_test_instruction_instructor.json"
    output_path = "hetionet5k_model_instructor_medllama"
    train = prepare_data(train_path, "train")
    test = prepare_data(test_path, "train")

    supervised_finetuning_trainer = SFTTrainer(
        base_model,
        train_dataset=train,
        eval_dataset=test,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            max_steps=5000,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            output_dir=output_path,
            save_steps=200,
            logging_steps=100,
            optim="paged_adamw_8bit",
            fp16=True,
        ),
        tokenizer=tokenizer,
        peft_config=qlora_config,
        dataset_text_field="text",
        max_seq_length=4098
    )

    supervised_finetuning_trainer.train()


if __name__ == "main":
    train_path = "hetionet5k_lp_train_instruction_instructor.json"
    test_path = "hetionet5k_lp_test_instruction_instructor.json"
    main(train_path, test_path)

