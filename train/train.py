import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, HfArgumentParser, Trainer, TrainingArguments, set_seed

from contrastive_trainer import ContrastiveTrainer

################################# Model + PEFT #################################

model = AutoModelForCausalLM.from_pretrained('../pretrained/MiniCPM-2B-dpo-bf16/', 
                                            torch_dtype=torch.bfloat16,
                                            trust_remote_code=True,
                                            local_files_only=True)

lora_config = LoraConfig(init_lora_weights="gaussian",
                        task_type=TaskType.CAUSAL_LM,
                        target_modules=["q_proj", "v_proj"],
                        r=8,
                        lora_alpha=32,
                        lora_dropout=0.1,
                        inference_mode=False)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

################################# Data #################################

train_dataset = load_from_disk('../data/processed')



parser = HfArgumentParser((TrainingArguments))

training_args = parser.parse_args_into_dataclasses()[0]

set_seed(training_args.seed)

trainer = ContrastiveTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()