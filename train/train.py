import os
import torch
import transformers
from typing import Optional
from datasets import load_from_disk
from dataclasses import dataclass, field
from contrastive_trainer import ContrastiveTrainer
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, HfArgumentParser, set_seed

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    lora_alpha: Optional[int] = field(default=32)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=8)
    lora_target_modules: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )

@dataclass
class DataArguments:
    train_data_path: str = field(
        metadata={"help": "Path to training data"}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    temperature: Optional[float] = field(default=0.05)

def main(model_args, data_args, training_args):
    set_seed(training_args.seed)

    # Model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, 
                                                torch_dtype=torch.bfloat16,
                                                trust_remote_code=True,
                                                local_files_only=True)

    # PEFT
    lora_config = LoraConfig(init_lora_weights="gaussian",
                            task_type=TaskType.CAUSAL_LM,
                            target_modules=["q_proj", "v_proj"],
                            r=model_args.lora_r,
                            lora_alpha=model_args.lora_alpha,
                            lora_dropout=model_args.lora_dropout,
                            inference_mode=False)

    model = get_peft_model(model, lora_config)

    # Data
    train_dataset = load_from_disk(data_args.train_data_path)

    trainer = ContrastiveTrainer(model=model,
                                args=training_args,
                                train_dataset=train_dataset)

    trainer.accelerator.print(f"{trainer.model}")
    trainer.model.print_trainable_parameters()

    # Train
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    # Saving final model
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "minicpm-dense-retrieval"
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)