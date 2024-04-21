import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, HfArgumentParser, Trainer, TrainingArguments, set_seed

from contrastive_trainer import ContrastiveTrainer

def main(training_args):
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

    ################################# Data #################################

    train_dataset = load_from_disk('../data/processed')

    set_seed(training_args.seed)

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.accelerator.print(f"{trainer.model}")

    # handle PEFT+FSDP case
    trainer.model.print_trainable_parameters()
    if getattr(trainer.accelerator.state, "fsdp_plugin", None):
        from peft.utils.other import fsdp_auto_wrap_policy

        fsdp_plugin = trainer.accelerator.state.fsdp_plugin
        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

    
    print("FSDP Enable", trainer.is_fsdp_enabled)


    # train
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments))
    training_args = parser.parse_args_into_dataclasses()[0]
    main(training_args)