formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time

accelerate launch --config_file ./configs/ddp_config.yaml train.py \
--output_dir output/$formatted_time/ \
--model_name_or_path ../pretrained/MiniCPM-2B-dpo-bf16/ \
--temperature 0.05 \
--train_data_path ../data/processed \
--learning_rate 5e-5 \
--per_device_train_batch_size 7 \
--bf16 \
--gradient_accumulation_steps 1 \
--warmup_steps 100 \
--max_steps 1000 \
--weight_decay 1e-4 \
--lr_scheduler_type "cosine" \
--lora_r 8 --lora_alpha 32 --lora_dropout 0.1 \
--save_strategy steps --save_steps 500 --seed 7 \
--remove_unused_columns False \
--log_level info --logging_strategy steps --logging_steps 10 --report_to wandb \