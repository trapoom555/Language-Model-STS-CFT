formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time

accelerate launch --config_file ./configs/fsdp_config.yaml train.py \
--output_dir output/$formatted_time/ \
--learning_rate 5e-5 \
--per_device_train_batch_size 10 \
--bf16 \
--gradient_accumulation_steps 1 \
--warmup_steps 100 \
--max_steps 1000 \
--weight_decay 1e-4 \
--lr_scheduler_type "cosine" \
--remove_unused_columns False \
--save_strategy steps --save_steps 500 --seed 7 \
--log_level info --logging_strategy steps --logging_steps 10 --report_to wandb \