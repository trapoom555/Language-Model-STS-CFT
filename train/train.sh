accelerate launch --config_file ./configs/default_config.yaml train.py \
--output_dir output/$formatted_time/ \
--learning_rate 5e-5 --per_device_train_batch_size 8 \
--bf16 --gradient_accumulation_steps 1 --warmup_steps 100 \
--max_steps 3000 --weight_decay 0.01 \
--evaluation_strategy steps --eval_steps 500 \
--save_strategy steps --save_steps 500 --seed 42 \
--log_level info --logging_strategy steps --logging_steps 10 --remove_unused_columns false