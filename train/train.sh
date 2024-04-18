python train.py \
--batch_size_per_gpu 4 \
--desire_batch_size 512 \
--init_lr 1e-4 \
--final_lr 1e-5 \
--temperature 0.05 \
--lora_rank 8 \
--max_epoch 2 \
--logging false