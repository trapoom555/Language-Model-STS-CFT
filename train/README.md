# Training

## Applied Techniques

The following training techniques are applied.

1. Gradient Accumulation : 128 batches (contrastive models require large batch size)
2. LoRA : rank 8
3. Mixed Precision Training : bf16
4. Learning Rate Scheduler : CosineAnnealingLR
