# Training

## Applied Techniques

Since a contrastive models requires large batch size, the following techniques are applied.

1. Gradient Accumulation : 128 batches 
2. LoRA : rank 8
3. Mixed Precision Training : bf16
