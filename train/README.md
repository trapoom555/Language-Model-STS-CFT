# Training

## How to train

1. Clone MiniCPM huggingface project to `$PROJ_DIR/pretrained`

```bash
git clone https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16
```
2. Change a tokenizer setting in `tokenizer_config.json`

```json
"add_eos_token": true
```

3. Create a `checkpoint` folder inside `$PROJ_DIR/train` folder to save a model checkpoint

```bash
mkdir $PROJ_DIR/train/checkpoint
```

4. Run train script

```bash
python train.py
```

## Applied Techniques

The following training techniques are applied.

1. Gradient Accumulation : 128 batches (contrastive models require large batch size)
2. LoRA : rank 8
3. Mixed Precision Training : bf16
4. Learning Rate Scheduler : CosineAnnealingLR

## TODO

Data parallel distributed training