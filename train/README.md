# Training

## How to Train

1. Clone MiniCPM huggingface project to `$PROJ_DIR/pretrained`

```bash
git clone https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16
```
2. Change a tokenizer setting in `tokenizer_config.json`

```json
"add_eos_token": true
```

3. Create a `output` folder inside `$PROJ_DIR/train` folder to save a model checkpoint

```bash
mkdir $PROJ_DIR/train/output
```

4. Make sure you have installed conda environment

```bash
conda env create --file=environment.yml
conda activate dr
```

5. Configure the number of GPUs in your system in `$PROJ_DIR/train/configs/ddp_config.yaml` at the `num_processes` field

6. Run train script

```bash
chmod +x train.sh
./train.sh
```

## Applied Techniques

The following training techniques are applied.

1. Gradient Accumulation
2. LoRA : rank 8
3. Mixed Precision Training : bf16
4. Learning Rate Scheduler : CosineAnnealingLR with Warmup
5. Data Distributed Parallel (DDP)
6. Efficiently calculate global loss by doing `all_gather` from all GPUs