import wandb
import lightning as L
from peft import LoraConfig
from model import MiniCPMEncoder
from dataset_utils import NLIDataset
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

DESIRE_BATCH_SIZE = 1024
N_GPUS = 3
LR = 1e-4
LORA_RANK = 8
MAX_EPOCH = 2

BATCH_SIZE = 4  # Fixed for RTX3090 (RAM Limit)
N_GRAD_ACC = int(DESIRE_BATCH_SIZE / N_GPUS / BATCH_SIZE)


################################# Logger #################################

config = {
    "batch_size" : BATCH_SIZE,
    "lora_rank" : LORA_RANK,
    "epoch": MAX_EPOCH,
    "max_lr": LR,
    "n_grad_acc": N_GRAD_ACC
}

wandb_logger = WandbLogger(project="minicpm-dense-retrieval")

##########################################################################

lora_config = LoraConfig(
        init_lora_weights="gaussian",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
        r=LORA_RANK,
        lora_alpha=32,
        lora_dropout=0.1,
        inference_mode=False)

checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        dirpath="./checkpoint",
        filename="minicpm-{step}-{train_loss:.4f}",
        every_n_train_steps=10)

dataset = NLIDataset('../data/nli_for_simcse.csv')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

model = MiniCPMEncoder(lora_config=lora_config, dataloader=dataloader, lr=LR, n_grad_acc=N_GRAD_ACC)

trainer = L.Trainer(
        max_epochs=MAX_EPOCH, 
        logger=wandb_logger,
        log_every_n_steps=1,
        accelerator="cuda", 
        devices=[1, 2, 3], 
        accumulate_grad_batches=N_GRAD_ACC, 
        callbacks=[checkpoint_callback],
        precision="bf16-mixed",
        strategy="ddp")

if trainer.global_rank == 0:
    wandb_logger.experiment.config.update(config)

trainer.fit(model=model)