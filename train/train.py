import wandb
import lightning as L
from peft import LoraConfig
from model import MiniCPMEncoder
from dataset_utils import NLIDataset
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

BATCH_SIZE = 8
LR = 1e-3
LORA_RANK = 8
N_GRAD_ACC = 128
MAX_EPOCH = 5

wandb_logger = WandbLogger(log_model="all")

################################# Logger #################################

config = {
    "batch_size" : BATCH_SIZE,
    "lora_rank" : LORA_RANK,
    "epoch": MAX_EPOCH,
    "max_lr": LR,
    "n_grad_acc": N_GRAD_ACC
}

wandb.init(
    project="minicpm-dense-retrieval",
    config=config
)

##########################################################################

lora_config = LoraConfig(
        init_lora_weights="gaussian",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
        r=LORA_RANK,
        lora_alpha=32,
        lora_dropout=0.1,
        inference_mode=False,
    )

checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        dirpath="./checkpoint",
        filename="minicpm-{step}-{train_loss:.4f}",
        every_n_train_steps=10,
    )

dataset = NLIDataset('../data/nli_for_simcse.csv')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

model = MiniCPMEncoder(lora_config=lora_config, dataloader=dataloader, lr=LR, n_grad_acc=N_GRAD_ACC)
wandb_logger.watch(model, log="all")

trainer = L.Trainer(
        max_epochs=MAX_EPOCH, 
        logger=wandb_logger, 
        accelerator="cuda", 
        devices=[1], 
        accumulate_grad_batches=N_GRAD_ACC, 
        callbacks=[checkpoint_callback],
        precision="bf16"
    )

trainer.fit(model=model)

wandb.finish()