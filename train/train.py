import argparse
import lightning as L
from peft import LoraConfig
from model import MiniCPMEncoder
from dataset_utils import NLIDataset
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

################################# Parser #################################

parser = argparse.ArgumentParser(description='MiniCPM Training Script')
parser.add_argument('--desire_batch_size', required=True)
parser.add_argument('--init_lr', required=True)
parser.add_argument('--final_lr', required=True)
parser.add_argument('--lora_rank', required=True)
parser.add_argument('--max_epoch', required=True)
parser.add_argument('--temperature', required=True)
parser.add_argument('--batch_size_per_gpu', required=True)
parser.add_argument('--logging', required=True)
args = vars(parser.parse_args())

DESIRE_BATCH_SIZE = int(args['desire_batch_size'])
N_GPUS = 3
LR = float(args['init_lr'])
FINAL_LR = float(args['final_lr'])
LORA_RANK = int(args['lora_rank'])
MAX_EPOCH = int(args['max_epoch'])
TEMPERATURE = float(args['temperature'])

BATCH_SIZE = int(args['batch_size_per_gpu'])  # 8 is safe for RTX3090 (RAM Limit)
N_GRAD_ACC = int(DESIRE_BATCH_SIZE / N_GPUS / BATCH_SIZE)
IS_LOG = (True if args['logging'] == 'true' else False)

################################# Logger #################################

config = {
    "desire_batch_size" : DESIRE_BATCH_SIZE,
    "init_lr": LR,
    "final_lr": FINAL_LR,
    "batch_size_per_gpu" : BATCH_SIZE,
    "lora_rank" : LORA_RANK,
    "epoch": MAX_EPOCH,
    "n_grad_acc": N_GRAD_ACC,
    "n_gpus" : N_GPUS,
    "temperature" : TEMPERATURE
}

if IS_LOG:
    logger = WandbLogger(project="minicpm-dense-retrieval")
else:
    logger = None
      
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

dataset = NLIDataset("../data/processed")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

model = MiniCPMEncoder(lora_config=lora_config,
                       dataloader=dataloader,
                       lr=LR,
                       n_grad_acc=N_GRAD_ACC,
                       max_epochs=MAX_EPOCH,
                       final_lr = FINAL_LR,
                       n_gpus=N_GPUS,
                       temperature=TEMPERATURE)

trainer = L.Trainer(
        max_epochs=MAX_EPOCH, 
        logger=logger,
        log_every_n_steps=1,
        accelerator="cuda", 
        devices=[0, 1, 2], 
        accumulate_grad_batches=N_GRAD_ACC, 
        callbacks=[checkpoint_callback],
        precision="bf16-mixed",
        strategy="ddp")

if IS_LOG and (trainer.global_rank == 0):
    logger.experiment.config.update(config)

trainer.fit(model=model)