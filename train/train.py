import wandb
import lightning as L
from peft import LoraConfig
from model import MiniCPMEncoder
from dataset_utils import NLIDataset
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger

batch_size = 12
lr = 1e-3
lora_rank = 8
n_grad_acc = 128
epoch = 1

wandb_logger = WandbLogger()

################################# Logger #################################

config = {
	"batch_size" : batch_size,
	"epoch": epoch,
	"max_lr": lr,
}

wandb.init(
	project="minicpm-dense-retrieval",
	config=config
)

##################################################################

lora_config = LoraConfig(
        init_lora_weights="gaussian",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
        r=lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        inference_mode=False,
    )


dataset = NLIDataset('../data/nli_for_simcse.csv')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


model = MiniCPMEncoder(lora_config=lora_config, dataloader=dataloader, lr=lr, n_grad_acc=n_grad_acc)
trainer = L.Trainer(max_epochs=epoch, logger=wandb_logger, accelerator="cuda", devices=[0], accumulate_grad_batches=n_grad_acc)

trainer.fit(model=model)

wandb.finish()