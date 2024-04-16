from torch.utils.data import DataLoader
from dataset_utils import NLIDataset
from model import MiniCPMEncoder
from peft import LoraConfig
import lightning as L

batch_size = 16
lr = 0.001
lora_rank = 8


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


model = MiniCPMEncoder(lora_config=lora_config, dataloader=dataloader, lr=lr)
trainer = L.Trainer(max_epochs=1, accelerator="cuda", devices=[0])

trainer.fit(model=model)