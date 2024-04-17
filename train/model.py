import torch
import lightning as L
from peft import LoraModel
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM


class MiniCPMEncoder(L.LightningModule):
    def __init__(self, lora_config, dataloader, lr, n_grad_acc, max_epochs, final_lr, n_gpus, temperature):
        super().__init__()

        path = '../pretrained/MiniCPM-2B-dpo-bf16/'

        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(path, 
                                                     torch_dtype=torch.bfloat16, 
                                                     device_map=self.device, 
                                                     trust_remote_code=True,
                                                     local_files_only=True)
        
        self.lora_model = LoraModel(model, lora_config, "default")
        
        self.lr = lr
        self.final_lr = final_lr
        self.dataloader = dataloader
        self.n_grad_acc = n_grad_acc
        self.max_epochs = max_epochs
        self.n_gpus = n_gpus
        self.temperature = temperature

        self.prompt = """#### Instruct: Given a premise, retrieve a hypothesis that is entailed by the premise Retrieve semantically similar text
        #### Query: {}"""

    def forward(self, x, with_prompt=True):
        if with_prompt:
            x = [self.prompt.format(t) for t in x]
        inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        out = self.lora_model(**inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]
        del inputs
        return out
    
    def info_nce(self, query, pos, neg):
        '''
            Use other samples in batch as negative samples.

            query, pos, neg : [B, E]

            where B is a batch_size, E is an embedding size
        '''

        # normalize
        query = F.normalize(query, dim=-1)
        pos = F.normalize(pos, dim=-1)
        neg = F.normalize(neg, dim=-1)

        # compute cosine sim
        logits_pos = query @ pos.T
        logits_neg = query @ neg.T

        # concat logits
        logits = torch.cat((logits_pos, logits_neg), dim=1)

        # generate label
        labels = torch.arange(len(query), device=self.device)

        # cross-entropy
        loss = F.cross_entropy(logits / self.temperature, labels, reduction='mean')

        return loss


    def training_step(self, batch, batch_idx):
        x, pos, neg = batch

        query_em = self.forward(x, with_prompt=True)
        pos_em = self.forward(pos, with_prompt=False)
        neg_em = self.forward(neg, with_prompt=False)
        
        loss = self.info_nce(query_em, pos_em, neg_em)

        # loggings
        lr_schedule = self.lr_schedulers()
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", lr_schedule.get_lr()[0])

        return loss

    def configure_optimizers(self):
        T_max = int(len(self.train_dataloader()) * self.max_epochs / self.n_grad_acc / self.n_gpus)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = {
                "scheduler": CosineAnnealingLR(optimizer, T_max=T_max, eta_min=self.final_lr),
                "interval": "step",
                "frequency": 1,
            }
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self.dataloader