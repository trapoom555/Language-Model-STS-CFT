import torch
from utils import AllGather
import torch.nn.functional as F
import torch.distributed as dist
from transformers import Trainer
from loss import InfoNCE

class ContrastiveTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        temperature = kwargs.get("args").temperature
        self.info_nce = InfoNCE(temperature=temperature,
                                device=self.accelerator.device)
    
    def encode(self, model, x):
        out = model(**x, output_hidden_states=True).hidden_states[-1][:, -1, :]
        return out
    
    def compute_loss(self, model, inputs, return_outputs=False):
        sent0 = {'input_ids': inputs.get('sent0_input_ids'),
                'attention_mask': inputs.get('sent0_attention_mask')}
        sent1 = {'input_ids': inputs.get('sent1_input_ids'),
                'attention_mask': inputs.get('sent1_attention_mask')}
        hard_neg = {'input_ids': inputs.get('hard_neg_input_ids'),
                    'attention_mask': inputs.get('hard_neg_attention_mask')}
        
        sent0_embed = self.encode(model, sent0)
        sent1_embed = self.encode(model, sent1)
        hard_neg_embed = self.encode(model, hard_neg)

        loss = self.info_nce(sent0_embed, sent1_embed, hard_neg_embed)

        return loss