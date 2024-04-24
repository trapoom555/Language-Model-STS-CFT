import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import Trainer

class ContrastiveTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = kwargs.get("args").temperature
    
    def encode(self, model, x):
        out = model(**x, output_hidden_states=True).hidden_states[-1][:, -1, :]
        return out
    
    def all_gather(self, tensor):
        all_tensor = [torch.zeros_like(tensor) for i in range(dist.get_world_size())]
        dist.all_gather(all_tensor, tensor)
        all_tensor[torch.distributed.get_rank()] = tensor
        all_tensor = torch.cat(all_tensor, dim=0)
        return all_tensor

    def info_nce(self, query, pos, neg):
        '''
        Use other samples in batch as negative samples.

        query, pos, neg : [B, E]

        where B is a batch_size, E is an embedding size
        '''
        # Normalize
        query = F.normalize(query, dim=-1)
        pos = F.normalize(pos, dim=-1)
        neg = F.normalize(neg, dim=-1)
        # All gather
        all_query = self.all_gather(query)
        all_pos = self.all_gather(pos)
        all_neg = self.all_gather(neg)
        # Compute cosine sim
        logits_pos = all_query @ all_pos.T
        logits_neg = all_query @ all_neg.T
        # Concat logits
        logits = torch.cat((logits_pos, logits_neg), dim=1)
        # Generate label
        labels = torch.arange(len(all_query)).to(self.accelerator.device)
        # Cross-entropy
        loss = F.cross_entropy(logits / self.temperature, labels, reduction='mean')

        return loss
    
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