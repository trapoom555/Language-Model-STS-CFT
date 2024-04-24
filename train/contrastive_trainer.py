import torch
from utils import AllGather
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
        all_pos = AllGather.apply(pos)
        all_neg = AllGather.apply(neg)
        # Compute cosine sim
        logits_pos = query @ all_pos.T
        logits_neg = query @ all_neg.T
        # Concat logits
        logits = torch.cat((logits_pos, logits_neg), dim=1)
        # Generate label
        local_batch_size = query.shape[0]
        rank = dist.get_rank()
        label_offset = local_batch_size * rank
        labels = (torch.arange(len(query)) + label_offset).to(self.accelerator.device)
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