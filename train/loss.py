import torch
from torch import nn
from utils import AllGather
import torch.nn.functional as F
import torch.distributed as dist

class InfoNCE(nn.Module):
    def __init__(self, temperature, device):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, query, pos, neg):
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
        labels = (torch.arange(len(query)) + label_offset).to(self.device)
        # Cross-entropy
        loss = F.cross_entropy(logits / self.temperature, labels, reduction='mean')

        return loss