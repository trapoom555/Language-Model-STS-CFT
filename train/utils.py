import torch
import torch.distributed as dist

class AllGather(torch.autograd.Function):
    @staticmethod
    def setup_context(ctx, inputs, output):
        x, = inputs
        ctx.rank = dist.get_rank()
        ctx.local_batch_size = x.shape[0]

    @staticmethod
    def forward(tensor):
        all_tensor = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensor, tensor)
        return torch.cat(all_tensor, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)
        return grad_output[ctx.local_batch_size * ctx.rank: ctx.local_batch_size * (ctx.rank + 1)]