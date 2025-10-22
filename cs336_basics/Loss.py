import torch

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    tmp = inputs.gather(dim=-1, index=targets.unsqueeze(-1))
    logsumexp = torch.logsumexp(inputs, dim=-1, keepdim=True)
    losses = logsumexp-tmp
    return torch.mean(losses)
