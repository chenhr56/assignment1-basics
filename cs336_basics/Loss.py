import torch

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # print(f'Loss: cross_entropy, inputs.device: {inputs.device}, targets.device: {targets.device}')
    tmp = inputs.gather(dim=-1, index=targets.unsqueeze(-1))
    logsumexp = torch.logsumexp(inputs, dim=-1, keepdim=True)
    losses = logsumexp-tmp
    return torch.mean(losses)