import math
from typing import Tuple, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.95),
                 eps: float = 1e-8,
                 weight_decay: float = 0.99):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                # !!! 提前更新状态
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                state = self.state[p]
                grad = p.grad.data
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                t = state.get('t', 1)
                m, v = state['m'], state['v']
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                alpha_t = lr * (math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))
                denom = v.sqrt() + eps
                p.data.addcdiv_(m, denom, value=-alpha_t)
                state['t'] = t + 1

        return loss

def learning_rate_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int):
    if t < T_w:
        return t / T_w * alpha_max
    if t > T_c:
        return alpha_min
    x = math.pi * (t-T_w) / (T_c-T_w)
    return alpha_min + (1 + math.cos(x)) * (alpha_max - alpha_min) / 2

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    paras = [p for p in parameters if p.grad is not None]
    if len(paras) == 0:
        return 
    l2norm = torch.sqrt(sum(torch.sum(p.grad.pow(2)) for p in paras))
    if not l2norm < max_l2_norm:
        eps = 1e-6
        scale = max_l2_norm / (l2norm + eps)
        for p in paras:
            p.grad.data.mul_(scale)
