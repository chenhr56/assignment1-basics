import os
import typing

import numpy.typing as npt
import torch
import numpy as np

def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    starts = np.random.randint(0, len(dataset)-context_length, batch_size)
    data1 = np.stack([dataset[start:start + context_length] for start in starts])
    data2 = np.stack([dataset[start+1:start + context_length+1] for start in starts])

    return (
        torch.from_numpy(data1).long().to(device), # 移动到指定设备: to(device)
        torch.from_numpy(data2).long().to(device),
    )

def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],):
    checkpoint = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        iteration=iteration,
    )
    if isinstance(out, (str, os.PathLike)):
        with open(out, 'wb') as f:
            torch.save(checkpoint, f)
    else:
        torch.save(checkpoint, out)

def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
):
    if isinstance(src, (str, os.PathLike)):
        with open(src, 'rb') as f:
            cp = torch.load(f)
    else:
        cp = torch.load(src)
    model.load_state_dict(cp['model'])
    optimizer.load_state_dict(cp['optimizer'])
    return cp['iteration']