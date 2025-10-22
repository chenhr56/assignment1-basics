import json
import os
import pathlib
import typing
from .utils import _model_device_and_compile
import torch
import numpy as np
from tqdm import tqdm

from cs336_basics.Loss import cross_entropy
from cs336_basics.Optimizer import gradient_clipping, learning_rate_schedule
from cs336_basics.Transformer import TransformerLM
from tests.adapters import get_adamw_cls

DATA_DIR = pathlib.Path(__file__).parent.resolve().parent / "data"
CONFIG_PATH = 'scripts/config.json'
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train.dat')
VALIDATE_DATA_PATH = os.path.join(DATA_DIR, 'valiadate.dat')


def data_loading(dataset, batch_size: int, context_length: int, device: str):
    starts = np.random.randint(0, len(dataset) - context_length, batch_size)
    data1 = np.stack([dataset[start:start + context_length] for start in starts])
    data2 = np.stack([dataset[start + 1:start + context_length + 1] for start in starts])

    return (
        torch.from_numpy(data1).long().to(device),  # 移动到指定设备: to(device)
        torch.from_numpy(data2).long().to(device),
    )


def data_loading_iterator(dataset, batch_size: int, context_length: int, device: str, num_batch: int = -1):
    N = len(dataset)
    num_batch = (N - context_length - 1) // batch_size
    for bi in range(num_batch):
        base = bi * batch_size
        data1 = np.stack([dataset[start:start + context_length] for start in range(base, base + batch_size)])
        data2 = np.stack([dataset[start + 1:start + context_length + 1] for start in range(base, base + batch_size)])
        yield (
            torch.from_numpy(data1).long().to(device),  # 移动到指定设备: to(device)
            torch.from_numpy(data2).long().to(device),
        )


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], ):
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


def main():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    model = TransformerLM(**config['model'])
    params = dict()
    for group in config.values():
        params.update(group)

    class attrDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    args = attrDict(params)
    model, device = _model_device_and_compile(model)

    os.makedirs(args.save_path, exist_ok=True)

    # load dataset
    train_data, validate_data = np.memmap(TRAIN_DATA_PATH), np.memmap(VALIDATE_DATA_PATH)

    # build optimator
    AdamW = get_adamw_cls()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # get checkpoinnt
    start_iter = 0
    if args.get_checkpoint:
        print(f"resume the checkpoint {args.get_checkpoint}")
        cp_path = pathlib.Path(__file__).parent.resolve().parent / f"checkpoints/cp_iter{args.get_checkpoint}.pt"
        start_iter = load_checkpoint(cp_path, model, optimizer)
        print(f"resume the iterator {start_iter}")

    # train
    for iter in tqdm(range(start_iter, args.max_iter), desc='training'):
        model.train()
        x, y = data_loading(train_data, args.batch_size, args.context_length)
        logics = model(x)
        loss = cross_entropy(logics.reshape(-1, logics.shape[-1]), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()

        gradient_clipping(model.parameters(), args.clip_grad_norm)

        # update learning rate
        lr = learning_rate_schedule(iter, args.max_lr, args.min_lr, args.warm_iter, args.cosine.iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        # validate
        if (iter + 1) % args.save_interval == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                count = 0
                for x_val, y_val in data_loading_iterator(validate_data, args.batch_size, args.context_length):
                    val_logics = model(x_val)
                    val_loss = cross_entropy(val_logics.reshape(-1, val_logics.shape[-1]), y_val.reshape(-1))
                    val_losses.append(val_loss.item())
                    count += 1
                    if count >= args.max_val_batch:
                        break
                val_losses_mean = np.mean(val_losses)
                print(f"iter: {iter}, mean validate loss: {val_losses_mean:.4f}")

        # save
        if (iter + 1) % args.save_interval == 0:
            cp_name = os.path.join(args.save_path, f"cp_iter{iter + 1}.pt")
            save_checkpoint(model, optimizer, iter, cp_name)
            print(f"save checkpoint: {cp_name}")


if __name__ == '__main__':
    main()
