import json
import os
import pathlib
import typing
import sys
# sys.path.append(".")
from cs336_basics.utils import _model_device_and_compile
import torch
import numpy as np
from tqdm import tqdm
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from tests.adapters import get_adamw_cls

from cs336_basics.Loss import cross_entropy
from cs336_basics.Optimizer import gradient_clipping, learning_rate_schedule, AdamW
from cs336_basics.model import TransformerLM
# from tests.adapters import get_adamw_cls

DATA_DIR = pathlib.Path(__file__).parent.resolve().parent / "data"
CONFIG_PATH = pathlib.Path(__file__).parent.resolve() /'config.json'
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train.dat')
VALIDATE_DATA_PATH = os.path.join(DATA_DIR, 'valiadate.dat')
# print(f'data_dir: {DATA_DIR}\nconfig_path: {CONFIG_PATH}\ntrain_data_path: {TRAIN_DATA_PATH}\nvalidate_data_path: {VALIDATE_DATA_PATH}')


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
    # model = model.to(device)

    os.makedirs(args.save_path, exist_ok=True)

    # load dataset
    train_data, validate_data = np.memmap(TRAIN_DATA_PATH, dtype=np.int32, mode='r'), np.memmap(VALIDATE_DATA_PATH, dtype=np.int32, mode='r')

    # build optimator
    adamW = AdamW
    optimizer = adamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # get checkpoint
    start_iter = 0
    if args.get_checkpoint:
        print(f"resume the checkpoint {args.get_checkpoint}")
        cp_path = pathlib.Path(__file__).parent.resolve().parent / f"checkpoints/cp_iter{args.get_checkpoint}.pt"
        start_iter = load_checkpoint(cp_path, model, optimizer)
        print(f"resume the iterator {start_iter}")

    # train
    for iter in tqdm(range(start_iter, args.max_iter), desc='training'):
        model.train()
        x, y = data_loading(train_data, args.batch_size, args.context_length, device)
        # x, y=x.to(device), y.to(device)
        # print(f'Train: x device: {x.device}, y device: {y.device}')
        logics = model(x)
        # print(f"logics device: {logics.device}, y device: {y.device}")
        loss = cross_entropy(
            logics.reshape(-1, logics.shape[-1]), 
            y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()

        gradient_clipping(model.parameters(), args.clip_grad_norm)

        # update learning rate
        lr = learning_rate_schedule(iter, args.lr, args.min_lr, args.warm_iter, args.cosine_iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        # validate
        if (iter + 1) % args.valid_interval == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                count = 0
                for x_val, y_val in data_loading_iterator(validate_data, args.batch_size, args.context_length, device):
                    # x_val, y_val = x_val.to(device), y_val.to(device)
                    val_logics = model(x_val)
                    val_loss = cross_entropy(val_logics.reshape(-1, val_logics.shape[-1]), y_val.reshape(-1))
                    val_losses.append(val_loss.item())
                    count += 1
                    if count >= args.val_batches:
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
