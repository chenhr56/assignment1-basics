import os
import sys
import pathlib
import pickle
from BPETokenizer import train_BPE

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
TINY_INPUT_PATH = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt")
OWT_INPUT_PATH = os.path.join(DATA_DIR, "owt_train.txt")

TOKENIZER_DIR = pathlib.Path(__file__).resolve().parent.parent / "tokenizer"
TINY_VOCAB_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_vocab.pkl")
TINY_MERGES_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_merges.pkl")

OWT_VOCAB_PATH = os.path.join(TOKENIZER_DIR, "owt_bpe_vocab.pkl")
OWT_MERGES_PATH = os.path.join(TOKENIZER_DIR, "owt_bpe_merges.pkl")

tiny_vocab_size  = 10000
owt_vocab_size = 32000
special_tokens = ["<|endoftext|>"]

vocab_tiny, merges_tiny = train_BPE(
    input_path = TINY_INPUT_PATH,
    vocab_size = tiny_vocab_size,
    special_tokens = special_tokens
)

vocab_owt, merges_owt = train_BPE(
    input_path = OWT_INPUT_PATH,
    vocab_size = owt_vocab_size,
    special_tokens = special_tokens
)

os.makedirs(TOKENIZER_DIR, exist_ok=True)
with open(TINY_VOCAB_PATH, 'wb') as f:
    pickle.dump(vocab_tiny, f)
with open(TINY_MERGES_PATH, 'wb') as f:
    pickle.dump(merges_tiny, f)

with open(OWT_VOCAB_PATH, 'wb') as f:
    pickle.dump(vocab_owt, f)
with open(OWT_MERGES_PATH, 'wb') as f:
    pickle.dump(merges_owt, f)

tiny_longest_token, owt_longeat_token = max(vocab_tiny.values(), key=len), max(vocab_owt.values(), key=len)
print(f"tiny longest token: {tiny_longest_token}, length: {len(tiny_longest_token)}")
print(f"owt longest token: {owt_longest_token}, length: {len(owt_longest_token)}")