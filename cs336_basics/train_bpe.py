import os
import sys
import pathlib
import pickle
from BPETokenizer import train_BPE

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
INPUT_PATH = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt")

TOKENIZER_DIR = pathlib.Path(__file__).resolve().parent.parent / "tokenizer"
VOCAB_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_vocab.pkl")
MERGES_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_merges.pkl")

vocab_size  =10000
special_tokens = ["<|endoftext|>"]

vocab, merges = train_BPE(
    input_path = INPUT_PATH,
    vocab_size = vocab_size,
    special_tokens = special_tokens
)

os.makedirs(TOKENIZER_DIR, exist_ok=True)
with open(VOCAB_PATH, 'wb') as f:
    pickle.dump(vocab, f)
with open(MERGES_PATH, 'wb') as f:
    pickle.dump(merges, f)

longest_token = max(vocab.values(), key=len)
print(f"longest token: {longest_token}, length: {len(longest_token)}")