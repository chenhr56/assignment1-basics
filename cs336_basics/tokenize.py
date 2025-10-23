import os
import pathlib
import pickle

from BPETokenizer import BPETokenizer
import tqdm
import numpy as np

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

TOKENIZER_DIR = pathlib.Path(__file__).resolve().parent.parent / "tokenizer"
VOCAB_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_vocab.pkl")
MERGES_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_merges.pkl")

TRAIN_TXT_DATA_PATH = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt")
VALIDATE_TXT_DATA_PATH = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-valid.txt")

TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train.dat')
VALIDATE_DATA_PATH = os.path.join(DATA_DIR, 'valiadate.dat')

special_tokens = ["<|endoftext|>"]

with open(VOCAB_PATH, 'rb') as f:
    vocab=pickle.load(f)
with open(MERGES_PATH, 'rb') as f:
    merges=pickle.load(f)

tokenizer = BPETokenizer(
    vocab=vocab,
    merges=merges,
    special_tokens=special_tokens
)

print('============== start test tokenizer ==============')
test_texts = [
    "Once upon a time, there was a little robot.",
    "Hello world! <|endoftext|> Some more text.",
    "<|endoftext|>",
    "你好，世界！"
]

for text in test_texts:
    print(f"\n text: {text}")
    encoded = tokenizer.encode(text)
    print(f"encoded: {encoded}")

    byte_tokens = [tokenizer.vocab[token_id] for token_id in encoded]
    str_tokens = [b.decode('utf-8', errors="replace") for b in byte_tokens]
    print(f"str_tokens: {str_tokens}")

    decoded = tokenizer.decode(encoded)
    print(f"decoded: {decoded}")

    print(f"text == decoded: {text == decoded}")

def encode_text2npArray(tokenizer, path2txt, save_path):
    
    with open(path2txt, 'r') as f:
        numlines = sum(1 for _ in f)
    
    tot_tokens = 0
    with open(path2txt, 'r') as f:
        for line in tqdm.tqdm(f, total =  numlines, desc="counting tokens"):
            tot_tokens += len(tokenizer.encode(line))

    tokensmm = np.memmap(save_path, dtype=int32, mopde="w+", shape=(tot_tokens,))
    pos=0
    with open(path2txt, 'r') as f:
        for line in tqdm.tqdm(f, total =  numlines, desc="tokenizing"):
            idxs = tokenizer.encode(line)
            n=len(idxs)
            tokensmm[pos:pos+n] = idxs
            pos+=n
    tokensmm.flush()


def main():
    encode_text2npArray(tokenizer, TRAIN_TXT_DATA_PATH, TRAIN_DATA_PATH)
    encode_text2npArray(tokenizer, VALIDATE_TXT_DATA_PATH, VALIDATE_DATA_PATH)

if __name__ == "__main__":
    main()