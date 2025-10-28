import os
import pathlib
import pickle
import timeit
from BPETokenizer import BPETokenizer
import tqdm
import numpy as np
import multiprocessing

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

TOKENIZER_DIR = pathlib.Path(__file__).resolve().parent.parent / "tokenizer"
VOCAB_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_vocab.pkl")
MERGES_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_merges.pkl")

TRAIN_TXT_DATA_PATH = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt")
VALIDATE_TXT_DATA_PATH = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-valid.txt")

OWT_TRAIN_TXT_DATA_PATH = os.path.join(DATA_DIR, "owt_train.txt")
OWT_VALIDATE_TXT_DATA_PATH = os.path.join(DATA_DIR, "owt_valid.txt")

TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_TinyStories.dat')
VALIDATE_DATA_PATH = os.path.join(DATA_DIR, 'valiadate_TinyStories.dat')

OWT_TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_owt.dat')
OWT_VALIDATE_DATA_PATH = os.path.join(DATA_DIR, 'valiadate_owt.dat')

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

def count_lines(filepath):
    print(f"正在计算总行数: {filepath}")
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count

def encode_text2npArray(tokenizer, path2txt, save_path):
    
    num_lines = count_lines(path2txt)
    num_workers = os.cpu_count()
    # chunksize 告诉 Pool 一次向工作进程发送多少行
    # 这比默认 chunksize=1 高效得多
    chunksize = 1000 
    
    print(f"开始第一遍：计数 token... (文件: {path2txt})")
    print(f"使用 {num_workers} 个核心, chunksize={chunksize}")
    tot_tokens = 0
    
    # 4. 使用 pool.imap_unordered 进行并行计数
    with multiprocessing.Pool(num_workers) as pool:
        with open(path2txt, 'r', encoding='utf-8') as f:
            # imap_unordered 最快，因为它不关心顺序
            results_iter = pool.imap_unordered(tokenizer.encode, f, chunksize=chunksize)
            
            # 使用 tqdm 显示进度
            for encoded_line in tqdm.tqdm(results_iter, total=num_lines, desc="计数中"):
                tot_tokens += len(encoded_line)

    print(f"总 token 数: {tot_tokens}")

    print(f"开始第二遍：编码并写入 memmap... (文件: {save_path})")
    
    tokensmm = np.memmap(save_path, dtype=np.int32, mode="w+", shape=(tot_tokens,))
    
    pos = 0
    # 5. 使用 pool.imap (保留顺序) 进行并行写入
    with multiprocessing.Pool(num_workers) as pool:
        with open(path2txt, 'r', encoding='utf-8') as f:
            # imap 保证输出顺序与输入(行)顺序相同
            results_iter = pool.imap(tokenizer.encode, f, chunksize=chunksize)
            
            for idxs in tqdm.tqdm(results_iter, total=num_lines, desc="编码写入中"):
                n = len(idxs)
                tokensmm[pos:pos+n] = idxs
                pos += n
            
    tokensmm.flush()
    print(f"Memmap 刷新完成: {save_path}")


def main():
    file = open("bpe_tokenize.txt", "w+")
    time1 = timeit.default_timer()
    # encode_text2npArray(tokenizer, TRAIN_TXT_DATA_PATH, TRAIN_DATA_PATH)
    # time2 = timeit.default_timer()
    # file.write(f"train TinyStoriesV2-GPT4-train, 耗时: {time2 - time1}s\n")
    # encode_text2npArray(tokenizer, VALIDATE_TXT_DATA_PATH, VALIDATE_DATA_PATH)
    # time3 = timeit.default_timer()
    # file.write(f"train TinyStoriesV2-GPT4-valid, 耗时: {time3 - time2}s\n")

    encode_text2npArray(tokenizer, OWT_TRAIN_TXT_DATA_PATH, OWT_TRAIN_DATA_PATH)
    time4 = timeit.default_timer()
    file.write(f"train owt_train, 耗时: {time4 - time1}s\n")
    encode_text2npArray(tokenizer, OWT_VALIDATE_TXT_DATA_PATH, OWT_VALIDATE_DATA_PATH)
    time5 = timeit.default_timer()
    file.write(f"train owt_valid, 耗时: {time5 - time4}s\n")
    file.close()
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()