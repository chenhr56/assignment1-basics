import os
import regex
from collections import defaultdict
from multiprocessing.context import Process
from multiprocessing import Queue
from typing import BinaryIO, Iterable, Iterator


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def split_with_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    Split the text on the special tokens
    """
    if not special_tokens:
        return [text]
    sorted_special_tokens = sorted(special_tokens, key=lambda x: -len(x))
    pattern = "|".join(regex.escape(token) for token in sorted_special_tokens)
    return regex.split('('+pattern+')', text)


def pre_tokenization(text: str, special_tokens: list[str] = None, special_token_flag: bool = True):
    # Split on special tokens
    parts = split_with_special_tokens(text, special_tokens)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens = []
    for part in parts:
        if part in special_tokens:
            if not special_token_flag:
                tokens.append([part.encode('utf-8')])
        else:
            text_tokens = regex.findall(PAT, part)
            part_tokens = [s.encode('utf-8') for s in text_tokens]
            tokens.append(part_tokens)
    res = [token for part_tokens in tokens for token in part_tokens]
    return res


def worker(chunk: str, special_tokens: list[str], q: Queue):
    pretokens = pre_tokenization(chunk, special_tokens)
    q.put(pretokens)


def merge(
    count: dict[tuple[int, int], int],
    index_dict: dict[tuple[int, int], set[int]],
    pretokens: list[list[int]],
    max_pair: (int, int),
    new_index: int
):
    index_set = index_dict[max_pair]
    for i in index_set:
        pretoken, new_pretoken = pretokens[i], []
        pos, j, pos_list = 0, 0, []
        while j < len(pretoken):
            if j < len(pretoken) - 1 and (pretoken[j], pretoken[j+1]) == max_pair:
                new_pretoken.append(new_index)
                pos_list.append(pos)
                j += 2
            else:
                new_pretoken.append(pretoken[j])
                j += 1
            pos += 1

        for pos in pos_list:
            count[max_pair] -= 1
            if pos > 0:
                if new_pretoken[pos-1] == new_index:
                    count[(max_pair[1], max_pair[0])] -= 1
                else:
                    count[(new_pretoken[pos-1], max_pair[0])] -= 1
                count[(new_pretoken[pos-1], new_pretoken[pos])] += 1
                index_dict[(new_pretoken[pos-1], new_pretoken[pos])].add(i)

            if pos < len(new_pretoken)-1:
                if new_pretoken[pos+1] == new_index:
                    count[(max_pair[1], max_pair[0])] -= 1
                else:
                    count[(max_pair[1], new_pretoken[pos+1])] -= 1
                count[(new_pretoken[pos], new_pretoken[pos+1])] += 1
                index_dict[(new_pretoken[pos], new_pretoken[pos+1])].add(i)

        pretokens[i] = new_pretoken


def train_BPE(
    vocab_size: int,
    input_path: str,
    special_tokens: list[str] = None,
    **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the training data.
    """
    # 1. init vocab
    special_tokens = [token.encode('utf-8') for token in special_tokens]
    # merge_count = max(vocab_size - len(special_tokens) - 256, 0)
    vocab = {x: bytes([x]) for x in range(256)}
    last_tokenID = 256
    for i, token in enumerate(special_tokens):
        if token not in vocab:
            vocab[last_tokenID] = token
            last_tokenID += 1
    merges = []

    threads = 4
    chunks = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, threads, "<|endoftext|>".encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    processes = []
    q = Queue()
    for chunk in chunks:
        p = Process(target=worker, args=(chunk, special_tokens, q))
        p.start()
        processes.append(p)

    pretokens_list = [q.get() for _ in processes]

    for p in processes:
        p.join()

    pretokens = [token for sublist in pretokens_list for token in sublist]

    # Merge
    counts, index_dict = defaultdict(int), defaultdict(set)

    for i, pretoken in enumerate(pretokens):
        for i1, i2 in zip(pretoken, pretoken[1:]):
            counts[(i1, i2)] += 1
            index_dict[(i1, i2)].add(i)

    for i in range(merge_count):
        max_pair = max(
            counts.items(),
            key=lambda x: (
                x[1],
                vocab[x[0][0]].decode("utf-8", errors="ignore"),
                vocab[x[0][1]].decode("utf-8", errors="ignore")
            )
        )[0]

        i1, i2 = max_pair
        new_idx = 256+len(special_tokens)+i
        vocab[new_idx] = vocab[i1] + vocab[i2]
        merges.append((vocab[i1], vocab[i2]))

        merge(counts, index_dict, pretokens, max_pair, new_idx)

    return (vocab, merges)


class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        reversed_vocab = {v: k for k, v in self.vocab.items()}
        byte_pretokens = pre_tokenization(
            text, self.special_tokens, special_token_flag=False)
        byte_special_tokens = [token.encode(
            'utf-8') for token in self.special_tokens]
        pretokens = []

        for i, pretoken in enumerate(byte_pretokens):
            new_pretoken = []
            if pretoken in byte_special_tokens:
                new_pretoken.append(reversed_vocab[pretoken])
            else:
                for t in pretoken:
                    new_pretoken.append(reversed_vocab[bytes([t])])

            pretokens.append(new_pretoken)

        for i, pretoken in enumerate(pretokens):
            for merge in self.merges:
                new_pretoken, new_index = [], reversed_vocab[merge[0]+merge[1]]
                j = 0
                while j < len(pretoken):
                    if j < len(pretoken)-1 and (self.vocab[pretoken[j]], self.vocab[pretoken[j+1]]) == merge:
                        new_pretoken.append(new_index)
                        j += 2
                    else:
                        new_pretoken.append(pretoken[j])
                        j += 1

                pretoken = new_pretoken

            pretokens[i] = pretoken

        tokens = [token for pretoken in pretokens for token in pretoken]
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for idx in self.encode(text):
                yield idx

    def decode(self, token_ids: list[int]) -> str:
        """
        Given a list of token IDs, return the decoded string.
        """
        tokens = bytes()
        vocab_size = len(self.vocab)
        replace_char = "\uFFFD"
        for token_id in token_ids:
            if token_id < vocab_size:
                token = self.vocab[token_id]
            else:
                token += bytes(replace_char, "utf-8")
            tokens += token
        decoded = tokens.decode("utf-8", errors="replace")
        return decoded
