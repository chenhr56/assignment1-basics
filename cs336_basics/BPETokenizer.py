import os
import time

import numpy as np
import regex
from collections import defaultdict, Counter
from multiprocessing.context import Process
from multiprocessing import Queue, cpu_count
from typing import BinaryIO, Iterable, Iterator, Tuple

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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
    # print('start init vocab: {}', time.time())
    special_token_bytes = [token.encode('utf-8') for token in special_tokens]
    vocab = {x: bytes([x]) for x in range(256)}
    last_tokenID = 256
    for token_bytes in special_token_bytes:
        if token_bytes not in vocab.values():
            vocab[last_tokenID] = token_bytes
            last_tokenID += 1

    def str2bytes(s: str) -> Tuple[bytes]:
        l = list(tuple(s.encode('utf-8')))
        l = [bytes([x]) for x in l]
        return tuple(l)

    # print('end init vocab and start pre tokenization: {}', time.time())
    # 2. pre tokenization
    pretoken_count = Counter()
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chunks = regex.split("|".join(map(regex.escape, special_tokens)), text)


    for chunk in chunks:
        for m in regex.finditer(PAT, chunk):
            word = m.group(0)
            pretoken_count[str2bytes(word)] += 1

    # print('end pre tokenization and start merge: {}', time.time())
    # 3. merge
    merges = []

    # # --- OPTIMIZATION 1: Build pair_count ONCE before the loop ---
    pair_count = defaultdict(int)
    for token, count in pretoken_count.items():
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            pair_count[pair] += count
    while len(vocab) < vocab_size:
        # --- OPTIMIZATION 2: Efficiently find the best pair in one pass ---
        # This replaces the slow max(values) -> filter -> max(keys)
        if not pair_count:
            break

        best_count = -1
        best_pair = None
        for pair, count in pair_count.items():
            if count > best_count:
                best_count = count
                best_pair = pair
            elif count == best_count:
                # Tie-breaking logic from original code: max(candidates)
                if best_pair is None or pair > best_pair:
                    best_pair = pair

        # If no pairs left or max count is 1 (no benefit in merging), stop.
        if best_pair is None or best_count < 2:
            break

        token1, token2 = best_pair
        newtoken = token1 + token2
        vocab[last_tokenID] = newtoken
        last_tokenID += 1
        merges.append(best_pair)  # Append the pair tuple

        # --- OPTIMIZATION 3: Incrementally update counts ---
        changes = []
        # Find all tokens that need changing (same as before)
        for token, count in pretoken_count.items():
            indexs = [i for i in range(len(token) - 1) if token[i:i + 2] == best_pair]
            if indexs:
                new_pretoken = []
                i = 0
                while i < len(token):
                    if i in indexs:
                        new_pretoken.append(newtoken)
                        i += 2
                    else:
                        new_pretoken.append(token[i])
                        i += 1
                new_pretoken = tuple(new_pretoken)
                changes.append((token, new_pretoken, count))

        # Apply changes and update counts incrementally
        for oldtoken, new_pretoken, count in changes:
            # 1. Update pretoken_count
            # (Note: Fixed potential bug, was 'oldtoken_')
            if oldtoken in pretoken_count:
                del pretoken_count[oldtoken]
            pretoken_count[new_pretoken] = pretoken_count.get(new_pretoken, 0) + count

            # 2. Update pair_count (the new efficient part)
            # Decrement counts from oldtoken's pairs
            for i in range(len(oldtoken) - 1):
                pair = (oldtoken[i], oldtoken[i + 1])
                pair_count[pair] -= count
                # Clean up dict if count drops to 0 or below
                if pair_count[pair] <= 0:
                    del pair_count[pair]

            # Increment counts from new_pretoken's pairs
            for i in range(len(new_pretoken) - 1):
                pair = (new_pretoken[i], new_pretoken[i + 1])
                pair_count[pair] = pair_count.get(pair, 0) + count
    return vocab, merges


class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

        self.byte_special_tokens = {
            token.encode('utf-8'): self.reversed_vocab[token.encode('utf-8')]
            for token in self.special_tokens
        }

        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """
        Given a string, return a list of token IDs.
        """
        # 1. Pre-tokenize text into "words" (byte strings)
        # We use the existing pre_tokenization function
        byte_pretokens = pre_tokenization(
            text, self.special_tokens, special_token_flag=False)
        
        final_tokens = []
        
        for pretoken_bytes in byte_pretokens:
            # --- OPTIMIZATION 2: Fast path for special tokens ---
            special_token_id = self.byte_special_tokens.get(pretoken_bytes)
            if special_token_id is not None:
                final_tokens.append(special_token_id)
                continue

            # 2. Convert word (bytes) to initial list of token IDs
            # e.g., b'hello' -> [104, 101, 108, 108, 111]
            parts_ids = [self.reversed_vocab[bytes([b])] for b in pretoken_bytes]

            if not parts_ids:
                continue

            # --- OPTIMIZATION 3: Efficient O(N^2) merge loop ---
            while True:
                best_rank = float('inf')
                best_pair_idx = -1

                # 3a. Find the best (lowest rank) pair in the current list
                for i in range(len(parts_ids) - 1):
                    # Get the byte representation of the pair of token IDs
                    pair = (self.vocab[parts_ids[i]], self.vocab[parts_ids[i+1]])
                    
                    # Check if this pair has a merge rule
                    rank = self.merge_ranks.get(pair)
                    
                    if rank is not None and rank < best_rank:
                        best_rank = rank
                        best_pair_idx = i
                
                # 3b. If no mergeable pair was found, we're done with this word
                if best_pair_idx == -1:
                    break
                    
                # 3c. We found a pair to merge. Apply it.
                idx1 = parts_ids[best_pair_idx]
                idx2 = parts_ids[best_pair_idx + 1]
                
                # Get the ID for the new merged token
                new_token_bytes = self.vocab[idx1] + self.vocab[idx2]
                new_token_id = self.reversed_vocab[new_token_bytes]
                
                # Replace the pair with the new token
                parts_ids = parts_ids[:best_pair_idx] + [new_token_id] + parts_ids[best_pair_idx + 2:]

            # Add the final merged token IDs for this word to the result
            final_tokens.extend(parts_ids)

        return final_tokens

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
                # Handle out-of-vocab IDs, though this shouldn't happen
                # with a BPE tokenizer trained on bytes
                token = bytes(replace_char, "utf-8")
            tokens += token
        decoded = tokens.decode("utf-8", errors="replace")
        return decoded
