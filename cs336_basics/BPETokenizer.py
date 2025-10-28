import os
import regex
import tqdm
import heapq
from collections import defaultdict, Counter
from multiprocessing import Pool
from typing import BinaryIO, Iterable, Iterator, Tuple

# 预分词用的正则
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _process_chunk(chunk_data: Tuple[list[str], str]) -> Counter:
    """
    [Map 阶段] 工作进程函数。
    输入: (lines_batch, special_token_pattern_str)
    输出: Counter{ token_seq(tuple[int,...]) : freq }
    这里的 token_seq 是一个 tuple[int,...]，每个 int 是一个 byte 值 (0..255)。
    这样每个 int 都是 Python 复用的小整数对象，极大减少内存重复分配。
    """
    lines_batch, special_token_pattern_str = chunk_data

    # 如果没有特殊 token，就不编译正则，节省一点
    special_tokens_pattern = (
        regex.compile(special_token_pattern_str)
        if special_token_pattern_str
        else None
    )
    word_re = regex.compile(PAT)

    pretoken_count_local = Counter()

    for line in lines_batch:
        # 1. 先按特殊 token 切分（这些特殊 token 本身不计入词频）
        if special_tokens_pattern:
            normal_chunks = special_tokens_pattern.split(line)
        else:
            normal_chunks = [line]

        # 2. 在普通文本块上跑 PAT
        for chunk in normal_chunks:
            if not chunk:
                continue
            for m in word_re.finditer(chunk):
                word = m.group(0)
                # 把这个“词”编码成 bytes，然后把每个 byte 转成 int (0..255)
                # 注意：tuple(b"...") 直接得到 (104,101,108,108,111)
                b = word.encode("utf-8")
                if not b:
                    continue
                token_seq_ids = tuple(b)
                pretoken_count_local[token_seq_ids] += 1

    return pretoken_count_local


def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes
) -> list[int]:
    """
    计算大文件的分块边界（按 special token 对齐）。
    目前主训练流程里没直接用到，但保留接口。
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # 文件总大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # 初始边界猜测
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # 4KB 微块向后找 split_special_token

    for bi in range(1, len(chunk_boundaries) - 1):
        pos = chunk_boundaries[bi]
        file.seek(pos)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                # EOF
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = pos + found_at
                break
            pos += mini_chunk_size

    return sorted(set(chunk_boundaries))


def split_with_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    把整段文本按 special_tokens 切开，并把 special_tokens 自身也保留下来
    （在 encode() 流程下会单独识别成一个token ID）。
    """
    if not special_tokens:
        return [text]
    sorted_special_tokens = sorted(special_tokens, key=lambda x: -len(x))
    pattern = "|".join(regex.escape(token) for token in sorted_special_tokens)
    # 注意：我们想在 encode() 时保留 special token 自身，所以这里用捕获组
    return regex.split("(" + pattern + ")", text)


def pre_tokenization(text: str,
                     special_tokens: list[str] = None,
                     special_token_flag: bool = True) -> list[bytes]:
    """
    推理时（encode用）的小型预分词。
    返回的是一串 bytes，每个 bytes 对应一个“预token”（单词/空白/标点/特殊token）。
    """
    special_tokens = special_tokens or []

    parts = split_with_special_tokens(text, special_tokens)
    tokens = []
    for part in parts:
        if part in special_tokens:
            # special token 整体当成一个原子
            if not special_token_flag:
                tokens.append([part.encode('utf-8')])
        else:
            text_tokens = regex.findall(PAT, part)
            part_tokens = [s.encode('utf-8') for s in text_tokens]
            tokens.append(part_tokens)

    # 展平
    res = [token for part_tokens in tokens for token in part_tokens]
    return res


def train_BPE(
        vocab_size: int,
        input_path: str,
        special_tokens: list[str] = None,
        **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练 BPE 词表与 merge 规则。

    核心特性：
    - 文件按行流式读 + 进程池并行做 Map，极大语料（~15GB）不会一次性塞进内存。
    - 统计的是“预token”（按 PAT 切出来的词/空白/符号），每个预token被表示成
      tuple[int,...]，其中 int 是单字节 (0..255)。小整数是 Python 复用对象，内存友好。
    - BPE 合并循环用堆 + pair2tokens 反向索引，避免每一轮全量重扫。
    """
    special_tokens = special_tokens or []

    # ====== 第1步：初始化 vocab ======
    # 0..255 -> 单字节，后面会不断追加新合并出来的 token
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    last_tokenID = 256

    # 把特殊token也塞进初始词表（整块bytes作为一个新ID）
    for tok in special_tokens:
        tok_b = tok.encode("utf-8")
        # 避免重复塞相同bytes
        already = False
        for v in vocab.values():
            if v == tok_b:
                already = True
                break
        if not already:
            vocab[last_tokenID] = tok_b
            last_tokenID += 1

    # ====== 第2步：并行预分词 (MapReduce) ======
    # 目标：得到 Counter{ token_seq(tuple[int,...]) : freq }
    # 注意：这里的“token_seq”是单个“预token”的字节序列，而不是整句。
    # 例如 "hello" -> (104,101,108,108,111)
    # 在 BPE 里我们就是对这些序列做 pair 合并。

    # special token 的分隔正则；为空时传 ""，worker 会跳过 split
    if special_tokens:
        special_token_pattern_str = "|".join(regex.escape(t) for t in special_tokens)
    else:
        special_token_pattern_str = ""

    num_workers = 6
    batch_size = 500  # 每批1000行扔给一个子进程去Map，避免一次读太多

    print(f"[BPE Train] 并行预分词 (MapReduce) | workers={num_workers}")
    try:
        total_size = os.path.getsize(input_path)
        print(f"[BPE Train] 处理文件: {input_path} ({total_size / (1024**3):.2f} GB)")
    except OSError:
        total_size = 0

    # 读取文件 -> 分批 -> imap_unordered -> 累加Counter
    def _batch_lines(f, _batch_size):
        batch = []
        for line in f:
            batch.append(line)
            if len(batch) >= _batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _task_generator(f, _batch_size):
        for line_batch in _batch_lines(f, _batch_size):
            # 传给子进程的参数必须是可pickle的
            yield (line_batch, special_token_pattern_str)

    pretoken_count_ids = Counter()

    with Pool(num_workers) as pool:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            task_gen = _task_generator(f, batch_size)
            pbar_map = tqdm.tqdm(desc="Pre-tokenizing [Map]", unit="batch")
            for local_counter in pool.imap_unordered(_process_chunk, task_gen):
                pretoken_count_ids.update(local_counter)
                pbar_map.update(1)
            pbar_map.close()

    print(f"[BPE Train] 预分词完成，唯一预token数量：{len(pretoken_count_ids)}")

    # ====== 第3步：建立初始 pair_count / pair2tokens / 最大堆 ======
    # pair_count[(id_a,id_b)] = 全局该pair出现的加权次数
    # pair2tokens[(id_a,id_b)] = { token_seq(tuple[int,...]) 出现过这个pair 的那些token }
    pair_count = defaultdict(int)
    pair2tokens = defaultdict(set)

    for token_seq, cnt_word in pretoken_count_ids.items():
        if len(token_seq) < 2:
            continue
        # 统计相邻pair
        for i in range(len(token_seq) - 1):
            pair = (token_seq[i], token_seq[i + 1])
            pair_count[pair] += cnt_word
            pair2tokens[pair].add(token_seq)

    # 最大堆 (heapq是小根堆，所以压入负数)
    heap = [(-cnt, pair) for pair, cnt in pair_count.items() if cnt > 0]
    heapq.heapify(heap)

    merges: list[tuple[bytes, bytes]] = []

    # ====== 第4步：BPE 主合并循环 ======
    init_vocab_len = len(vocab)
    target_extra = max(0, vocab_size - init_vocab_len)

    print(f"[BPE Train] 开始 BPE 合并循环 (目标 vocab_size={vocab_size}) ...")
    pbar_merge = tqdm.tqdm(total=target_extra, desc="Merging")

    while len(vocab) < vocab_size and pair_count:
        # 4.1 选出当前出现频次最高的 pair（懒更新堆）
        best_pair = None
        best_count = 0
        while heap:
            negcnt, cand_pair = heapq.heappop(heap)
            cnt_now = pair_count.get(cand_pair, 0)
            if cnt_now == -negcnt and cnt_now > 0:
                best_pair = cand_pair
                best_count = cnt_now
                break
            # 否则这是过期的记录，继续pop

        if best_pair is None:
            # 堆空或全过期，没得合并了
            break

        # BPE 通常不会再合并频率太低的 pair
        if best_count < 2:
            break

        left_id, right_id = best_pair

        # new token 的 bytes = 左右两个 token bytes 拼接
        new_token_bytes = left_id + right_id
        new_token_id = last_tokenID
        vocab[new_token_id] = new_token_bytes
        last_tokenID += 1

        # 记录 merge 规则（encode 时会用）
        merges.append((vocab[left_id], vocab[right_id]))
        pbar_merge.update(1)

        # 4.2 只更新“真的包含这个pair”的 token，而不是全表扫描
        affected_tokens = list(pair2tokens.get(best_pair, set()))

        for old_token_seq in affected_tokens:
            cnt_word = pretoken_count_ids.get(old_token_seq)
            if cnt_word is None:
                # 这个 token_seq 也许已经在本轮之前被改写过
                continue

            old_ids = list(old_token_seq)
            L = len(old_ids)
            if L < 2:
                continue

            # ---- 贪心线性合并 best_pair -> new_token_id ----
            merged_ids = []
            i = 0
            changed = False
            while i < L:
                if (
                    i + 1 < L
                    and old_ids[i] == left_id
                    and old_ids[i + 1] == right_id
                ):
                    merged_ids.append(new_token_id)
                    i += 2
                    changed = True
                else:
                    merged_ids.append(old_ids[i])
                    i += 1

            if not changed:
                # 可能 old_token_seq 里其实已经没有这个 pair 了（过期映射）
                continue

            new_token_seq = tuple(merged_ids)

            # ---- 统计该 token 内旧pair / 新pair 的多重集 ----
            old_pairs_list = [
                (old_ids[j], old_ids[j + 1]) for j in range(L - 1)
            ]
            new_pairs_list = [
                (merged_ids[j], merged_ids[j + 1])
                for j in range(len(merged_ids) - 1)
            ]
            old_pair_counter = Counter(old_pairs_list)
            new_pair_counter = Counter(new_pairs_list)

            # ---- 把出现次数整体从旧 token_seq 挪到新 token_seq ----
            del pretoken_count_ids[old_token_seq]
            pretoken_count_ids[new_token_seq] = (
                pretoken_count_ids.get(new_token_seq, 0) + cnt_word
            )

            impacted_pairs = set(old_pair_counter.keys()) | set(new_pair_counter.keys())

            # ---- 从 pair2tokens 里移除旧 token_seq，后面会把新 token_seq 加回去 ----
            for p in impacted_pairs:
                s = pair2tokens.get(p)
                if s is not None and old_token_seq in s:
                    s.discard(old_token_seq)

            # ---- 更新 pair_count：先减旧pair，再加新pair ----
            for p, occ in old_pair_counter.items():
                pair_count[p] -= occ * cnt_word
                if pair_count[p] <= 0:
                    if p in pair_count:
                        del pair_count[p]

            for p, occ in new_pair_counter.items():
                pair2tokens[p].add(new_token_seq)
                pair_count[p] = pair_count.get(p, 0) + occ * cnt_word

            # ---- 把这些受影响的 pair 的最新计数重新压回堆（懒更新）----
            for p in impacted_pairs:
                c = pair_count.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))

        # 这个 pair 已经被全部折叠成 new_token_id，不再需要旧的映射
        pair2tokens[best_pair] = set()

    pbar_merge.close()
    print("[BPE Train] 合并循环完成。"
          f" 最终 vocab 大小 = {len(vocab)}, merges 数量 = {len(merges)}")

    # 返回 vocab 和 merges：
    # vocab: {token_id:int -> bytes}
    # merges: [(bytes_left, bytes_right), ...]  (按合并顺序)
    return vocab, merges


class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        vocab: dict[int, bytes]
        merges: list[(bytes_left, bytes_right)]，按合并顺序
        special_tokens: list[str]
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # bytes -> token_id
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

        # 特殊token单独映射，encode时判断整块命中就直接拿ID
        self.byte_special_tokens = {
            tok.encode("utf-8"): self.reversed_vocab[tok.encode("utf-8")]
            for tok in self.special_tokens
        }

        # pair(bytes,bytes) -> merge次序（rank 越小优先级越高）
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """
        把字符串转成 token_id 序列。
        步骤：
        1. 先用 pre_tokenization 切成“预token”（word/空白/标点/特殊token）
        2. 对于普通token，先拆成单字节ID序列
        3. 按 merge_ranks 反复合并相邻pair，直到没有更高优先级的pair可以合并
        """
        byte_pretokens = pre_tokenization(
            text, self.special_tokens, special_token_flag=False
        )

        final_tokens = []

        for pretoken_bytes in byte_pretokens:
            # 如果是特殊token，整块直接映射为一个ID
            special_token_id = self.byte_special_tokens.get(pretoken_bytes)
            if special_token_id is not None:
                final_tokens.append(special_token_id)
                continue

            # 否则，把 bytes 里的每个字节拆成基本ID
            # 例：b"hello" -> [104,101,108,108,111]
            parts_ids = [
                self.reversed_vocab[bytes([b])]
                for b in pretoken_bytes
            ]

            if not parts_ids:
                continue

            # 按 merges 循环合并
            while True:
                best_rank = float("inf")
                best_pair_idx = -1

                # 找到当前序列里rank最高(数值最小)的pair
                for i in range(len(parts_ids) - 1):
                    pair_bytes = (
                        self.vocab[parts_ids[i]],
                        self.vocab[parts_ids[i + 1]],
                    )
                    rank = self.merge_ranks.get(pair_bytes)
                    if rank is not None and rank < best_rank:
                        best_rank = rank
                        best_pair_idx = i

                if best_pair_idx == -1:
                    # 没得合并了
                    break

                # 合并这对
                id1 = parts_ids[best_pair_idx]
                id2 = parts_ids[best_pair_idx + 1]
                new_token_bytes = self.vocab[id1] + self.vocab[id2]
                new_token_id = self.reversed_vocab[new_token_bytes]

                parts_ids = (
                    parts_ids[:best_pair_idx]
                    + [new_token_id]
                    + parts_ids[best_pair_idx + 2:]
                )

            final_tokens.extend(parts_ids)

        return final_tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for idx in self.encode(text):
                yield idx

    def decode(self, token_ids: list[int]) -> str:
        """
        把 token_id 序列还原成字符串。
        """
        out_bytes = bytearray()
        vocab_size = len(self.vocab)
        replace_char = "\uFFFD".encode("utf-8")  # UTF-8 for U+FFFD

        for tid in token_ids:
            if tid < vocab_size:
                out_bytes += self.vocab[tid]
            else:
                out_bytes += replace_char

        return out_bytes.decode("utf-8", errors="replace")
