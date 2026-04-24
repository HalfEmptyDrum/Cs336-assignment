import regex as re




from typing import BinaryIO
from cs336_basics.pretokenization_example import find_chunk_boundaries

from multiprocessing import Pool

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize(text: bytes, special_pattern: str):
    pretoken_freq: dict[tuple[bytes, ...], int] = {}
    chunks = re.split(special_pattern, text)
    for chunk in chunks:
        for item in re.finditer(PAT, chunk):
            item = item[0].encode('utf-8')
            item = tuple(bytes([b]) for b in item)
            pretoken_freq[item] = pretoken_freq.setdefault(item, 0) + 1
    
    return pretoken_freq
    

def _parallel_pretokenize(f: BinaryIO, num_processes: int, special_tokens: list[str]):

    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
    special_pattern = "|".join(re.escape(tok) for tok in special_tokens)
    
    args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        args.append((f.read(end - start).decode('utf-8', errors="ignore"), special_pattern))
    
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(pretokenize, args)
    
    return results

def parallel_pretokenize(input_file: str, special_tokens: list[str]):
    with open(input_file, "rb") as f:
        results = _parallel_pretokenize(f, num_processes=8, special_tokens=special_tokens)
        pretoken_freq: dict[tuple[bytes, ...], int] = {}
        for freq in results:
            for key in freq.keys():
                pretoken_freq[key] = pretoken_freq.setdefault(key, 0) + freq[key]
        return pretoken_freq
    
    
    

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def add_tokens(vocab: dict[int, bytes], ix: int, tokens: list[str]) -> int:
    for token in tokens:
        ix = add_token(vocab, ix, token.encode('utf-8'))
    return ix

def add_token(vocab: dict[int, bytes], ix: int, token: bytes) -> int:
    vocab[ix] = token
    return ix + 1


def train_bpe(input_path: str,
              vocab_size: int,
              special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    
    vocab = {i:i.to_bytes(1, 'big') for i in range(256)}
    
    ix = len(vocab)
    ix = add_tokens(vocab, ix, special_tokens)
    
    freq_table: dict[tuple[bytes, bytes], int] = {}
    
    merges: list[tuple[bytes, bytes]] = []
    

    pretoken_freq = parallel_pretokenize(input_path, special_tokens)

    
    init_phase = True
        
    while len(vocab) < vocab_size:
        # For each of these, do byte pair counting.

        if init_phase:
            init_phase = False
            for pretoken in pretoken_freq.keys():
                for i in range(len(pretoken)-1):
                    byte_pair = (pretoken[i], pretoken[i+1])
                    freq_table[byte_pair] = freq_table.setdefault(byte_pair, 0) + pretoken_freq[pretoken]

        if len(freq_table) == 0:
            break
        token = max(freq_table, key=lambda k: (freq_table[k], k))

        merges.append(token)
        ix = add_token(vocab, ix, token[0]+token[1])
        
        old_new_pretokens : list[tuple[bytes, bytes]] = []
        
        for pretoken in pretoken_freq:
            new_pretoken = ()
            i = 0
            merged = False
            
            # This here saves the pairs of indices that we are going to have to decrease/increase in count.
            index_set = set()
            
            while i < len(pretoken):
                if i < len(pretoken) - 1 and token == (pretoken[i], pretoken[i+1]):
                    new_pretoken = new_pretoken + (pretoken[i] + pretoken[i+1],)
                    
                    merged = True
                    # let's insert the pairs of indices:
                    if i > 0:
                        index_set.add((i-1,i))
                    if i < len(pretoken) - 2:
                        index_set.add((i+1, i+2))
                    i += 1
                else:
                    new_pretoken = new_pretoken + pretoken[i:i+1]
                i += 1
            if merged:
                old_new_pretokens.append((pretoken, new_pretoken, index_set))
        
        for old_pretoken, new_pretoken, index_set in old_new_pretokens:
            pretoken_freq[new_pretoken] = pretoken_freq.pop(old_pretoken)
            # print(old_pretoken, new_pretoken, index_set)
            for i1, i2 in index_set:
                byte_pair = (old_pretoken[i1], old_pretoken[i2])
                freq_table[byte_pair] -= pretoken_freq[new_pretoken]

            
            for i in range(len(new_pretoken)):
                a,b = token
                new_token = a+b
                if new_pretoken[i] == new_token:
                    if i > 0:
                        tmp_token = (new_pretoken[i-1], new_token)
                        freq_table[tmp_token] = freq_table.setdefault(tmp_token, 0) + pretoken_freq[new_pretoken]
                    if i < len(new_pretoken) -1:
                        tmp_token = (new_token, new_pretoken[i+1])
                        freq_table[tmp_token] = freq_table.setdefault(tmp_token, 0) + pretoken_freq[new_pretoken]
            
        
        # let's delete it afterwards.
        freq_table.pop(token)
        



        
    return vocab, merges



if __name__ == "__main__":
    print("training tokenizer...")
    vocab, merges = train_bpe('input.txt', vocab_size=10000, special_tokens=["<|endoftext|>"])
    print("done!")
    print(max(vocab.values(), key=len))
    