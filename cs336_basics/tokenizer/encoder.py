from typing import Iterable, Iterator
import regex as re

from functools import lru_cache

from tests.common import gpt2_bytes_to_unicode

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_encoded = [tok.encode('UTF-8') for tok in self.special_tokens]
        ix = max(self.vocab.keys()) + 1 # get index definitely not in the vocab.
        for token in self.special_tokens_encoded:
            if token not in self.vocab.values():
                self.vocab[ix] = token
                ix += 1
        self.vtoi = {s:i for (i,s) in self.vocab.items()}
        
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}
        
        self._encode_pretoken = lru_cache(maxsize=None)(self._encode_pretoken_impl)

    def _encode_pretoken_impl(self, pretoken_bytes: bytes) -> tuple[int, ...]:
        pretoken = [bytes([b]) for b in pretoken_bytes]
        pretoken = self._apply_merge(pretoken)
        return tuple(self.vtoi[b] for b in pretoken)
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        import json

        # Vocab: JSON file mapping token string -> integer ID.
        # Stored as strings (with byte-escape encoding) because JSON can't hold raw bytes.
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_json = json.load(f)
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        vocab = {
            idx: bytes([gpt2_byte_decoder[ch] for ch in token_str])
            for token_str, idx in raw_json.items()
        }

        # Merges: text file, one merge per line, two space-separated tokens.
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                a, b = line.split(" ", 1)
                merges.append((a.encode("utf-8"), b.encode("utf-8")))

        return cls(vocab, merges, special_tokens)
    
    def _apply_merge(self, pretoken: list[bytes]) -> list[bytes]:
        if len(pretoken) < 2:
            return pretoken
        while True:
            best_rank = len(self.merges)
            best_idx = -1
            for i in range(len(pretoken) - 1):
                rank = self.merge_rank.get((pretoken[i], pretoken[i + 1]))
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx == -1:
                break
            pretoken = (
                pretoken[:best_idx]
                + [pretoken[best_idx] + pretoken[best_idx + 1]]
                + pretoken[best_idx + 2:]
            )
        return pretoken
    
    def _pretoken_to_integer(self, pretoken: list[bytes]) -> list[int]:
        return [self.vtoi[b] for b in pretoken]
    
    def _merge_with_special_tokens(self, encoded_chunks: list[list[int]], special_matches: list[str]) -> list[int]:
        ix = 0
        final_result: list[int] = []
        while ix < len(encoded_chunks):
            final_result += encoded_chunks[ix]
            if ix < len(special_matches):
                key = special_matches[ix].encode('UTF-8')
                if key == b'':
                    ix += 1
                    continue
                final_result.append(self.vtoi[key])
            ix += 1
            
        return final_result
    
    def encode(self, text: str) -> list[int]:
        special_matches = []
        if len(self.special_tokens) == 0:
            special_chunks = [text]
        else:
            special_pattern = "|".join(
                re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True)
            )
            special_matches = re.findall(special_pattern, text)
            special_chunks = re.split(special_pattern, text)

        encoded_chunks: list[list[int]] = []
        for chunk in special_chunks:
            encoded_chunk: list[int] = []
            for pretoken_match in re.finditer(PAT, chunk):
                encoded_chunk.extend(
                    self._encode_pretoken(pretoken_match[0].encode("utf-8"))
                )
            encoded_chunks.append(encoded_chunk)

        return self._merge_with_special_tokens(encoded_chunks, special_matches)
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
    
    def decode(self, ids: list[int]) -> str:
        byte_seq: bytes = b''
        for id in ids:
            if id not in self.vocab.keys():
                byte_seq += bytes([254])
            else:
                byte_seq += self.vocab[id]
        return byte_seq.decode('UTF-8', errors='replace')
    