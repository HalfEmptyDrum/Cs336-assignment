import json
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode  # adjust import

def save_bpe(vocab: dict[int, bytes],
             merges: list[tuple[bytes, bytes]],
             vocab_path: str,
             merges_path: str) -> None:
    byte_encoder = gpt2_bytes_to_unicode()  # int -> str (1 char)

    def encode(b: bytes) -> str:
        return "".join(byte_encoder[x] for x in b)

    # {token_str: id}
    encoded_vocab = {encode(tok): idx for idx, tok in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(encoded_vocab, f, ensure_ascii=False)

    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for a, b in merges:
            f.write(f"{encode(a)} {encode(b)}\n")