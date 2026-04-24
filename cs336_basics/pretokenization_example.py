import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

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


# ## Usage
# with open(..., "rb") as f:
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

#     # The following is a serial implementation, but you can parallelize this
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token


from typing import BinaryIO
import regex as re


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

def parallel_pretokenize(input_file: str):
    with open(input_file, "rb") as f:
        results = _parallel_pretokenize(f, num_processes=4, special_tokens=['<|endoftext|>'])
        pretoken_freq: dict[tuple[bytes, ...], int] = {}
        for freq in results:
            for key in freq.keys():
                pretoken_freq[key] = pretoken_freq.setdefault(key, 0) + freq[key]
        return pretoken_freq

if __name__ == "__main__":
    import time

    start = time.perf_counter()
    
    pretoken_freq = parallel_pretokenize("tests/fixtures/corpus.en")
    
    elapsed = time.perf_counter() - start
    print(f"Took {elapsed:.2f} seconds")
    print(len(pretoken_freq))
    
