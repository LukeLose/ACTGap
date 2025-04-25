import os, gzip, random, json, pickle, itertools, urllib.request
from pathlib import Path
from typing import Iterator, List, Dict, Tuple
import numpy as np
import tensorflow as tf

def windows_from_fasta (fasta_path, window_size: int) -> Iterator[str]:
    '''
    breaks the fasta down into window chunks which we can later use to clump into tokens then tokenize
    '''
    opener = gzip.open if fasta_path.endswith((".gz", ".gzip")) else open
    with opener(fasta_path, "rt") as fh:
        fasta_line_buffer = []
        for line in fh:
            if line.startswith(">"):
                #gets the whole sequence on that fasta line
                full_seq = "".join(fasta_line_buffer).upper()
                for i in range(0, len(full_seq) - window_size + 1, window_size):
                    #yield the whole 512 sequence size
                    yield full_seq[i : i + window_size]
                fasta_line_buffer = []
            else:
                fasta_line_buffer.append(line.strip())

        if fasta_line_buffer:
            full_seq = "".join(fasta_line_buffer).upper()

            for i in range(0, len(full_seq) - window_size + 1, window_size):
                yield full_seq[i : i + window_size]

def kmer_pkl_generation(kmer_length: int, output_path: str) -> None:
    alphabet = 'ACTG'
    output_path = Path(output_path)
    kmer_dict = {
        #pair creation
        idx: ''.join(kmer)
        #for all combos of alphabet
        for idx, kmer in enumerate(itertools.product(alphabet, repeat=kmer_length))
    }
    with open(output_path, 'wb') as f:
        pickle.dump(kmer_dict, f)
    print(f"dump {len(kmer_dict)} k-mers of length {kmer_length} to {output_path}")

# def window_to_kmer_ids(
#     window: str,
#     kmer_len: int,
#     str_to_id: dict[str, int]
# ) -> list[int]:
#     n_kmers = len(window) - kmer_len + 1
#     return [
#         str_to_id[window[i : i + kmer_len]]
#         for i in range(n_kmers)
#     ]

def encode_fasta_to_kmer_ids(fasta_path: str, window_size: int, kmer_length: int,
    kmer_pkl_path: str) ->Tuple[np.ndarray, Dict[int, str]]:
    # #build pickle if not done prior
    # if kmer_dict is None:
    #     alphabet = "ACTG"
    #     kmer_dict = {
    #         idx: "".join(kmer)
    #         for idx, kmer in enumerate(itertools.product(alphabet, repeat=kmer_length))
    #     }
    with open(kmer_pkl_path, "rb") as f:
        kmer_dict: Dict[int, str] = pickle.load(f)
    # reverse-lookup for constant time searches in the right direction
    str_to_id: Dict[str, int] = {v: k for k, v in kmer_dict.items()}



    windows = list(windows_from_fasta(fasta_path, window_size))
    n_batches = len(windows)
    if n_batches == 0:
        raise ValueError("No sequence windows produced, something wen horribly wrong")

    n_kmers = window_size - kmer_length + 1
    #2d array where each row is the batch and the columns are kmers
    seq_arr = np.empty((n_batches, n_kmers), dtype=object)

    #make are kmers by slicing are windows
    for j, seq in enumerate(windows):
        seq_arr[j] = [seq[i : i + kmer_length] for i in range(n_kmers)]

    #analagous to seq_arr but for numeric translation of kmers
    kmer_id_arr = np.empty_like(seq_arr, dtype=int)
    for j in range(n_batches):
        for i in range(n_kmers):
            #pulls the 'ACTGAC' value from the kmer
            kmer_str = seq_arr[j, i]
            try:
                kmer_id_arr[j, i] = str_to_id[kmer_str]
            except KeyError:
                raise KeyError("Non-ACTG alpabet string found")

    #add beginning and ending token       

    #next two unused tokens
    beg_id = len(kmer_dict) 
    end_id = len(kmer_dict) + 1
    kmer_dict[beg_id] = "BEGINS"
    kmer_dict[end_id] = "ENDING"

    #add these tokens to the beginning and ending of all our sequences
    kmer_id_arr = np.insert(kmer_id_arr, 0, beg_id, axis=1)
    kmer_id_arr = np.insert(kmer_id_arr, kmer_id_arr.shape[1], end_id, axis=1)

    return kmer_id_arr, kmer_dict


#now let us make our gaps
def make_contiguous_gaps(kmer_id_arr, min_gap=50, max_gap=100):
    batch_size, kmer_len = kmer_id_arr.shape
    masks = np.ones((batch_size, kmer_len), dtype=int)
    for i in range(batch_size):
        #get gap size
        span_len = np.random.randint(min_gap, max_gap + 1)
        #pick anywhere besides the beginning and end tokens
        start = np.random.randint(1, kmer_len - span_len - 1)
        masks[i, start : start + span_len] = 0
    return masks

def make_dataset(kmer_array: np.ndarray, masks_array: np.ndarray, batch_size: int) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((kmer_array, masks_array))
    buff_size = len(kmer_array)
    dataset = dataset.shuffle(buffer_size=buff_size, reshuffle_each_iteration=True)
    #prefetch speeds up the process significantly
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset



if __name__ == "__main__":
    fasta_path     = "genome_data/sequence_first_100k.fasta"
    kmer_len       = 6
    window_size    = 512
    pkl_path  = Path("pkl_data/six_mer.pkl")

    if not pkl_path.exists():
        print("Dictionary missing – generating & pickling …")
        kmer_pkl_generation(kmer_len, pkl_path)

    kmer_ids, kmer_dict = encode_fasta_to_kmer_ids(
        fasta_path       = fasta_path,
        window_size      = window_size,
        kmer_length      = kmer_len,
        kmer_pkl_path    = str(pkl_path)
    )

    masks_array = make_contiguous_gaps(kmer_ids, 50, 100)
    

    print("- Loaded pickle, num k-mers:", len(kmer_dict))
    print("- kmer_ids shape           :", kmer_ids.shape)
    print("- first row (15 entries)   :", kmer_ids[0, :15])
    print("- masking array shape      :", masks_array.shape)
    print("- masking(whole first row  :", masks_array[0,])

