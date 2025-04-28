#!/usr/bin/env python3

from pathlib import Path
import tensorflow as tf
import numpy as np
import argparse


from luke_preprocessing import encode_fasta_to_kmer_ids,make_contiguous_gaps,kmer_pkl_generation, make_dataset
from model import TransformerModel, loss_function, accuracy_function, mask_seq


def default_settings_and_PARSER() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="default args for transformer model")
    parser.add_argument("--fasta", required=True, help="FASTA path: REQUIRED")
    parser.add_argument("--kmer_length", type=int, default=6)
    parser.add_argument("--window", type=int, default=32) #change back to 512
    parser.add_argument("--pkl", type=Path, default=Path("pkl_data/six_mer.pkl"))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--min_gap", type=int, required=True, help="min kmer gap: REQUIRED")
    parser.add_argument("--max_gap", type=int, required=True, help="max kmer gap: REQUIRED")
    return parser.parse_args()

def main():
    args = default_settings_and_PARSER()

    if not args.pkl.exists():
        print("\npickle path not found, must create")
        kmer_pkl_generation(args.kmer_length, str(args.pkl))

    kmer_inputs, _ = encode_fasta_to_kmer_ids(
        fasta_path   = args.fasta,
        window_size  = args.window,
        kmer_length  = args.kmer_length,
        kmer_pkl_path= args.pkl
    )
    masks_input = make_contiguous_gaps(kmer_inputs, args.min_gap, args.max_gap)
    print("k_mer input matrix:", kmer_inputs.shape)
    print("masks input matrix:", masks_input.shape)

    dataset = make_dataset(kmer_inputs, masks_input, args.batch)
    print("hello")
    print(dataset)
    model = TransformerModel()

    #Training
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        for batch_ids, batch_masks in dataset:
            model.train(batch_ids, batch_masks, batch_ids.shape[0])
        print()

    #TAKE THIS OUT, ONLY NEEDED BEFORE IMPLEMENTING SPLITTING OF 80/20

    model.test()

    

if __name__ == "__main__":
    main()
