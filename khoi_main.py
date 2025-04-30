#!/usr/bin/env python3

from pathlib import Path
import tensorflow as tf
import numpy as np
import argparse


from luke_preprocessing import encode_fasta_to_kmer_ids,make_contiguous_gaps,kmer_pkl_generation, make_dataset, make_end_masked_gaps
from khoi_model import TransformerModel, loss_function, accuracy_function, mask_seq
from khoi_lstm import RNNDecoder


def default_settings_and_PARSER() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="default args for transformer model")
    parser.add_argument("--fasta", required=True, help="FASTA path: REQUIRED")
    parser.add_argument("--kmer_length", type=int, default=6)
    parser.add_argument("--window", type=int, default=32) #change back to 512
    parser.add_argument("--pkl", type=Path, default=Path("pkl_data/two_mer.pkl"))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--min_gap", type=int, required=True, help="min kmer gap: REQUIRED")
    parser.add_argument("--max_gap", type=int, required=True, help="max kmer gap: REQUIRED")
    parser.add_argument("--gap_location", type=str, choices=["end", "random"], required=True, help="gap location choice: random or end")
    parser.add_argument("--model", type=str, required=True, default="lstm", help="model: REQUIRED")
    parser.add_argument("--test_fasta", required=True, help="test_FASTA path: REQUIRED")
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

    if args.gap_location == "end": 
        # This adds gaps only to the end of a sequence
        masks_input = make_end_masked_gaps(kmer_inputs, args.min_gap, args.max_gap)
        print("k_mer input matrix:", kmer_inputs.shape)
        print("masks input matrix:", masks_input.shape)
        dataset = make_dataset(kmer_inputs, masks_input, args.batch)
        #(gap of 3) Returns final accuracy of 0.285 and loss of 2.052 and perp of 7.784
    elif args.gap_location == "random":
        # This uses contiguous gaps randomly inserted in sequence
        masks_input = make_contiguous_gaps(kmer_inputs, args.min_gap, args.max_gap)
        print("k_mer input matrix:", kmer_inputs.shape)
        print("masks input matrix:", masks_input.shape)
        dataset = make_dataset(kmer_inputs, masks_input, args.batch)
        #(gap of 3) Returns final accuracy of 0.212 and loss of 2.252 and perp of 9.505


    if args.model == "lstm":
        model = RNNDecoder(args.kmer_length)
    else:
        model = TransformerModel()

    #Training
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        model.train(dataset)
        print()

    #TAKE THIS OUT, ONLY NEEDED BEFORE IMPLEMENTING SPLITTING OF 80/20
    test_kmer_inputs, _ = encode_fasta_to_kmer_ids(
        fasta_path   = args.test_fasta,
        window_size  = args.window,
        kmer_length  = args.kmer_length,
        kmer_pkl_path= args.pkl
    )
    test_masks_input = make_contiguous_gaps(test_kmer_inputs, args.min_gap, args.max_gap)


    model.test(test_kmer_inputs, test_masks_input, args.batch)


    

if __name__ == "__main__":
    main()
