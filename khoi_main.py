#!/usr/bin/env python3
from pathlib import Path
from genome_data.delete_pls import DNAFormer
import tensorflow as tf
import numpy as np
import argparse

#from luke_preprocessing import MASK_ID


<<<<<<< HEAD:main.py
from luke_preprocessing import encode_fasta_to_kmer_ids,make_contiguous_gaps,kmer_pkl_generation#, make_dataset
from genome_data.model_luke import TransformerModel, loss_function, accuracy_function, mask_seq
=======
from luke_preprocessing import encode_fasta_to_kmer_ids,make_contiguous_gaps,kmer_pkl_generation, make_dataset, make_end_masked_gaps
from khoi_model import TransformerModel, loss_function, accuracy_function, mask_seq
from khoi_lstm import RNNDecoder
>>>>>>> 2dcee8b553a442c7d5b19140d5e94bc6afc199ba:khoi_main.py


def default_settings_and_PARSER() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="default args for transformer model")
    parser.add_argument("--fasta", required=True, help="FASTA path: REQUIRED")
    parser.add_argument("--kmer_length", type=int, default=6)
<<<<<<< HEAD:main.py
    parser.add_argument("--window", type=int, default=256) #change back to 512
=======
    parser.add_argument("--window", type=int, default=32) #change back to 512
>>>>>>> 2dcee8b553a442c7d5b19140d5e94bc6afc199ba:khoi_main.py
    parser.add_argument("--pkl", type=Path, default=Path("pkl_data/two_mer.pkl"))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--min_gap", type=int, required=True, help="min kmer gap: REQUIRED")
    parser.add_argument("--max_gap", type=int, required=True, help="max kmer gap: REQUIRED")
    parser.add_argument("--gap_location", type=str, choices=["end", "random"], required=True, help="gap location choice: random or end")
    parser.add_argument("--model", type=str, required=True, default="lstm", help="model: REQUIRED")
    parser.add_argument("--test_fasta", required=True, help="test_FASTA path: REQUIRED")
    return parser.parse_args()


def make_dataset(kmer_ids: np.ndarray,
                 gap_masks: np.ndarray,
                 batch_size: int,
                 shuffle: bool = False) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        {"ids": kmer_ids.astype(np.int32),
         "mask": gap_masks.astype(np.int32)}
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(kmer_ids))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds



def main():
    # args = default_settings_and_PARSER()

    # if not args.pkl.exists():
    #     print("\npickle path not found, must create")
    #     kmer_pkl_generation(args.kmer_length, str(args.pkl))

    # kmer_inputs, _ = encode_fasta_to_kmer_ids(
    #     fasta_path   = args.fasta,
    #     window_size  = args.window,
    #     kmer_length  = args.kmer_length,
    #     kmer_pkl_path= args.pkl
    # )
    # masks_input = make_contiguous_gaps(kmer_inputs, args.min_gap, args.max_gap)
    # print("k_mer input matrix:", kmer_inputs.shape)
    # print("masks input matrix:", masks_input.shape)

    # # #make model
    # # max_tokens = kmer_inputs.shape[1]   # includes BEGIN/END
    # # model = TransformerModel(seq_len=max_tokens)

    # # dataset = make_dataset(kmer_inputs, masks_input, args.batch)
    # # print("hello")
    # # print(dataset)

    # # #Training
    # # for epoch in range(1, args.epochs + 1):
    # #     print(f"\nEpoch {epoch}/{args.epochs}")
    # #     for batch_ids, batch_masks in dataset:
    # #         model.train(batch_ids, batch_masks, batch_ids.shape[0])
    # #     print()

    # # test_kmer_inputs, _ = encode_fasta_to_kmer_ids(
    # #     fasta_path   = args.test_fasta,
    # #     window_size  = args.window,
    # #     kmer_length  = args.kmer_length,
    # #     kmer_pkl_path= args.pkl
    # # )
    # # test_masks_input = make_contiguous_gaps(test_kmer_inputs, args.min_gap, args.max_gap)

    # # model.test(test_kmer_inputs, test_masks_input, args.batch)

    # vocab_size = kmer_inputs.max() + 1      # or len(kmer_dict) + 3 if you prefer
    # seq_len    = kmer_inputs.shape[1]

    # model = DNAFormer(
    #     vocab_size   = vocab_size,
    #     seq_len      = seq_len,
    #     mask_id      = MASK_ID,
    #     pad_id       = 0,
    #     num_layers   = 4,    # tweak as you like
    #     embed_dim    = 256,
    #     num_heads    = 8,
    #     mlp_dim      = 512,
    #     dropout_rate = 0.1,
    # )

    # model.compile(optimizer=tf.keras.optimizers.Adam())

    # # -----------------------------------
    # # wrap ids & masks in a dict so the custom train_step sees both
    # train_data = tf.data.Dataset.from_tensor_slices(
    #     {"ids": kmer_inputs, "mask": gap_masks}
    # ).batch(batch_size).shuffle(1000)

    # model.fit(train_data, epochs=3)

    # # -----------------------------------
    # # evaluation
    # test_data = tf.data.Dataset.from_tensor_slices(
    #     {"ids": test_inputs, "mask": test_masks}
    # ).batch(batch_size)

    # model.evaluate(test_data)

    args = default_settings_and_PARSER()

    # -------------------------------------------------- k-mer dictionary
    if not args.pkl.exists():
        print("Dictionary pickle missing – generating …")
        kmer_pkl_generation(args.kmer_length, str(args.pkl))

    # -------------------------------------------------- TRAIN data
    train_ids, kmer_dict, mask_id = encode_fasta_to_kmer_ids(
        fasta_path    = args.fasta,
        window_size   = args.window,
        kmer_length   = args.kmer_length,
        kmer_pkl_path = str(args.pkl),
    )
<<<<<<< HEAD:main.py
    print("First mask ID:", mask_id)
    print("First mask ID:", mask_id)
    train_masks = make_contiguous_gaps(
        train_ids, args.min_gap, args.max_gap
=======

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
>>>>>>> 2dcee8b553a442c7d5b19140d5e94bc6afc199ba:khoi_main.py
    )
    #print("train_ids shape :", train_ids.shape)
    #print("train_masks shape:", train_masks.shape)

    # -------------------------------------------------- VAL data
    val_ids, _, mask_id = encode_fasta_to_kmer_ids(
        fasta_path    = args.test_fasta,
        window_size   = args.window,
        kmer_length   = args.kmer_length,
        kmer_pkl_path = str(args.pkl),
    )
    val_masks = make_contiguous_gaps(
        val_ids, args.min_gap, args.max_gap
    )
    print(" mask ID:", mask_id)
    #print("val_ids shape   :", val_ids.shape)
    #print("val_masks shape :", val_masks.shape)

    # -------------------------------------------------- tf.data
    train_ds = make_dataset(train_ids, train_masks, args.batch, shuffle=True)
    val_ds = make_dataset(val_ids,   val_masks,   args.batch, shuffle=False)

    # -------------------------------------------------- Build DNAFormer
    VOCAB_SIZE = max(kmer_dict.keys()) + 1
    seq_len    = train_ids.shape[1]

    model = DNAFormer(
        vocab_size   = VOCAB_SIZE,
        seq_len      = seq_len,
        mask_id      = mask_id,
        pad_id       = 0,
        num_layers   = 4,
        embed_dim    = 256,
        num_heads    = 8,
        mlp_dim      = 512,
        dropout_rate = 0.1,
    )
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam())

    # -------------------------------------------------- Train / Evaluate
    print("\n—— Training ——")
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=args.epochs)

<<<<<<< HEAD:main.py
    print("\n—— Final evaluation ——")
    metrics = model.evaluate(val_ds, return_dict=True)
    print({k: f"{v:.4f}" for k, v in metrics.items()})
=======

    model.test(test_kmer_inputs, test_masks_input, args.batch)
>>>>>>> 2dcee8b553a442c7d5b19140d5e94bc6afc199ba:khoi_main.py


    

if __name__ == "__main__":
    main()
