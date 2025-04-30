import argparse
import tensorflow as tf
from decoder_model import DecoderModel
import decoder_preprocess as preprocess


#main method to train and run the model

parser = argparse.ArgumentParser()
parser.add_argument("--fasta", required=True, help="training fasta path")
parser.add_argument("--val_fasta", required=True, help="testing fasta path")
parser.add_argument("--k", type=int, default=2, help="k‑mer size")
parser.add_argument("--gap_len", type=int, default=6)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--val_split", type=float, default=0.15,
                    help="fraction of examples for testing")
parser.add_argument("--outfile",   default="train_test.log",
                    help="accuracy output file")
args = parser.parse_args()

#take the
print("Initializing kmer dictionary")
token_to_bases, _ = preprocess.build_kmer_dictionary(args.k)
vocab_size = max(token_to_bases.values()) + 2
GAP_ID = vocab_size - 1
seq_len = preprocess.max_length_helper(args.fasta, args.k)

#prep datasets
print("Creating train and test datasets")
train_ex = preprocess.fasta_to_inputs(args.fasta, token_to_bases, args.gap_len, args.k)
train_ds = preprocess.make_dataset(train_ex, GAP_ID, seq_len, args.batch, shuffle=True)

val_ex = preprocess.fasta_to_inputs(args.val_fasta, token_to_bases, args.gap_len, args.k)
val_ds = preprocess.make_dataset(val_ex, GAP_ID, seq_len, args.batch, shuffle=False)

#make the model
model = DecoderModel(
    vocab_size=vocab_size,
    seq_len=seq_len,
    num_layers=4,
    embed_size=256,
    num_heads=8,
    hidden_size=512,
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none")

def masked_loss(labels, logits):
    #mask out values where we have our known padding value
    mask = tf.not_equal(labels, -100)

    #convert the mask to put put 0s in the padded areas
    safe_labels = tf.where(mask, labels, tf.zeros_like(labels))
    loss = loss_fn(safe_labels, logits)
    #use your padding mask to 0 out loss where padded
    loss = tf.where(mask, loss, 0.0)
    total_loss = tf.reduce_sum(loss)
    #use the mask to find valid tokens
    num_tokens = tf.reduce_sum(tf.cast(mask, tf.float32))
    return total_loss / num_tokens

@tf.function
def masked_accuracy(labels, logits):
    labels = tf.cast(labels, tf.int32)
    #find only the valid non-padded tokens in the labels
    mask = tf.not_equal(labels, -100)
    safe_labels = tf.where(mask, labels, tf.zeros_like(labels))

    #pull the best kmer from the probabilities
    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
    #correct/incorrect tensor
    match = tf.cast(tf.equal(safe_labels, preds), tf.float32)
    #set match to have incorrect if it is padded (but we correct for it later)
    match = tf.where(mask, match, 0.0)


    total_acc = tf.reduce_sum(match)
    #use the mask to find valid tokens
    num_tokens = tf.reduce_sum(tf.cast(mask, tf.float32))
    return total_acc / num_tokens

model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(args.lr),
    loss=masked_loss,
    metrics=[masked_accuracy],
)
a
print("training!")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
)

print("complete runthrough")