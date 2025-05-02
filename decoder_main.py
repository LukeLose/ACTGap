import argparse
import os
import logging
import numpy as np
import tensorflow as tf
from decoder_model import DecoderModel
import decoder_preprocess as preprocess

'''
OVERVIEW: This is the main file that allows the decoder model to run, it uses argparser to read out
essential and optional arguments to detail model training and then records the output on a
final project
'''

#prevents the long warning message about not releveant loss function syntax
tf.get_logger().setLevel(logging.ERROR)

'''
Argument parser: takes in all the necessary arguments to train the the decoder based model, for 
most arguments we dont specifically need to set as we have default model params,
but we have essential outputs like the input fasta
'''
parser = argparse.ArgumentParser()
parser.add_argument("--fasta", required=True, help="training fasta path")
parser.add_argument("--k", type=int, default=2, help="k‑mer size")
parser.add_argument("--gap_len", type=int, default=6)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--val_split", type=float, default=0.15,
                    help="fraction of examples for testing")
parser.add_argument("--outfile_train",   default="results_decoder/train_acc.tsv",
                    help="accuracy output file")
parser.add_argument("--outfile_test",   default="results_decoder/test_acc.tsv",
                    help="accuracy output file")
args = parser.parse_args()

#clean the prior file path and make room for the new one
for pth in (args.outfile_train, args.outfile_test):
    if os.path.exists(pth):
        os.remove(pth)

#build the token to bases dictionary allowing for decoding,
#pull max vocab size as kmer size plus pad and gap tokens
#set vector space and amt of padding equal to largest sequence
#on the fasta
token_to_bases, _ = preprocess.build_kmer_dictionary(args.k)
vocab_size = max(token_to_bases.values()) + 2
GAP_ID = vocab_size - 1
seq_len = preprocess.max_length_helper(args.fasta, args.k)

#prep datasets
print("Creating train and test datasets")
#pulls fasta and produces an encompassing dataset wich we later split on
all_context = preprocess.fasta_to_inputs(args.fasta, token_to_bases, args.gap_len, args.k)
#add training and testing blocks
#use a 'known' random to make testing easier
rng = np.random.default_rng(0)
#shuffle all the data for random test-train split
rng.shuffle(all_context)
#take the expected test/train ratio and multiply it by len of data to get split index
split_index = int(len(all_context) * (1 - args.val_split))
train_data = all_context[:split_index]
test_data = all_context[split_index:]

#pull out vectorized versions of our tuple list for processing in our model
train_ds = preprocess.make_dataset(
    train_data, GAP_ID, seq_len, args.batch, shuffle=True)
test_ds = preprocess.make_dataset(
    test_data, GAP_ID, seq_len, args.batch, shuffle=False)

#make model using our achitecture with room for model customization
#set loss as the standard sparse CE loss function
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

'''
masked loss is just using our loss fn to find the mean loss
while filtering out tokens that were not relevant to the current
sequence, basically pulling out all padding tokens and making
sure they are not used in the loss calculations
'''
@tf.function
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

'''
masked acc is just using our acc fn to find the mean acc
while filtering out tokens that were not relevant to the current
sequence, basically pulling out all padding tokens and making
sure they are not used in the acc calculations
'''
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

#set optimizer before training
optimizer=tf.keras.optimizers.legacy.Adam(args.lr)

print("training!")
'''
train the model using masked loss and masked accuracy along with typtical TF 
training workflow, also keep important stats that we can output into a accuracy
file which we can use for later visualization
'''
for epoch in range(args.epochs):
    total_loss = 0.0
    total_correct = 0.0
    total_seen = 0.0
    #counts the batches in the training dataset and converts tensor to int
    num_batches = tf.data.experimental.cardinality(train_ds).numpy()

    with open(args.outfile_train, "a") as f:
        f.write(f"Epoch {epoch+1} TRAIN\n")
        f.write("batch\tavg_loss\tavg_acc\tbatch_acc\tperp\n")

    for index, (context, gap) in enumerate(train_ds):
        with tf.GradientTape() as tape:
            logits = model(context, training=True)
            loss = masked_loss(gap, logits)
            batch_acc = masked_accuracy(gap, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        #pull out the mask tokens in order to add to total tokens seen
        #for overarching accuracy
        mask = tf.not_equal(gap, -100)
        mask_float = tf.cast(mask, tf.float32)
        num_tokens_tensor = tf.reduce_sum(mask_float)
        num_tokens = float(num_tokens_tensor)

        total_loss += float(loss) * num_tokens
        total_correct += float(batch_acc) * num_tokens
        total_seen += num_tokens

        avg_loss = total_loss / total_seen
        avg_acc = total_correct / total_seen
        perp = np.exp(avg_loss)

        #consistent output with the accuracy/loss categories we established 
        #at the beginning of training
        with open(args.outfile_train, "a") as f:
            f.write(f"{index+1}\t{avg_loss:.3f}\t{avg_acc:.3f}\t"
                    f"{batch_acc:.3f}\t{perp:.3f}\n")
        print(f"\r[TRAIN {index+1}/{num_batches}]"f"  loss={avg_loss:.3f}"f"  acc={avg_acc:.3f}"f"  batch_acc={batch_acc:.3f}"f"  perp={perp:.3f}",end="")

    print("testing!")
    '''
    test the model using masked loss and masked accuracy along with typtical TF 
    testing workflow, also keep important stats that we can output into a accuracy
    file which we can use for later visualization. The split of training and testing 
    dataset is a parameter of our model!
    '''
    total_loss = 0.0
    total_correct = 0.0
    total_seen = 0.0
    #counts the batches in the training dataset and converts tensor to int
    num_batches = tf.data.experimental.cardinality(test_ds).numpy()

    with open(args.outfile_test, "a") as f:
        f.write(f"Epoch {epoch+1} TEST\n")
        f.write("batch\tavg_loss\tavg_acc\tbatch_acc\tperp\n")

    for index, (context, gap) in enumerate(test_ds):
        logits = model(context, training=False)
        batch_loss = masked_loss(gap, logits)
        batch_acc = masked_accuracy(gap, logits)

        #pull out the mask tokens in order to add to total tokens seen
        #for overarching accuracy
        mask = tf.not_equal(gap, -100)
        mask_float = tf.cast(mask, tf.float32)
        num_tokens_tensor = tf.reduce_sum(mask_float)
        num_tokens = float(num_tokens_tensor)

        total_loss += float(batch_loss) * num_tokens
        total_correct += float(batch_acc) * num_tokens
        total_seen += num_tokens

        avg_loss = total_loss / total_seen
        avg_acc = total_correct / total_seen
        perp = np.exp(avg_loss)

        #consistent output with the accuracy/loss categories we established 
        #at the beginning of training
        with open(args.outfile_test, "a") as f:
            f.write(f"{index+1}\t{avg_loss:.3f}\t{avg_acc:.3f}\t"
                    f"{batch_acc:.3f}\t{perp:.3f}\n")
        print(f"\r[TEST {index+1}/{num_batches}]"f"  loss={avg_loss:.3f}"f"  acc={avg_acc:.3f}"f"  batch_acc={batch_acc:.3f}"f"  perp={perp:.3f}",end="")

#end message for clarity that we ran through the whole model w/o issue!
print("complete runthrough")