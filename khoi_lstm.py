import tensorflow as tf
import numpy as np
from khoi_model import loss_function, accuracy_function, mask_seq
import os


class RNNDecoder(tf.keras.layers.Layer):

    def __init__(self, kmer_size, outfile):

        super().__init__()
        self.vocab_size  = 4 ** kmer_size + 2
        self.hidden_size = 512

        self.feed_forward = tf.keras.layers.Dense(self.hidden_size)
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.hidden_size)
        self.decoder = tf.keras.layers.LSTM(self.hidden_size, return_sequences=True)
        self.classification = tf.keras.layers.Dense(self.vocab_size)

        self.optimizer = tf.keras.optimizers.Adam()
        self.outfile = outfile

    def call(self, sequences, masks):
        """
        :param encoded_images: tensor of shape [BATCH_SIZE x 2048]
        :param captions: tensor of shape [BATCH_SIZE x WINDOW_SIZE]
        :return: batch logits of shape [BATCH_SIZE x WINDOW_SIZE x VOCAB_SIZE]
        """
        print("")
        # print(f"actual: {sequences[0]}")
        masked_input = mask_seq(sequences, masks)
        output2 = self.decoder(self.embedding(masked_input))
        output3 = self.classification(output2)
        # print(f"pred: {tf.argmax(output3[0], axis=-1)}")
        # print(f"pred_masked: {tf.where(tf.cast(masks[0], tf.bool), -1, tf.argmax(output3[0], axis=-1))}")
        return output3
    
    def train(self, dataset):
        # print(type(dataset))
        total_loss = total_seen = total_correct = 0
        num_batches = len(dataset)
    
        if not os.path.exists(self.outfile):
            with open(self.outfile, 'a') as f:
                f.write("train_index\tloss\tacc\tbatch_acc\tperp\n")
                
        for index, (batch_ids, batch_masks) in enumerate(dataset):
            with tf.GradientTape() as tape:
                probs = self(batch_ids, batch_masks)
                num_predictions = tf.reduce_sum(tf.cast(batch_masks == 0, tf.float32))
                loss = loss_function(probs, batch_ids, batch_masks)
                accuracy = accuracy_function(probs, batch_ids, batch_masks)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            ## Compute and report on aggregated statistics
            total_loss += loss
            total_seen += num_predictions
            total_correct += num_predictions * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)
            with open(self.outfile, 'a') as f:
                f.write(f"{index+1}\t{avg_loss:.3f}\t{avg_acc:.3f}\t{accuracy:.3f}\t{avg_prp:.3f}\n")
            print(f"\r[Train {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t batch_acc: {accuracy:.3f}\t perp: {avg_prp:.3f}", end='')

            
        return
    
    def test(self, input, mask, batch_size):
        num_batches = int(len(input) / batch_size)

        total_loss = total_seen = total_correct = 0
        with open(self.outfile, 'a') as f:
                f.write("test_index\tloss\tacc\tbatch_acc\tperp\n")
        for index, end in enumerate(range(batch_size, len(input)+1, batch_size)):

            start = end - batch_size
            batch_input = input[start:end, :-1]
            batch_mask = mask[start:end, :-1]

            ## Perform a training forward pass. Make sure to factor out irrelevant labels.
            probs = self(batch_input, batch_mask)
            num_predictions = tf.reduce_sum(tf.cast(batch_mask, tf.float32))
            loss = loss_function(probs, batch_input, batch_mask)
            accuracy = accuracy_function(probs, batch_input, batch_mask)
            
            ## Compute and report on aggregated statistics
            total_loss += loss
            total_seen += num_predictions
            total_correct += num_predictions * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)
            with open(self.outfile, 'a') as f:
                f.write(f"{index+1}\t{avg_loss:.3f}\t{avg_acc:.3f}\t{accuracy:.3f}\t{avg_prp:.3f}\n")
            print(f"\r[TEST {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t batch_acc: {accuracy:.3f}\t perp: {avg_prp:.3f}", end='')

        return