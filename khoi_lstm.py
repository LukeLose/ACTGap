import tensorflow as tf
import numpy as np
from khoi_model import loss_function, accuracy_function, mask_seq

kmer_size = 2
vocab_size = 4 ** kmer_size + 2
hidden_size = 512
window_size = 10


class RNNDecoder(tf.keras.layers.Layer):

    def __init__(self):

        super().__init__()
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        self.feed_forward = tf.keras.layers.Dense(self.hidden_size)
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.hidden_size)
        self.decoder = tf.keras.layers.LSTM(self.hidden_size, return_sequences=True)
        self.classification = tf.keras.layers.Dense(self.vocab_size)

        self.optimizer = tf.keras.optimizers.Adam()
        
    def call(self, sequences, masks):
        """
        :param encoded_images: tensor of shape [BATCH_SIZE x 2048]
        :param captions: tensor of shape [BATCH_SIZE x WINDOW_SIZE]
        :return: batch logits of shape [BATCH_SIZE x WINDOW_SIZE x VOCAB_SIZE]
        """
        print("")
        print(f"actual: {sequences[0]}")
        masked_input = mask_seq(sequences, masks)
        output2 = self.decoder(self.embedding(masked_input))
        output3 = self.classification(output2)
        print(f"pred: {tf.argmax(output3[0], axis=-1)}")
        print(f"pred_masked: {tf.where(tf.cast(masks[0], tf.bool), -1, tf.argmax(output3[0], axis=-1))}")
        return output3
    
    def train(self, dataset):
        print(type(dataset))
        total_loss = total_seen = total_correct = 0
        num_batches = len(dataset)
    
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
            print(f"\r[Train {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        return