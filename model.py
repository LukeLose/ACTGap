import tensorflow as tf
import numpy as np

# hyperparameters
kmer_size = 3
vocab_size = 4 ** kmer_size
seq_len = 512
embed_size = 512
num_encoders = 3
num_attention_heads = 4
classifier_dim = 64
classifier_num_layers = 3
mask_token = 0
hidden_size = 32

def positional_encoding(length, depth):
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]    
    depths = np.arange(depth)[np.newaxis, :]/depth   

    angle_rates = 1 / (10000**depths)        
    angle_rads = positions * angle_rates    

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, seq_len):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)
        self.pos_encoding = positional_encoding(length=seq_len, depth=embed_size)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

def mask_seq(input, mask):
    mask_pos = mask == 0
    mask_values = tf.fill(input.shape, mask_token)
    return tf.where(mask_pos, mask_values, input)

def loss_function(predicted_seq, actual_seq, mask):
    masked_pred = tf.boolean_mask(predicted_seq, mask)
    masked_act = tf.boolean_mask(actual_seq, mask)
    scce = tf.keras.losses.sparse_categorical_crossentropy(masked_act, masked_pred, from_logits=True)
    return tf.reduce_sum(scce) 

def accuracy_function(probs, inputs, mask):
    correct = tf.cast(tf.argmax(probs, axis=-1), tf.int32) == tf.cast(inputs, tf.int32)
    return tf.reduce_mean(tf.boolean_mask(tf.cast(correct, tf.float32), mask))

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, emb_sz):
        super().__init__()
        self.feed_forward1 = tf.keras.layers.Dense(units=emb_sz, activation='relu')
        self.feed_forward2 = tf.keras.layers.Dense(units=emb_sz, activation='relu')
        self.norm_layer1 = tf.keras.layers.LayerNormalization()
        self.norm_layer2 = tf.keras.layers.LayerNormalization()
        self.attention = tf.keras.layers.MultiHeadAttention(num_attention_heads, key_dim=64)

    def call(self, inputs, mask):
        #print("asdhjk")
        attention_output = self.attention(inputs, inputs, inputs, attention_mask = mask)
        residuals = self.norm_layer1(inputs + attention_output)
        output = self.feed_forward1(residuals)
        output = self.feed_forward2(output)
        output = self.norm_layer2(output)
        output = tf.nn.relu(output)
        #print("qwert")
        return output

class TransformerModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.optimizer = tf.keras.optimizers.Adam()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pos_encoding = PositionalEncoding(vocab_size, embed_size, seq_len)
        self.transformer_blocks = [TransformerBlock(embed_size) for i in range(num_encoders)]
        self.classifier = tf.keras.layers.Dense(units=vocab_size, activation='softmax')


    def call(self, input_seq, mask=None):
        masked_input = mask_seq(input_seq, mask)
        embed_seq = self.pos_encoding(masked_input)
        #print("pppppp")
        for block in self.transformer_blocks:
            #print("12345")
            embed_seq = block(embed_seq, mask)
        #print("ooooooooo")
        logits = self.classifier(embed_seq)
        #print("qqqqqqq")
        return logits
    
    def train(self, input, mask, batch_size):
        num_batches = int(len(input) / batch_size)

        total_loss = total_seen = total_correct = 0
        for index, end in enumerate(range(batch_size, len(input)+1, batch_size)):

            ## Get the current batch of data, making sure to try to predict the next word
            # start = end - batch_size
            # input = input[start:end, :-1]

            ## Perform a training forward pass. Make sure to factor out irrelevant labels.
            with tf.GradientTape() as tape:
                probs = self(input, mask)
                num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
                loss = loss_function(probs, input, mask)
                accuracy = accuracy_function(probs, input, mask)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            ## Compute and report on aggregated statistics
            total_loss += loss
            total_seen += num_predictions
            total_correct += num_predictions * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)
            print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        return



sample_input = tf.constant([[1, 5, 6, 7]])
sample_mask = tf.constant([[1, 0, 0, 1]])
model = TransformerModel()
#print(model(sample_input, sample_mask))
#print("hiiiii")
model.train(sample_input, sample_mask, 1)