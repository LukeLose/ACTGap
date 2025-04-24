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

class TransformerModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.pos_encoding = PositionalEncoding(vocab_size, embed_size, seq_len)
        self.feed_forward1 = tf.keras.layers.Dense(units=embed_size, activation='relu')
        self.feed_forward2 = tf.keras.layers.Dense(units=embed_size, activation='relu')
        self.norm_layer1 = tf.keras.layers.LayerNormalization()
        self.norm_layer2 = tf.keras.layers.LayerNormalization()
        self.attention = tf.keras.layers.MultiHeadAttention(num_attention_heads, key_dim=64)
    
    def call(self, input_seq, mask=None):
        masked_input = mask_seq(input_seq, mask)
        x = self.pos_encoding(masked_input)
        attention_output = self.attention(x, x, x, attention_mask = mask)
        residuals = self.norm_layer1(x + attention_output)
        output = self.feed_forward1(residuals)
        output = self.feed_forward2(output)
        output = self.norm_layer2(output)
        output = tf.nn.relu(output)
        return x

sample_input = tf.constant([[1, 5, 6, 7]])
sample_mask = tf.constant([[1, 0, 0, 1]])
model = TransformerModel()
print(model(sample_input, sample_mask))
