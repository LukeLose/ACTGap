import tensorflow as tf
import numpy as np

# hyperparameters
kmer_size = 3
vocab_size = 4 ^ kmer_size
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
        self.pos_encoding = positional_encoding(length=seq_len, depth=embed_size)[..., :seq_len, :]

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.pos_encoding = PositionalEncoding(vocab_size, embed_size)
    
    def call(self, input_seq, mask):
        x = self.pos_encoding(input)
        return x
