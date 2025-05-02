import tensorflow as tf
import numpy as np

'''
OVERVIEW: This file is for the actual transformer decoder model that is being trained and tested. 
This includes a Positional Encoding class, a Decoder Block class, and Decoder Model class. 
'''

'''
Positional encoding function. 
Takes in length and depth of the embedding matrix and calculates sinusoidal positional encoding.
Returns a positional encoding matrix.
'''
def positional_encoding(length, depth):
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]    
    depths = np.arange(depth)[np.newaxis, :]/depth   
    angle_rates = 1 / (10000**depths)        
    angle_rads = positions * angle_rates    
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
    return tf.cast(pos_encoding, dtype=tf.float32)

'''
Positional Encoding Layer. 
Parameters: vocab size (number of possible tokens, int), embed size (dimension of the embedding layer, int), 
and seq_len (length of the input sequences, int). 
Takes in the input sequences, creates embeddings, and adds positional encoding. 
Returns the embedded inputs with positional encoding.
'''
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, seq_len):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)
        self.pos_encoding = positional_encoding(length=seq_len, depth=embed_size)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)   # create embedding matrix for input sequences
        x *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]   # add positional encoding to embeddings
        return x

'''
Decoder Block Layer. 
Parameters: embed size (dimension of the embedding layer, int), num_heads (number of attention heads, int),
and hidden_size (hidden dimension of the feed forward dense layer).
Takes in the input sequences. Uses masked multiheaded self-attention and a feed forward layer, along with 
residual connections, dropout, and normalization. 
'''
class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_size, num_heads, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads, key_dim=embed_size, use_bias=True, dropout=0.3)
        self.feed_forward = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='leaky_relu'),
            tf.keras.layers.Dense(embed_size),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, inputs, training=False):
        # Self‑attention
        attention_output = self.attention(inputs, inputs, inputs, use_causal_mask=True, training=training)
            # causal mask to prevent attention on future tokens, used in a decoder Transformer
        # Residual, dropout, and normalization
        out1 = self.norm1(inputs + self.dropout(attention_output, training=training))
        # Feed‑forward
        feed_forward_out = self.feed_forward(out1, training=training)
        # Residual, dropout, and normalization
        out2 = self.norm2(out1 + self.dropout(feed_forward_out, training=training))
        return out2

'''
Transformer Decoder Model. 
Parameters: vocab_size (number of possible tokens, int), seq_len (length of the input sequences, int),
num_layers (number of Decoder Blocks, int), embed_size (dimension of the embedding layer, int), 
num_heads (number of attention heads, int), and hidden_size (hidden dimension of the feed forward dense layer).
Takes in the input sequences, creates the sequence embeddings with positional encoding, passes them through the 
decoder blocks, normalizes the output, passes it through a classifer (dense) layer.
Returns logits (over the vocab_size i.e. for all possible tokens)

'''
class DecoderModel(tf.keras.Model):
    def __init__(self, vocab_size, seq_len, num_layers=4, embed_size=256, num_heads=8, hidden_size=512, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.positional_encoding = PositionalEncoding(vocab_size, embed_size, seq_len)
        self.decoder_blocks = [DecoderBlock(embed_size, num_heads, hidden_size) for i in range(num_layers)]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.classifer = tf.keras.layers.Dense(vocab_size, use_bias=False)

    def call(self, inputs, training=False):
        # inputs: (batch, seq_len)
        seq_embeddings = self.positional_encoding(inputs)
        for block in self.decoder_blocks:
            out = block(seq_embeddings, training=training)
        out = self.norm(out)
        logits = self.classifer(out)
        return logits