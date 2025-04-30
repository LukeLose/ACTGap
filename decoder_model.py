import tensorflow as tf
import numpy as np

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


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_size, num_heads, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads, key_dim=embed_size, use_bias=True, dropout=0.1)
        self.feed_forward = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='leaky_relu'),
            tf.keras.layers.Dense(embed_size),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.dropout = tf.keras.layers.Dropout(0.1)

    def call(self, inputs, training=False):
        # Self‑attention with residual
        attention_output = self.attention(inputs, inputs, inputs, use_causal_mask=True, training=training)
            # causal mask to prevent attention on future tokens, used in a decoder Transformer
        out1 = self.norm1(inputs + self.drop(attention_output, training=training))
        # Feed‑forward with residual
        feed_forward_out = self.feed_forward(out1, training=training)
        out2 = self.norm2(out1 + self.drop(feed_forward_out, training=training))
        return out2


class DecoderModel(tf.keras.Model):
    def __init__(self, vocab_size, seq_len, num_layers=4, embed_size=256, num_heads=8, hidden_size=512, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.positional_encoding = PositionalEncoding(vocab_size, embed_size, seq_len)
        self.decoder_blocks = [DecoderBlock(embed_size, num_heads, hidden_size) for i in range(num_layers)]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.classifer = tf.keras.layers.Dense(vocab_size, activation='softmax', use_bias=False)

    def call(self, inputs, training=False):
        # inputs: (batch, seq_len)
        seq_embeddings = self.positional_encoding(inputs)
        for block in self.decoder_blocks:
            out = block(seq_embeddings, training=training)
        out = self.norm(out)
        logits = self.classifer(out)  # (batch, seq_len, vocab)
        return logits