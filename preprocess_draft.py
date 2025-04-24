import pickle
import random
import numpy as np
import tensorflow as tf
from urllib.request import urlretrieve
from itertools import product

# https://www.ncbi.nlm.nih.gov/nuccore/NC_000001.11?report=fasta

# Open File
infile = open('./GRCh38_mito.fasta', 'r')
infile = infile.readlines()
infile = infile[1:]
print(type(infile))
for l in range(len(infile)):
    infile[l] = infile[l].replace('\n','')
    # line = line.replace('N','')
# print(infile[0:10])
mito = ''.join(infile)
print(type(mito))
print(len(mito))

'''
# Sliding Window
print(infile.find('T'))
x = infile[:10000]
start = x.rfind('N') + 1
print(start)
window_size = 512
n = 20
length = window_size * n
end = start + length
print(end)
chr1 = infile[start:end+1]
print(len(chr1))
print(chr1[0:10])
'''

# Sliding Window 
window_size = 512
n_batches = len(mito)//window_size
sequences = np.array([mito[i*window_size:(i+1)*window_size] for i in range(n_batches)])
print(sequences.shape)

# K-mer Tokenization
kmer_length = 6
kmers = list(product('ACTG', repeat=kmer_length))
kmers = [''.join(c) for c in kmers]
kmer_dict = {k:v for k,v in zip(np.arange(4**kmer_length),kmers)}
with open('./kmer_dict.p', 'wb') as pickle_file:
    pickle.dump(kmer_dict, pickle_file)
print(f'Data has been dumped into ./kmer_dict.p!')
'''
with open('./kmer_dict.p', 'rb') as data_file:
        kmer_dict = pickle.load(data_file)
'''
n_kmers = window_size-kmer_length+1
kmer_sequences = np.array([[sequences[j][i:i+kmer_length] for i in range(n_kmers)] for j in range(n_batches)])
print(kmer_sequences.shape)
print(kmer_sequences[0,:10])
kmer_seq = np.zeros((n_batches,n_kmers))
for j in range(n_batches):
    for i in range(n_kmers):
        val = kmer_sequences[j,i]
        keys = [k for k,v in kmer_dict.items() if v==val]
        if len(keys)==0:
            print(j,i)
            print(val)
        kmer_seq[j,i] = keys[0]
print(kmer_seq.shape)
print(kmer_seq[0,:10])
kmer_sequences = np.insert(kmer_sequences, 0, 'BEG', axis=1)
kmer_sequences = np.insert(kmer_sequences, -1, 'END', axis=1)
print(kmer_sequences.shape)

# Shuffle data
indices = tf.random.shuffle(np.arange(n_batches))
shuffled_sequences = tf.gather(kmer_sequences, indices)
print(shuffled_sequences.shape)
'''
# Masking (Gaps)
n_gaps = 3
gap_sequences = np.copy(kmer_sequences)
for n in range(n_batches): 
    i = np.random.choice(n_kmers-n_gaps+1)
    gap_sequences[n][i:i+n_gaps] = ['MASK'*n_gaps]
'''
