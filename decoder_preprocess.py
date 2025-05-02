from typing import Tuple, List
import numpy as np
import tensorflow as tf
from itertools import product

#set the known padding ID
pad_id = 0

def build_kmer_dictionary(k: int) -> Tuple[dict, dict]:
    '''
    Takes a kmer size and generates all possible permutations of
    the base alphabet
    '''
    base_alphabet = ["A", "C", "G", "T", "N"]
    #create every permutation of our alphabet for the given length
    kmers = ["".join(base) for base in product(base_alphabet, repeat=k)]
    token_to_bases = {}
    for i, kmer in enumerate(kmers):
        token_to_bases[kmer] = i + 1
    bases_to_token = {i: kmer for kmer, i in token_to_bases.items()}
    #returns both the forward and backward kv dictionaries for token and ID
    return token_to_bases, bases_to_token

def fasta_to_inputs(fasta_path: str,
                      token_to_bases: dict,
                      gap_length: int,
                      k_length: int) -> List[np.ndarray]:
    '''
    open the fasta path, find sequence lines and create sequences among multiple
    lines of the fasta
    '''
    seqs = []
    with open(fasta_path) as fh:
        sequence = ""
        for line in fh:
            #do not take into account ENST transcript identifiers
            if line.startswith(">"):
                if sequence:
                    seqs.append(sequence.upper())
                    sequence = ""
            else:
                sequence += line.strip()
        if sequence:
            seqs.append(sequence.upper())

    model_inputs = []
    #now that we have all sequences, make them kmers using dictionary translation
    for sequence in seqs:
        tokens = []
        lenseq = len(sequence)
        #goes through whole sequence and sets sequnces of k-length
        #to be turned into kmers
        for i in range(0, lenseq - k_length + 1, k_length):
            kmer = sequence[i : i + k_length]
            tokens.append(kmer)

        ids = []
        for token in tokens:
            # get the ID if it exists, otherwise PAD_ID
            token_id = token_to_bases.get(token, pad_id)
            ids.append(token_id)

        #take all ids from the start to the gap
        context_ids = ids[:-gap_length]
        #take all from the gap on
        gap_ids = ids[-gap_length:]
        model_inputs.append((context_ids, gap_ids))
    return model_inputs


def make_dataset(model_inputs: List[Tuple[List[int], List[int]]], gap_id: int, seq_len: int,
                 batch_size: int = 32, shuffle: bool = True): 
    '''
    speed processing step that turns our group of tuples into an dataset array
    '''
    #deafult to shuffling
    #Seperate back into indiv list
    input_arr = []
    label_arr = []
    for context, gap in model_inputs:
        #change the input to a max size array and add what the 
        #model will know
        context_arr = context + [gap_id] + gap[:-1]
        #for label do the opposite, add context then put the gap
        #sequence information in
        label = [-100] * len(context) + gap
        #take the largest sequence and set the padding to
        #the largest sequence length
        pad_len = seq_len - len(context_arr)
        context_arr = context_arr + [pad_id]*pad_len
        label = label + [-100]*pad_len
        input_arr.append(context_arr)
        label_arr.append(label)

    dataset = tf.data.Dataset.from_tensor_slices((input_arr, label_arr))
    if shuffle:
        #use built in dataset shuffle methods
        dataset = dataset.shuffle(len(input_arr))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def max_length_helper(fasta_path, k_length):
    '''
    helper function that takes in the fasta and parses it to determine 
    the largest sequence (for our training data it should be 1024)
    '''
    max_nt = 0
    with open(fasta_path) as fh:
        sequence = []
        #same sequence finder code
        for line in fh:
            if line.startswith(">"):
                if sequence:
                    #see if this one line is longest then the max_nt so far
                    max_nt = max(max_nt, len("".join(sequence)))
                sequence = []
            else:
                sequence.append(line.strip())
        if sequence:
            #another comparison check for the last line
            max_nt = max(max_nt, len("".join(sequence)))
    return (max_nt // k_length) + 1