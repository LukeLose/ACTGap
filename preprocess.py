def open_file(file_name):
    infile = open(f'./{file_name}', 'r')
    infile = infile.readlines() # list
    infile = infile[1:]
    for l in range(len(infile)):
        infile[l] = infile[l].replace('\n','')
    infile = ''.join(infile) # string
    print(f'{file_name} has been read')
    return infile

def section_file(infile, length):
    # randomly choose sequence within chromosome?
    sequence = infile[:length]
    print(f'Sequence of {length} bp selected from input sequence')
    return sequence

def sliding_window(sequence, window_size):
    n_batches = len(sequence)//window_size
    sequences = np.array([sequence[i*window_size:(i+1)*window_size] for i in range(n_batches)])
    print(f'Sequence has been split into {window_size} bp sequences (non-overlapping)')
    return sequences

def create_kmer_dict(kmer_length):
    kmers = list(product('ACTG', repeat=kmer_length))
    kmers = [''.join(c) for c in kmers]
    kmer_dict = {k:v for k,v in zip(np.arange(4**kmer_length),kmers)}
    with open('./kmer_dict.p', 'wb') as pickle_file:
        pickle.dump(kmer_dict, pickle_file)
    print(f'Pickled k-mer dictionary has been dumped into ./kmer_dict.p')
    return kmer_dict

def kmer_tokenization(kmer_dict, sequences, window_size, kmer_length):
    n_kmers = window_size-kmer_length+1
    kmer_sequences = np.array([[sequences[j][i:i+kmer_length] for i in range(n_kmers)] for j in range(n_batches)])
    kmer_seq = np.zeros((n_batches,n_kmers))
    for j in range(n_batches):
        for i in range(n_kmers):
            val = kmer_sequences[j,i]
            keys = [k for k,v in kmer_dict.items() if v==val]
            kmer_seq[j,i] = keys[0]
    kmer_sequences = np.insert(kmer_sequences, 0, 'BEG', axis=1)
    kmer_sequences = np.insert(kmer_sequences, -1, 'END', axis=1)
    kmer_seq = np.insert(kmer_seq, 0, -2, axis=1)
    kmer_seq = np.insert(kmer_seq, -1, -1, axis=1)
    print(f'Sequences have been tokenized into {kmer_length}-mers')
    return kmer_sequences, kmer_seq

def shuffle_data(n_batches, kmer_sequences):
    indices = tf.random.shuffle(np.arange(n_batches))
    shuffled_sequences = tf.gather(kmer_sequences, indices) 
    print('Data batches have been shuffled')
    return shuffled_sequences

def gap_masking(n_batches, n_kmers, n_gaps):
    mask = np.zeros((n_batches,n_kmers+2))
    for n in range(n_batches): 
        i = np.random.choice(n_kmers-n_gaps+1)
        mask[n,i:i+n_gaps] = [1]*n_gaps
    print(f'Gap mask with {n_gaps} gaps has been created')
    return mask

def main():
    infile = open_file('T2T_chr21.fasta')
    length = 20000
    sequence = section_file(infile, length)
    window_size = 512
    n_batches = len(sequence)//window_size
    sequences = sliding_window(sequence, window_size)
    kmer_length = 6
    kmer_dict = create_kmer_dict(kmer_length)
    kmer_sequences, kmer_seqs = kmer_tokenization(kmer_dict, sequences, window_size, kmer_length)
    n_kmers = window_size-kmer_length+1
    shuffled_sequences = shuffle_data(n_batches, kmer_seqs)
    mask = gap_masking(n_batches, n_kmers, n_gaps=10)

main()
