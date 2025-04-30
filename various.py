import numpy as np

def estimate_max_len(fasta_path, k):
    lengths = []
    with open(fasta_path) as fh:
        seq = ""
        for line in fh:
            if line.startswith(">"):
                if seq:
                    lengths.append(len(seq) // k)
                seq = ""
            else:
                seq += line.strip()
        if seq:
            lengths.append(len(seq) // k)
    return int(np.percentile(lengths, 100))

# example:
max_seq_len = estimate_max_len("../genome_data/filtered_cds.fasta", k=2)
print("100th %-tile k-mer length =", max_seq_len)