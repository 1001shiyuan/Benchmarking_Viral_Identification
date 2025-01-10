import os
import random
from Bio import SeqIO
import glob
import gzip

def count_total_bases(directory):
    """Count total number of bases in all .fna.gz files in a directory."""
    total_bases = 0
    total_sequences = 0
    
    for file_path in glob.glob(os.path.join(directory, "*.fna.gz")):
        with gzip.open(file_path, 'rt') as f:
            for record in SeqIO.parse(f, "fasta"):
                total_bases += len(record.seq)
                total_sequences += 1
    
    return total_bases, total_sequences

def collect_all_sequences(directory):
    """Collect all sequences from a directory into a list."""
    sequences = []
    for file_path in glob.glob(os.path.join(directory, "*.fna.gz")):
        with gzip.open(file_path, 'rt') as f:
            for record in SeqIO.parse(f, "fasta"):
                sequences.append(record)
    return sequences

def downsample_and_save(sequences, target_bases, output_dir, dataset_type):
    """Randomly sample sequences until reaching target base count."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Shuffle sequences
    random.shuffle(sequences)
    
    # Sample sequences until reaching target
    sampled_sequences = []
    current_bases = 0
    
    for seq in sequences:
        if current_bases >= target_bases:
            break
        sampled_sequences.append(seq)
        current_bases += len(seq.seq)
    
    # Save sampled sequences
    output_file = os.path.join(output_dir, f"bacterial_{dataset_type}.fna.gz")
    with gzip.open(output_file, 'wt') as f:
        SeqIO.write(sampled_sequences, f, "fasta")
    
    return len(sampled_sequences), current_bases

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Input directories
    contigs_dir = "./contigs"
    viral_train_dir = os.path.join(contigs_dir, "viral/train")
    viral_test_dir = os.path.join(contigs_dir, "viral/test")
    bacterial_train_dir = os.path.join(contigs_dir, "bacterial/train")
    bacterial_test_dir = os.path.join(contigs_dir, "bacterial/test")
    
    # Output directory
    output_dir = "./balanced_data"
    
    # Process training data
    print("\nProcessing training data...")
    viral_train_bases, viral_train_seqs = count_total_bases(viral_train_dir)
    print(f"Viral training dataset: {viral_train_bases:,} bases in {viral_train_seqs:,} sequences")
    
    print("\nCollecting bacterial training sequences...")
    bacteria_train_sequences = collect_all_sequences(bacterial_train_dir)
    print(f"Total bacterial training sequences collected: {len(bacteria_train_sequences):,}")
    
    print("\nDownsampling bacterial training sequences...")
    n_sampled_train, sampled_train_bases = downsample_and_save(
        bacteria_train_sequences, 
        viral_train_bases,
        output_dir,
        "train"
    )
    
    # Copy viral training data to output directory
    print("\nCopying viral training data...")
    viral_train_out = os.path.join(output_dir, "viral_train.fna.gz")
    with gzip.open(viral_train_out, 'wt') as f_out:
        for file_path in glob.glob(os.path.join(viral_train_dir, "*.fna.gz")):
            with gzip.open(file_path, 'rt') as f_in:
                for record in SeqIO.parse(f_in, "fasta"):
                    SeqIO.write(record, f_out, "fasta")
    
    # Process test data
    print("\nProcessing test data...")
    viral_test_bases, viral_test_seqs = count_total_bases(viral_test_dir)
    print(f"Viral test dataset: {viral_test_bases:,} bases in {viral_test_seqs:,} sequences")
    
    print("\nCollecting bacterial test sequences...")
    bacteria_test_sequences = collect_all_sequences(bacterial_test_dir)
    print(f"Total bacterial test sequences collected: {len(bacteria_test_sequences):,}")
    
    print("\nDownsampling bacterial test sequences...")
    n_sampled_test, sampled_test_bases = downsample_and_save(
        bacteria_test_sequences,
        viral_test_bases,
        output_dir,
        "test"
    )
    
    # Copy viral test data to output directory
    print("\nCopying viral test data...")
    viral_test_out = os.path.join(output_dir, "viral_test.fna.gz")
    with gzip.open(viral_test_out, 'wt') as f_out:
        for file_path in glob.glob(os.path.join(viral_test_dir, "*.fna.gz")):
            with gzip.open(file_path, 'rt') as f_in:
                for record in SeqIO.parse(f_in, "fasta"):
                    SeqIO.write(record, f_out, "fasta")
    
    # Print final statistics
    print("\nFinal Statistics:")
    print("Training Data:")
    print(f"Viral:     {viral_train_bases:,} bases in {viral_train_seqs:,} sequences")
    print(f"Bacterial: {sampled_train_bases:,} bases in {n_sampled_train:,} sequences")
    print("\nTest Data:")
    print(f"Viral:     {viral_test_bases:,} bases in {viral_test_seqs:,} sequences")
    print(f"Bacterial: {sampled_test_bases:,} bases in {n_sampled_test:,} sequences")