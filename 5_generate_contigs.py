import os
import random
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import glob
import gzip

def cut_sequence_into_contigs(sequence, min_length=300, max_length=2000):
    """Cut a sequence into random-length contigs between 300-2000bp."""
    contigs = []
    seq_length = len(sequence)
    current_pos = 0
    
    while current_pos < seq_length:
        remaining_length = seq_length - current_pos
        max_possible_length = min(max_length, remaining_length)
        
        if max_possible_length < min_length:
            break
            
        contig_length = random.randint(min_length, max_possible_length)
        contig = sequence[current_pos:current_pos + contig_length]
        contigs.append(contig)
        current_pos += contig_length
    
    return contigs

def process_file(input_file, output_file):
    """Process a single file, cutting sequences into contigs."""
    contig_counter = 0
    with gzip.open(input_file, 'rt') as f_in, gzip.open(output_file, 'wt') as f_out:
        for record in SeqIO.parse(f_in, "fasta"):
            contigs = cut_sequence_into_contigs(str(record.seq))
            
            for i, contig in enumerate(contigs):
                new_record = SeqRecord(
                    Seq(contig),
                    id=f"{record.id}_contig_{i+1}",
                    description=f"Contig {i+1} from {record.description}"
                )
                SeqIO.write(new_record, f_out, "fasta")
                contig_counter += 1
    
    return contig_counter

def process_directory(input_dir, output_dir):
    """Process all files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    total_contigs = 0
    
    for input_file in glob.glob(os.path.join(input_dir, "*.fna.gz")):
        base_name = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f"contigs_{base_name}")
        
        print(f"Processing {base_name}...")
        num_contigs = process_file(input_file, output_file)
        total_contigs += num_contigs
        print(f"Generated {num_contigs} contigs from {base_name}")
    
    return total_contigs

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define input and output directories
    directories = [
        # Training data from data_split
        ("./data_split/bacterial/train", "./contigs/bacterial/train"),
        ("./data_split/viral/train", "./contigs/viral/train"),
        # Test data from test_filtered
        ("./test_filtered/bacterial", "./contigs/bacterial/test"),
        ("./test_filtered/viral", "./contigs/viral/test")
    ]
    
    # Process each directory
    for input_dir, output_dir in directories:
        print(f"\nProcessing directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        total_contigs = process_directory(input_dir, output_dir)
        print(f"Total contigs generated: {total_contigs}")