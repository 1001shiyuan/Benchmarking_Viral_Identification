import os
import subprocess
import glob
import gzip
import shutil
from Bio import SeqIO
from collections import defaultdict

def create_blast_db_from_training(train_folder, output_path):
    """Create BLAST database from training sequences."""
    print(f"\nCreating BLAST database from training sequences in {train_folder}...")
    
    # Create temporary uncompressed file containing all training sequences
    temp_fasta = output_path + "_temp.fasta"
    with open(temp_fasta, 'w') as outfile:
        for gz_file in glob.glob(os.path.join(train_folder, "*.fna.gz")):
            with gzip.open(gz_file, 'rt') as f:
                for record in SeqIO.parse(f, "fasta"):
                    SeqIO.write(record, outfile, "fasta")
    
    # Create BLAST database
    makeblastdb_cmd = [
        "makeblastdb",
        "-in", temp_fasta,
        "-dbtype", "nucl",
        "-out", output_path
    ]
    subprocess.run(makeblastdb_cmd, check=True)
    
    os.remove(temp_fasta)
    print("BLAST database creation complete!")

def calculate_coverage(blast_results):
    """
    Calculate overall coverage from BLAST alignments.
    Returns coverage percentage and merged alignment regions.
    """
    # Sort alignments by start position
    blast_results.sort(key=lambda x: x[0])
    
    # Merge overlapping regions
    merged = []
    total_length = None
    
    for start, end, length in blast_results:
        if not merged:
            merged.append([start, end])
            total_length = length
        else:
            if start <= merged[-1][1]:
                # Overlapping regions - extend the end if needed
                merged[-1][1] = max(merged[-1][1], end)
            else:
                # Non-overlapping - add new region
                merged.append([start, end])
    
    # Calculate total coverage
    covered_length = sum(end - start for start, end in merged)
    coverage_percent = (covered_length / total_length) * 100 if total_length else 0
    
    return coverage_percent, merged

def filter_test_sequences(test_folder, train_db, output_folder, identity_threshold=70, coverage_threshold=50):
    """Filter test sequences based on similarity to training sequences."""
    os.makedirs(output_folder, exist_ok=True)
    
    for test_file in glob.glob(os.path.join(test_folder, "*.fna.gz")):
        base_name = os.path.basename(test_file)
        print(f"\nProcessing {base_name}")
        
        # Create temporary uncompressed file for BLAST
        temp_input = os.path.join(output_folder, "temp_input.fna")
        with gzip.open(test_file, 'rt') as f_in, open(temp_input, 'w') as f_out:
            for record in SeqIO.parse(f_in, "fasta"):
                SeqIO.write(record, f_out, "fasta")
        
        # Run BLASTN
        blast_output = os.path.join(output_folder, f"{base_name}.blast")
        blastn_cmd = [
            "blastn",
            "-query", temp_input,
            "-db", train_db,
            "-out", blast_output,
            "-outfmt", "6 qseqid qstart qend qlen pident",
            "-evalue", "1e-10",
            "-num_threads", "80"
        ]
        subprocess.run(blastn_cmd, check=True)
        
        # Process BLAST results
        sequences_to_keep = set()
        with open(blast_output) as f:
            query_alignments = defaultdict(list)
            for line in f:
                qseqid, qstart, qend, qlen, pident = line.strip().split()
                if float(pident) >= identity_threshold:
                    query_alignments[qseqid].append(
                        (int(qstart), int(qend), int(qlen))
                    )
            
            # Calculate coverage for each sequence
            for qseqid, alignments in query_alignments.items():
                coverage, _ = calculate_coverage(alignments)
                if coverage < coverage_threshold:
                    sequences_to_keep.add(qseqid)
        
        # Write filtered sequences
        output_file = os.path.join(output_folder, base_name)
        with gzip.open(test_file, 'rt') as f_in, gzip.open(output_file, 'wt') as f_out:
            for record in SeqIO.parse(f_in, "fasta"):
                if (record.id not in query_alignments or 
                    record.id in sequences_to_keep):
                    SeqIO.write(record, f_out, "fasta")
        
        # Clean up
        os.remove(temp_input)
        os.remove(blast_output)
        print(f"Processed {base_name}")

if __name__ == "__main__":
    # Directory structure
    data_split_dir = "./data_split"
    output_base_dir = "./test_filtered"
    
    # Process both viral and bacterial data
    for data_type in ['viral', 'bacterial']:
        print(f"\nProcessing {data_type} data...")
        
        # Set up paths
        train_folder = os.path.join(data_split_dir, data_type, 'train')
        test_folder = os.path.join(data_split_dir, data_type, 'test')
        output_folder = os.path.join(output_base_dir, data_type)
        blast_db = os.path.join(output_base_dir, f"{data_type}_train_db")
        
        # Create output directories
        os.makedirs(output_folder, exist_ok=True)
        
        # Create BLAST database from training sequences
        create_blast_db_from_training(train_folder, blast_db)
        
        # Filter test sequences
        filter_test_sequences(test_folder, blast_db, output_folder)