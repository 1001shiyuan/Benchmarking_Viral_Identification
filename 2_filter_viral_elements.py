import os
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import glob
import gzip
import shutil
import argparse

def create_viral_db(viral_folder, output_path):
    """Create BLAST database from viral sequences."""
    print("\nCreating BLAST database from viral sequences...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Concatenate all viral genomes
    print("1. Concatenating viral genomes...")
    combined_gz = os.path.join(os.path.dirname(output_path), "viral_genomes_combined.fna.gz")
    with gzip.open(combined_gz, 'wb') as outfile:
        for genome_file in glob.glob(os.path.join(viral_folder, "*.fna.gz")):
            with open(genome_file, 'rb') as infile:
                shutil.copyfileobj(infile, outfile)
    
    # Decompress combined file
    print("2. Decompressing combined file...")
    combined_fna = os.path.join(os.path.dirname(output_path), "viral_genomes_combined.fna")
    with gzip.open(combined_gz, 'rt') as f_in:
        with open(combined_fna, 'w') as f_out:
            for line in f_in:
                f_out.write(line)
    
    os.remove(combined_gz)
    
    # Create BLAST database
    print("3. Creating BLAST database...")
    makeblastdb_cmd = [
        "makeblastdb",
        "-in", combined_fna,
        "-dbtype", "nucl",
        "-out", output_path
    ]
    subprocess.run(makeblastdb_cmd, check=True)
    
    os.remove(combined_fna)
    print("BLAST database creation complete!")

def filter_viral_elements(bacterial_folder, viral_db, output_folder, evalue=1e-10):
    """Remove viral elements from bacterial sequences using BLAST."""
    os.makedirs(output_folder, exist_ok=True)
    
    for genome_file in glob.glob(os.path.join(bacterial_folder, "*.fna.gz")):
        base_name = os.path.basename(genome_file)
        output_file = os.path.join(output_folder, f"filtered_{base_name}")
        blast_output = os.path.join(output_folder, f"{base_name}.blast")
        
        # Decompress for BLAST
        temp_input = os.path.join(output_folder, "temp_input.fna")
        print(f"Processing {base_name}...")
        with gzip.open(genome_file, 'rt') as f_in, open(temp_input, 'w') as f_out:
            for record in SeqIO.parse(f_in, "fasta"):
                SeqIO.write(record, f_out, "fasta")
        
        # Run BLASTN
        blastn_cmd = [
            "blastn",
            "-query", temp_input,
            "-db", viral_db,
            "-out", blast_output,
            "-outfmt", "6 qseqid qstart qend",
            "-evalue", str(evalue),
            "-num_threads", "80"
        ]
        
        subprocess.run(blastn_cmd, check=True)
        os.remove(temp_input)
        
        # Process BLAST results
        viral_regions = {}
        if os.path.exists(blast_output):
            with open(blast_output) as f:
                for line in f:
                    seqid, start, end = line.strip().split()
                    if seqid not in viral_regions:
                        viral_regions[seqid] = []
                    viral_regions[seqid].append((int(start), int(end)))
        
        # Merge overlapping regions
        for seqid in viral_regions:
            viral_regions[seqid].sort()
            merged = []
            for region in viral_regions[seqid]:
                if not merged or merged[-1][1] < region[0]:
                    merged.append(region)
                else:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], region[1]))
            viral_regions[seqid] = merged
        
        # Filter sequences
        filtered_records = []
        with gzip.open(genome_file, "rt") as f:
            for record in SeqIO.parse(f, "fasta"):
                if record.id not in viral_regions:
                    filtered_records.append(record)
                else:
                    seq_str = str(record.seq)
                    current_pos = 0
                    for start, end in viral_regions[record.id]:
                        if start > current_pos:
                            new_seq = seq_str[current_pos:start]
                            if len(new_seq) >= 300:  # Only keep sequences >= 300bp
                                new_record = SeqRecord(
                                    Seq(new_seq),
                                    id=f"{record.id}_{current_pos}_{start}",
                                    description=f"Non-viral region from {record.id}"
                                )
                                filtered_records.append(new_record)
                        current_pos = end
                    
                    if current_pos < len(seq_str):
                        new_seq = seq_str[current_pos:]
                        if len(new_seq) >= 300:
                            new_record = SeqRecord(
                                Seq(new_seq),
                                id=f"{record.id}_{current_pos}_{len(seq_str)}",
                                description=f"Non-viral region from {record.id}"
                            )
                            filtered_records.append(new_record)
        
        # Write filtered sequences
        with gzip.open(output_file, 'wt') as f:
            SeqIO.write(filtered_records, f, "fasta")
        print(f"Wrote {len(filtered_records)} filtered sequences to {output_file}")
        
        os.remove(blast_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter viral elements from bacterial genomes')
    parser.add_argument('--bacterial_folder', type=str, required=True, 
                      help='Folder containing bacterial genomes')
    parser.add_argument('--viral_folder', type=str, required=True,
                      help='Folder containing viral genomes')
    parser.add_argument('--output_folder', type=str, required=True,
                      help='Output folder for filtered sequences')
    args = parser.parse_args()
    
    # Create viral database
    viral_db = os.path.join(args.viral_folder, "viral_db")
    create_viral_db(args.viral_folder, viral_db)
    
    # Filter viral elements
    filter_viral_elements(args.bacterial_folder, viral_db, args.output_folder)