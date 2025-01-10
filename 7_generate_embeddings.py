import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
import numpy as np
from pathlib import Path
import gzip

def load_model_and_tokenizer():
    """Load DNABERT-2 model and tokenizer."""
    print("Loading DNABERT-2-117M model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    base_model = model.bert
    base_model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = base_model.to(device)
    
    return tokenizer, base_model, device

def generate_embedding(sequence, tokenizer, model, device):
    """Generate embedding for a single sequence using [CLS] token."""
    with torch.no_grad():
        inputs = tokenizer(sequence, return_tensors='pt')["input_ids"].to(device)
        hidden_states = model(inputs)[0]
        embedding = hidden_states[0, 0, :]  # Get [CLS] token embedding
    return embedding.cpu().numpy()

def read_gzipped_fasta(file_path):
    """Read sequences from a gzipped FASTA file and return sequences with their lengths."""
    sequences = []
    lengths = []
    current_seq = []
    
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    seq = ''.join(current_seq)
                    sequences.append(seq)
                    lengths.append(len(seq))
                current_seq = []
            else:
                current_seq.append(line)
    if current_seq:
        seq = ''.join(current_seq)
        sequences.append(seq)
        lengths.append(len(seq))
    
    return sequences, lengths

def get_length_group(length):
    """Determine which length group a sequence belongs to."""
    if length < 500:
        return 'less_than_500bp'
    elif length < 1000:
        return '500_to_1000bp'
    else:
        return '1000_to_2000bp'

def process_contigs_directory(input_dir, output_dir, is_test=False):
    """Process all contig files in directory and generate embeddings."""
    tokenizer, model, device = load_model_and_tokenizer()
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create length group directories for test data
    if is_test:
        length_groups = ['less_than_500bp', '500_to_1000bp', '1000_to_2000bp']
        group_dirs = {group: os.path.join(output_dir, group) for group in length_groups}
        for dir_path in group_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    # Process each gzipped FASTA file
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.fna.gz'):
            print(f"\nProcessing {file_name}...")
            input_path = os.path.join(input_dir, file_name)
            sequences, lengths = read_gzipped_fasta(input_path)
            print(f"Found {len(sequences)} sequences")
            
            if is_test:
                # Group test sequences by length
                group_sequences = {group: [] for group in length_groups}
                for seq, length in zip(sequences, lengths):
                    group = get_length_group(length)
                    group_sequences[group].append(seq)
                
                # Generate embeddings for each length group
                for group in length_groups:
                    if group_sequences[group]:
                        print(f"Processing {group} sequences...")
                        embeddings = []
                        for i, seq in enumerate(group_sequences[group], 1):
                            try:
                                embedding = generate_embedding(seq, tokenizer, model, device)
                                embeddings.append(embedding)
                                if i % 100 == 0:
                                    print(f"Processed {i} sequences in {group}")
                            except Exception as e:
                                print(f"Error processing sequence {i} in {group}: {str(e)}")
                        
                        if embeddings:
                            embeddings = np.array(embeddings)
                            output_path = os.path.join(group_dirs[group], 
                                                     f"{os.path.splitext(file_name)[0]}_embeddings.npy")
                            np.save(output_path, embeddings)
                            print(f"Saved {len(embeddings)} embeddings for {group}")
            
            else:
                # Process training data normally
                embeddings = []
                for i, seq in enumerate(sequences, 1):
                    try:
                        embedding = generate_embedding(seq, tokenizer, model, device)
                        embeddings.append(embedding)
                        if i % 100 == 0:
                            print(f"Processed {i} sequences")
                    except Exception as e:
                        print(f"Error processing sequence {i}: {str(e)}")
                
                if embeddings:
                    embeddings = np.array(embeddings)
                    output_path = os.path.join(output_dir, 
                                             f"{os.path.splitext(file_name)[0]}_embeddings.npy")
                    np.save(output_path, embeddings)
                    print(f"Saved {len(embeddings)} embeddings")

if __name__ == "__main__":
    # Directory structure
    EMBEDDINGS_DIR = "./embeddings"
    TRAIN_EMBEDDINGS_DIR = os.path.join(EMBEDDINGS_DIR, "train")
    TEST_EMBEDDINGS_DIR = os.path.join(EMBEDDINGS_DIR, "test")
    
    # Create directories
    os.makedirs(TRAIN_EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(TEST_EMBEDDINGS_DIR, exist_ok=True)
    
    # Input directories
    TRAIN_CONTIGS_DIR = "./balanced_data"
    TEST_CONTIGS_DIR = "./balanced_data"
    
    # Process training contigs
    print("\nProcessing training contigs...")
    process_contigs_directory(TRAIN_CONTIGS_DIR, TRAIN_EMBEDDINGS_DIR, is_test=False)
    
    # Process test contigs with length grouping
    print("\nProcessing test contigs...")
    process_contigs_directory(TEST_CONTIGS_DIR, TEST_EMBEDDINGS_DIR, is_test=True)