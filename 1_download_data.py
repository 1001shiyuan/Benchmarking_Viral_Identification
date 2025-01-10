import subprocess
import os

def run_genome_updater(taxon, output_dir, genomes_per_genus=None):
    """
    Run genome_updater.sh to download genomes from RefSeq.
    
    Args:
        taxon: Taxonomy group to download (e.g., 'viral', 'bacteria')
        output_dir: Directory to save downloaded genomes
        genomes_per_genus: If specified, number of genomes to download per genus
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Base command
    command = [
        "./genome_updater.sh",
        "-d", "refseq",
        "-g", taxon,
        "-f", "genomic.fna.gz",
        "-o", output_dir,
        "-t", "120"
    ]
    
    # Add genus sampling if specified
    if genomes_per_genus is not None:
        command.extend(["-A", f"genus:{genomes_per_genus}"])
    
    # Run the command
    subprocess.run(command, check=True)

def download_genomes():
    """Download viral (positive) and bacterial (negative) genomes."""
    # Download all viral genomes (positive samples)
    print("Downloading positive samples: viral genomes from RefSeq")
    viral_output_dir = "./positive_samples/viral"
    run_genome_updater("viral", viral_output_dir)

    # Download bacterial genomes (negative samples), 2 per genus
    print("Downloading negative samples: bacterial genomes from RefSeq (2 per genus)")
    bacterial_output_dir = "./negative_samples/bacterial"
    run_genome_updater("bacteria", bacterial_output_dir, genomes_per_genus=2)

if __name__ == "__main__":
    download_genomes()