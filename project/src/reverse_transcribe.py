import os

# The exact amino acid and nucleotide sequence for HIV-1 HXB2 Protease
HXB2_PR_AA = "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
HXB2_PR_NT = "CCTCAGATCACTCTTTGGCAACGACCCCTCGTCACAATAAAGATAGGGGGGCAACTAAAGGAAGCTCTATTAGATACAGGAGCAGATGATACAGTATTAGAAGAAATGAGTTTGCCAGGAAGATGGAAACCAAAAATGATAGGGGGAATTGGAGGTTTTATCAAAGTAAGACAGTATGATCAGATACTCATAGAAATCTGTGGACATAAAGCTATAGGTACAGTATTAGTAGGACCTACACCTGTCAACATAATTGGAAGAAATCTGTTGACTCAGATTGGTTGCACTTTAAATTTT"

# Fallback codons derived from common HIV-1 usage for when a mutation is present
FALLBACK_CODON_MAP = {
    'A': 'GCA', 'C': 'TGC', 'D': 'GAT', 'E': 'GAA', 'F': 'TTT',
    'G': 'GGA', 'H': 'CAT', 'I': 'ATA', 'K': 'AAA', 'L': 'TTA',
    'M': 'ATG', 'N': 'AAC', 'P': 'CCT', 'Q': 'CAG', 'R': 'AGA',
    'S': 'TCA', 'T': 'ACA', 'V': 'GTA', 'W': 'TGG', 'Y': 'TAT',
    '*': 'TAA', 'X': 'NNN'
}

def back_translate_smart(input_fasta, output_fasta):
    """
    Translates protein to DNA using the HXB2 genome as a template to 
    maintain high nucleotide sequence identity for downstream alignment.
    """
    with open(input_fasta, 'r') as fin, open(output_fasta, 'w') as fout:
        header = ""
        seq_lines = []

        def write_current_record():
            if header:
                aa_seq = "".join(seq_lines).upper()
                dna_seq = []
                
                # Align to the 99-AA HXB2 reference positionally
                for i, aa in enumerate(aa_seq):
                    # If position exists in HXB2 and the amino acid matches
                    if i < len(HXB2_PR_AA) and aa == HXB2_PR_AA[i]:
                        # Use the exact 3-nucleotide codon from the HIV reference
                        dna_seq.append(HXB2_PR_NT[i*3 : i*3+3])
                    else:
                        # Mutation found (or extra sequence), use standard fallback codon
                        dna_seq.append(FALLBACK_CODON_MAP.get(aa, 'NNN'))
                        pass
                    pass
                
                fout.write(f"{header}\n{''.join(dna_seq)}\n")
                pass
            return
        
        for line in fin:
            line = line.strip()
            if not line: continue
                        
            if line.startswith(">"):
                write_current_record()
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
                pass
            pass
        
        write_current_record()
        
    print(f"Success! Smart-translated '{input_fasta}' to '{output_fasta}'.")

if __name__ == "__main__":
    input_file = "data/dataset_sequences.fa"
    output_file = "data/dataset_sequences_nt.fa"
    back_translate_smart(input_file, output_file)
