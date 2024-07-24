#!/opt/homebrew/bin/python3
import ssl
from Bio import SeqIO
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
import pandas as pd
from Bio.SeqUtils import ProtParam

def calculate_e_descriptors(protein_sequence):
    # Define hydrophobicity scale (Kyte-Doolittle scale as an example)
    hydrophobicity_scale = {'m': 1.9, 'l': 1.3, 'i': 1.3, 'f': 2.8, 'v': 1.0, 's': -0.8, 'p': -1.6, 't': -0.7, 'a': 0.5, 'y': 2.8, 'h': -3.2, 'q': -3.5, 'n': -3.5, 'k': -3.9, 'd': -3.5, 'e': -3.5, 'c': 2.5, 'w': -0.9, 'r': -4.5, 'g': 0.4}

    # Define helical propensity scale
    helical_propensity_scale = {'m': 1.1, 'l': 0.9, 'i': 1.1, 'f': 1.2, 'v': 1.1, 's': 0.8, 'p': 1.1, 't': 0.8, 'a': 0.8, 'y': 0.7, 'h': 0.7, 'q': 1.0, 'n': 1.0, 'k': 0.6, 'd': 0.9, 'e': 1.0, 'c': 0.7, 'w': 1.3, 'r': 0.6, 'g': 0.6}

    # Define strand propensity scale
    strand_propensity_scale = {'m': 1.2, 'l': 1.4, 'i': 1.4, 'f': 1.2, 'v': 1.3, 's': 1.1, 'p': 0.7, 't': 1.2, 'a': 0.7, 'y': 1.0, 'h': 0.9, 'q': 1.0, 'n': 1.2, 'k': 1.2, 'd': 1.3, 'e': 1.3, 'c': 0.8, 'w': 1.0, 'r': 0.7, 'g': 0.5}

    # E1: Hydrophilic nature of peptides
    e1 = sum(hydrophobicity_scale.get(aa.lower(), 0) for aa in protein_sequence) / len(protein_sequence)

    # E2: Length of peptides
    e2 = len(protein_sequence)

    # E3: Tendency for helical formation
    e3 = sum(helical_propensity_scale.get(aa.lower(), 0) for aa in protein_sequence) / len(protein_sequence)

    # E4: Abundance and distribution (example: count of positively charged amino acids)
    positively_charged_aa = ['r', 'k']
    e4 = sum(protein_sequence.lower().count(aa) for aa in positively_charged_aa) / len(protein_sequence)

    # E5: Tendency for β strand formation
    e5 = sum(strand_propensity_scale.get(aa.lower(), 0) for aa in protein_sequence) / len(protein_sequence)

    #E11
    motif_count = protein_sequence.lower().count('abc')
    e11 = motif_count / len(protein_sequence)  # Normalize by protein sequence length

    #E9 susceptibility at low pH
    #This code assumes that amino acids 'D' (aspartic acid) and 'E' (glutamic acid) are susceptible to low pH. 
    low_ph_susceptible_aa = ['d', 'e']
    low_ph_susceptibility_score = sum(protein_sequence.lower().count(aa) for aa in low_ph_susceptible_aa)
    e9 = low_ph_susceptibility_score / len(protein_sequence)  # Normalize by protein sequence length
    

    # # E10: Cross-reactivity 
    # # Define a scoring system for amino acids based on cross-reactivity propensity
    # cross_reactivity_scores = {'a': 0.2, 'r': 0.8, 'n': 0.5, 'd': 0.7, 'c': 0.3, 'q': 0.5, 'e': 0.7, 'g': 0.2, 'h': 0.6,
    #                            'i': 0.4, 'l': 0.4, 'k': 0.7, 'm': 0.5, 'f': 0.6, 'p': 0.4, 's': 0.3, 't': 0.3, 'w': 0.7,
    #                            'y': 0.6, 'v': 0.3}

    # # Calculate cross-reactivity score for the protein sequence
    # cross_reactivity_score = sum(cross_reactivity_scores.get(aa.lower(), 0) for aa in protein_sequence)
    # e10 = cross_reactivity_score / len(protein_sequence)  # Normalize by protein sequence length
    return e1, e2, e3, e4, e5, e11, e9



def calculate_additional_features(protein_sequence):
    protein_sequence = protein_sequence.replace('X', '').replace('Z', '')

    # E6: Disulfide bond formation
    num_disulfide_bonds = protein_sequence.count('C') // 2  # Counting cysteine residues
    
    # E7: Stability from pepsin (pH 1.0)
    pepsin_stability = ProtParam.ProteinAnalysis(protein_sequence).molecular_weight()  

    # E8 Thermal_stability
    prot_param = ProtParam.ProteinAnalysis(protein_sequence)
    alpha_helix_fraction = prot_param.secondary_structure_fraction()[0]
    beta_sheet_fraction = prot_param.secondary_structure_fraction()[1]

    # A simple linear combination of alpha helix and beta sheet fractions
    Thermal_stability = alpha_helix_fraction * 30 + beta_sheet_fraction * 20
    return num_disulfide_bonds, pepsin_stability, Thermal_stability


def calculate_cross_reactivity(sequence):
    # Perform a BLAST search against a database of known allergens or epitopes
    # Disable SSL certificate verification
    ssl._create_default_https_context = ssl._create_unverified_context
    result_handle = NCBIWWW.qblast("blastp", "nr", sequence)
    
    # Parse the BLAST results
    blast_records = NCBIXML.parse(result_handle)
    cross_reactivity_score = 0
    
    for blast_record in blast_records:
        for alignment in blast_record.alignments:
            # Calculate the sequence identity (percentage of identical amino acids)
            sequence_identity = alignment.length / len(sequence)
            
            # Increase cross-reactivity score if sequence identity is above a certain threshold
            if sequence_identity > 0.8:
                cross_reactivity_score += 1
    return cross_reactivity_score / len(sequence)  # Normalize by sequence length

# Read CSV file
input_csv_path = '/Users/sarvajeethuk/Downloads/train_Sequence.csv'
df = pd.read_csv(input_csv_path)

# Assuming the column name is 'Sequence'
protein_sequences = df['Sequence'].tolist()

# Calculate E Descriptors for each protein sequence
e_descriptors_list = [calculate_e_descriptors(seq.lower()) for seq in protein_sequences]

# Calculate additional features for each protein sequence
additional_features_list = [calculate_additional_features(seq) for seq in protein_sequences]
cross_reactivity = [calculate_cross_reactivity(seq) for seq in protein_sequences]

# print(additional_features_list)
# Add E Descriptors and additional features to the DataFrame
e_descriptors_df = pd.DataFrame(e_descriptors_list, columns=['E1', 'E2', 'E3', 'E4', 'E5','E11', 'E9'])
additional_features_df = pd.DataFrame(additional_features_list, columns=['E6', 'E7','E8'])
cross_reactivity_df = pd.DataFrame(additional_features_list, columns=['E10'])
# additional_features_df = pd.DataFrame(additional_features_list, columns=['E6', 'E7'])
result_df = pd.concat([df, e_descriptors_df, additional_features_df, cross_reactivity_df], axis=1)

# Save the resulting DataFrame to a new CSV file
output_csv_path = 'result_with_e_descriptors_and_additional_features_Train.csv'
result_df.to_csv(output_csv_path, index=False)