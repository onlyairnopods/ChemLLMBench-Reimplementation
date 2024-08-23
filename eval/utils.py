from typing import List, Any, Literal, Optional, Union, Tuple, Sequence

import pandas as pd
import difflib

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import DataStructs

def canonicalize_smiles(smiles: str) -> str:
    """
    convert SMILES to Canonical SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        canonical_smiles = Chem.MolToSmiles(mol)
        return canonical_smiles
    except:
        return smiles
    
def get_scaffold_fp(smiles: str) -> Any:
    """
    Returns a Morgan fingerprint for a molecule SMILES string.
    """
    try:
        mol = Chem.MolFromSmiles(x)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_fp = rdMolDescriptors.GetMorganFingerprint(scaffold, 2)
        return scaffold_fp
    except:
        return None
    
def top_n_scaffold_similar_molecules(target_smiles: str, molecule_scaffold_list: Sequence, molecule_smiles_list: Sequence, top_n: int = 5) -> List:
    """
    Sample top-n similar molecules SMILES based on molecular scaffold Tanimoto Similarity from Morgan Fingerprint with 2048-bit and radius=2.
    """
    target_mol = Chem.MolFromSmiles(target_smiles)
    target_scaffold = MurckoScaffold.GetScaffoldForMol(target_mol)
    target_fp = rdMolDescriptors.GetMorganFingerprint(target_scaffold, 2)

    similarities = []

    for idx, scaffold_fp in enumerate(molecule_scaffold_list):
        try:
            tanimoto_similarity = DataStructs.TanimotoSimilarity(target_fp, scaffold_fp)
            similarities.append((idx, tanimoto_similarity))
        except Exception as e:
            print(e)
            continue

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_n_similar_molecules = similarities[:top_n]

    return [molecule_smiles_list[i[0]] for i in top_n_similar_molecules]
    
def similarity_ratio(s1: str, s2:str) -> float:
    """
    Calculate the similarity ratio between the two strings
    """
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def top_n_similar_strings(query: str, candidates: Sequence[str], top_n: int = 5) -> List[str]:
    """
    Calculate the Levenshtein distance between the query and each candidate
    """
    distances = [(c, similarity_ratio(query, c)) for c in candidates]

    # Sort the candidates by their Levenshtein distance to the query
    sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)

    # Get the top n candidates with the smallest Levenshtein distance
    top_candidates = [d[0] for d in sorted_distances[:top_n]]
    
    # Return the top n candidates
    return top_candidates
